"""差距判定：把需求与本机画像比对，给出四档结论。

判定原则（宁可说"待确认"，不假装满足）：
- 信息不足 → UNKNOWN
- 物理限制（卡数、显存差距过大、无 GPU）→ BLOCK
- 改配置能跑（batch 过大、精度可调）→ DEGRADE
- 装/换版本即可 → ADD
- 满足 → OK
"""

from __future__ import annotations

from .spec import DiffItem, EnvRequirement, FeasibilityReport, LocalEnvProfile, PaperEnvSpec, Verdict
from .version import Constraint, parse_version

# 显存差距超过该倍数，认为降级手段不足以弥补
VRAM_BLOCK_RATIO = 6.0


def _fmt_gb(v: float) -> str:
    return f"{v:g} GB"


def _torch_install_cmd(version: str, cuda: str | None) -> str:
    tag = f"cu{cuda.replace('.', '')}" if cuda else "cpu"
    return f"python -m pip install torch=={version} --index-url https://download.pytorch.org/whl/{tag}"


def _python_install_cmd(version: str) -> str:
    """Python 不是 pip 包，必须用环境管理器装。"""
    short = ".".join(version.split(".")[:2])
    return f"conda create -n repro python={short} -y   # 或 pyenv install {version}"


def _judge_version(req: EnvRequirement, local: str | None, pkg: str,
                   cuda_hint: str | None = None) -> tuple[Verdict, str, list[str]]:
    """版本类需求的判定。

    cuda_hint 用于 torch 这类需要区分 CPU/CUDA 构建的包，
    取本机驱动支持的 CUDA 版本，避免生成一个 CPU 版安装命令。
    """
    if not local:
        if pkg == "python":
            return Verdict.ADD, "本机无可用 Python", [_python_install_cmd(req.value)]
        return (
            Verdict.ADD,
            f"本机未安装 {pkg}",
            [_torch_install_cmd(req.value, cuda_hint) if pkg == "torch"
             else f"python -m pip install {pkg}"],
        )

    raw = req.extra.get("raw_spec") or f"{req.op}{req.value}"
    ok = Constraint(raw).satisfied_by(local)
    if ok is True:
        return Verdict.OK, f"本机 {local} 满足 {raw}", []
    if ok is None:
        return Verdict.UNKNOWN, f"无法判断本机 {local} 是否满足 {raw}", []

    if pkg == "python":
        return (
            Verdict.ADD,
            f"本机 {local} 不满足 {raw}，需换解释器",
            [_python_install_cmd(req.value)],
        )
    return (
        Verdict.ADD,
        f"本机 {local} 不满足 {raw}，需换版本",
        [_torch_install_cmd(req.value, cuda_hint) if pkg == "torch"
         else f"python -m pip install '{pkg}{raw}'"],
    )


def _judge_cuda(req: EnvRequirement, profile: LocalEnvProfile) -> tuple[Verdict, str, str, list[str]]:
    """CUDA 需求判定：分「驱动支不支持」和「torch 装没装对」两层。"""
    local = profile.torch_cuda or profile.driver_cuda or ""
    driver = profile.driver_cuda or ""

    if not profile.gpu_count:
        return Verdict.BLOCK, local or "无 GPU", "本机没有可用的 NVIDIA GPU", ["租用云 GPU 实例"]

    need = parse_version(req.value)
    drv = parse_version(driver)
    if need and drv and need > drv:
        return (
            Verdict.BLOCK,
            driver or "未知",
            f"需求 CUDA {req.value} 高于驱动支持的 {driver}",
            ["升级 NVIDIA 驱动到支持该 CUDA 的版本", "或改装基于更低 CUDA 的框架版本"],
        )

    if profile.torch_cuda:
        if parse_version(profile.torch_cuda) == need:
            return Verdict.OK, profile.torch_cuda, "torch 已按该 CUDA 版本构建", []
        return (
            Verdict.ADD,
            profile.torch_cuda,
            f"驱动支持 CUDA {driver}，可重装对应版本",
            [f"重装基于 CUDA {req.value} 的框架版本"],
        )

    return (
        Verdict.ADD,
        driver or "未知",
        f"驱动支持 CUDA {driver}，但尚未安装对应框架",
        [f"python -m pip install torch --index-url https://download.pytorch.org/whl/cu{req.value.replace('.', '')}"],
    )


def _judge_gpu_count(req: EnvRequirement, profile: LocalEnvProfile) -> tuple[Verdict, str, str, list[str]]:
    try:
        need = float(req.value)
    except ValueError:
        return Verdict.UNKNOWN, str(profile.gpu_count), "需求数值无法解析", []
    have = float(profile.gpu_count)
    if have >= need:
        return Verdict.OK, f"{int(have)} 卡", "", []
    if have == 0:
        return Verdict.BLOCK, "0 卡", "本机无 GPU", ["租用多卡云实例"]
    return (
        Verdict.BLOCK,
        f"{int(have)} 卡",
        f"需要 {int(need)} 卡，本机只有 {int(have)} 卡，卡数无法用算法弥补",
        ["租用多卡实例（如 8×A100）", "或缩小实验规模，只验证单卡可跑的部分"],
    )


def _judge_vram(req: EnvRequirement, profile: LocalEnvProfile) -> tuple[Verdict, str, str, list[str]]:
    try:
        need = float(req.value)
    except ValueError:
        return Verdict.UNKNOWN, _fmt_gb(profile.gpu_vram_gb), "需求数值无法解析", []
    have = profile.gpu_vram_gb

    if have <= 0:
        return Verdict.BLOCK, "0 GB", "本机无可用显存", ["租用 GPU 实例"]
    if have >= need:
        return Verdict.OK, _fmt_gb(have), "", []

    ratio = need / have
    remedies = [
        "减小 batch size，用梯度累积保持等效批量",
        "开启混合精度（amp / bf16）",
        "对模型用 LoRA 或冻结主干减少显存",
    ]
    if ratio >= VRAM_BLOCK_RATIO:
        return (
            Verdict.BLOCK,
            _fmt_gb(have),
            f"需要 {_fmt_gb(need)}，本机 {_fmt_gb(have)}，差距 {ratio:.1f} 倍，降级手段难以弥补",
            remedies + ["租用更大显存的 GPU"],
        )
    return (
        Verdict.DEGRADE,
        _fmt_gb(have),
        f"需要 {_fmt_gb(need)}，本机 {_fmt_gb(have)}，需降低批量或精度",
        remedies,
    )


def _judge_disk(req: EnvRequirement, profile: LocalEnvProfile) -> tuple[Verdict, str, str, list[str]]:
    try:
        need = float(req.value)
    except ValueError:
        return Verdict.UNKNOWN, _fmt_gb(profile.disk_free_gb), "需求数值无法解析", []
    have = profile.disk_free_gb
    if have <= 0:
        return Verdict.UNKNOWN, "未知", "无法读取磁盘余量", []
    if have >= need:
        return Verdict.OK, _fmt_gb(have), "", []
    return (
        Verdict.BLOCK,
        _fmt_gb(have),
        f"需要 {_fmt_gb(need)}，剩余 {_fmt_gb(have)}",
        ["清理磁盘或挂载更大分区", "把数据集放到外置盘"],
    )


def _judge_exists(req: EnvRequirement, profile: LocalEnvProfile) -> tuple[Verdict, str, str, list[str]]:
    """存在性需求，如「必须有可用 CUDA」。"""
    if req.id == "cuda.required":
        if profile.gpu_count and profile.driver_cuda:
            return Verdict.OK, f"{profile.gpu_count} 卡 / CUDA {profile.driver_cuda}", "", []
        return Verdict.BLOCK, "无", "该实验需要 NVIDIA GPU，本机未检测到", ["租用云 GPU"]
    return Verdict.UNKNOWN, "", "存在性需求类型未知", []


def _pick_torch_cuda(req: EnvRequirement, profile: LocalEnvProfile,
                     cuda_declared: str | None = None) -> str | None:
    """为指定 torch 版本挑 CUDA 构建。

    优先遵循仓库声明的 CUDA 版本（Dockerfile / environment.yml），
    只有在没有声明时才按本机驱动能力选最高可用版本。
    否则会出现「CUDA 要求 11.8、torch 却让装 cu124」的自相矛盾建议。
    """
    from .scanner import TORCH_CUDA_BUILDS

    ver = parse_version(req.value)
    if not ver or len(ver) < 2:
        return None
    builds = TORCH_CUDA_BUILDS.get((ver[0], ver[1]))
    if not builds:
        return None

    if cuda_declared and cuda_declared in builds:
        return cuda_declared

    drv = parse_version(profile.driver_cuda)
    if not drv:
        return None
    usable = [b for b in builds if (parse_version(b) or (99,)) <= drv]
    if not usable:
        return None
    return max(usable, key=lambda b: parse_version(b) or (0,))


def _judge(req: EnvRequirement, profile: LocalEnvProfile,
           cuda_declared: str | None = None) -> DiffItem:
    rid = req.id

    if rid == "python.version":
        verdict, reason, rem = _judge_version(req, profile.python_version, "python")
    elif rid == "cuda":
        verdict, local, reason, rem = _judge_cuda(req, profile)
        return DiffItem(req, local, verdict, reason, rem)
    elif rid == "cuda.required":
        verdict, local, reason, rem = _judge_exists(req, profile)
        return DiffItem(req, local, verdict, reason, rem)
    elif rid == "gpu.count":
        verdict, local, reason, rem = _judge_gpu_count(req, profile)
        return DiffItem(req, local, verdict, reason, rem)
    elif rid == "gpu.vram":
        verdict, local, reason, rem = _judge_vram(req, profile)
        return DiffItem(req, local, verdict, reason, rem)
    elif rid == "disk":
        verdict, local, reason, rem = _judge_disk(req, profile)
        return DiffItem(req, local, verdict, reason, rem)
    elif rid.endswith(".version"):
        pkg = req.extra.get("package") or rid.rsplit(".", 1)[0]
        if pkg == "python":
            verdict, reason, rem = _judge_version(req, profile.python_version, pkg)
        elif pkg == "torch":
            verdict, reason, rem = _judge_version(
                req, profile.torch_version, pkg,
                _pick_torch_cuda(req, profile, cuda_declared))
        else:
            # 未探测的框架（tensorflow / jax 等）
            return DiffItem(req, "未检测", Verdict.UNKNOWN,
                            f"尚未探测 {pkg} 的安装情况", [f"手动确认：python -c \"import {pkg}; print({pkg}.__version__)\""])
    else:
        return DiffItem(req, "", Verdict.UNKNOWN, f"未实现 {rid} 的判定规则", [])

    local_text = profile.python_version if rid == "python.version" else (profile.torch_version or "未安装")
    return DiffItem(req, local_text, verdict, reason, rem)


def diff(spec: PaperEnvSpec, profile: LocalEnvProfile) -> FeasibilityReport:
    """生成差距报告。只对「有证据来源」的需求做判定。"""
    accepted = spec.accepted
    # 仓库声明的 CUDA 版本，供 torch 选构建时对齐
    cuda_declared = next((r.value for r in accepted if r.id == "cuda"), None)
    items = [_judge(r, profile, cuda_declared) for r in accepted]
    items.sort(key=lambda i: (i.verdict.order, i.requirement.category.value, i.requirement.id))
    return FeasibilityReport(spec=spec, profile=profile, items=items)
