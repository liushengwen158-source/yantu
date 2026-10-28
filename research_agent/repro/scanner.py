"""仓库扫描：从论文配套代码里提取环境需求。

只采信**可确定性推断**的信息（L2）。论文正文属于 L1，需要读 PDF，不在本模块范围。
每一条需求都带文件名和行号，没有来源的一律丢弃。

解析策略按可信度排序：
    Dockerfile / environment.yml  >  requirements.txt  >  pyproject.toml
    >  .python-version / runtime.txt  >  README 文字描述
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from .spec import Category, EnvRequirement, Evidence, EvidenceLevel, PaperEnvSpec
from .version import Constraint, parse_version

MAX_FILE_BYTES = 2_000_000

# torch 版本 -> 官方提供的 CUDA 构建（用于反查与冲突检测）
TORCH_CUDA_BUILDS: dict[tuple[int, int], list[str]] = {
    (1, 13): ["11.6", "11.7"],
    (2, 0): ["11.7", "11.8"],
    (2, 1): ["11.8", "12.1"],
    (2, 2): ["11.8", "12.1"],
    (2, 3): ["11.8", "12.1"],
    (2, 4): ["11.8", "12.1", "12.4"],
    (2, 5): ["11.8", "12.1", "12.4"],
    (2, 6): ["11.8", "12.4", "12.6"],
    (2, 7): ["11.8", "12.6", "12.8"],
    (2, 8): ["12.6", "12.8"],
}

# 常见 GPU 型号 -> 单卡显存（GB），用于把「A100」换算成显存需求
GPU_VRAM: dict[str, float] = {
    "h100": 80, "h200": 141, "a100": 40, "a800": 40, "a6000": 48,
    "a40": 48, "l40": 48, "l40s": 48, "v100": 16, "t4": 16,
    "rtx4090": 24, "rtx 4090": 24, "4090": 24,
    "rtx3090": 24, "rtx 3090": 24, "3090": 24,
    "rtx3080": 10, "rtx 3080": 10, "3080": 10,
    "2080ti": 11, "2080 ti": 11, "1080ti": 11, "1080 ti": 11,
    "titan": 24, "p100": 16, "k80": 12, "mi300": 192,
}

_FRAMEWORKS = ("torch", "tensorflow", "jax", "mindspore", "paddlepaddle", "mxnet")


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------

def _read(path: Path) -> str:
    """安全读取：编码错、文件过大、无权限都不抛异常。"""
    try:
        if path.stat().st_size > MAX_FILE_BYTES:
            return ""
        raw = path.read_bytes()
    except OSError:
        return ""
    for enc in ("utf-8", "utf-8-sig", "gbk", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", "replace")


def _norm(name: str) -> str:
    """包名规范化：PEP 503，Torch 与 torch 视为同一个。"""
    return re.sub(r"[-_.]+", "-", name.strip().lower())


def _parse_req_line(line: str) -> tuple[str, str] | None:
    """解析一行依赖声明，返回 (包名, 约束串)。

    跳过注释、空行、-r/-e/--index-url、URL、本地路径。
    """
    s = line.strip()
    if not s or s.startswith("#"):
        return None
    if s.startswith(("-", "git+", "http://", "https://", ".", "/")):
        return None
    s = s.split(" #")[0].strip()
    if not s:
        return None
    m = re.match(r"^([A-Za-z0-9][A-Za-z0-9._-]*)\s*(\[[^\]]*\])?\s*(.*)$", s)
    if not m:
        return None
    name = _norm(m.group(1))
    rest = (m.group(3) or "").strip()
    # 去掉环境标记 ; python_version<"3.9"
    rest = rest.split(";")[0].strip()
    return name, rest


# ---------------------------------------------------------------------------
# 各文件解析
# ---------------------------------------------------------------------------

def _scan_requirements(root: Path, spec: PaperEnvSpec) -> list[EnvRequirement]:
    out: list[EnvRequirement] = []
    patterns = ["requirements.txt", "requirements/*.txt", "requirements*.txt"]

    files: list[Path] = []
    for pat in patterns:
        files.extend(sorted(root.glob(pat)))
    seen: set[Path] = set()
    for f in files:
        if f in seen or not f.is_file():
            continue
        seen.add(f)
        text = _read(f)
        if not text:
            continue
        spec.scanned_files.append(str(f.relative_to(root)))
        for idx, line in enumerate(text.splitlines(), 1):
            parsed = _parse_req_line(line)
            if not parsed:
                continue
            name, constraint = parsed
            if not constraint:
                continue
            cat = Category.FRAMEWORK if name in _FRAMEWORKS else None
            if cat is None:
                continue
            out.append(EnvRequirement(
                id=f"{name}.version",
                category=cat,
                op=_op_of(constraint),
                value=constraint.lstrip("=<>!~=").strip() or constraint,
                why=f"{f.name} 声明了 {name} 的版本",
                evidence=Evidence(EvidenceLevel.L2, str(f.relative_to(root)), idx, line.strip()),
                extra={"raw_spec": constraint, "package": name},
            ))
    return out


def _op_of(constraint: str) -> str:
    """从约束串取首个操作符，用于展示。"""
    for op in ("==", ">=", "<=", "!=", "~=", ">", "<"):
        if constraint.startswith(op):
            return op
    return "=="


def _scan_environment_yml(root: Path, spec: PaperEnvSpec) -> list[EnvRequirement]:
    out: list[EnvRequirement] = []
    for f in sorted(root.glob("environment*.y*ml")) + sorted(root.glob("*.y*ml")):
        if not f.is_file():
            continue
        if f.name in ("environment.yml", "environment.yaml") or f.name.startswith("environment"):
            text = _read(f)
            if not text:
                continue
            spec.scanned_files.append(str(f.relative_to(root)))
            for idx, line in enumerate(text.splitlines(), 1):
                # - python=3.9   /  - torch==1.13.1  /  - pip:  （子列表里同样处理）
                m = re.match(r"^\s*-\s*([A-Za-z0-9][A-Za-z0-9._-]*)\s*(=+|>=|<=|~=|>|<)\s*([^\s#]+)", line)
                if not m:
                    continue
                name = _norm(m.group(1))
                ver = m.group(3).strip().strip("'\"")
                if name == "python":
                    out.append(EnvRequirement(
                        id="python.version", category=Category.PYTHON,
                        op="==", value=ver,
                        why=f"{f.name} 指定了 Python 版本",
                        evidence=Evidence(EvidenceLevel.L2, str(f.relative_to(root)), idx, line.strip()),
                    ))
                elif name in _FRAMEWORKS:
                    out.append(EnvRequirement(
                        id=f"{name}.version", category=Category.FRAMEWORK,
                        op=m.group(2) if m.group(2) != "=" else "==", value=ver,
                        why=f"{f.name} 声明了 {name} 的版本",
                        evidence=Evidence(EvidenceLevel.L2, str(f.relative_to(root)), idx, line.strip()),
                        extra={"package": name, "raw_spec": f"{m.group(2)}{ver}"},
                    ))
                elif name in ("cudatoolkit", "cuda-version", "cuda"):
                    out.append(EnvRequirement(
                        id="cuda", category=Category.CUDA,
                        op=m.group(2) if m.group(2) != "=" else "==", value=ver,
                        why=f"{f.name} 指定了 CUDA 版本",
                        evidence=Evidence(EvidenceLevel.L2, str(f.relative_to(root)), idx, line.strip()),
                    ))
    return out


def _scan_dockerfile(root: Path, spec: PaperEnvSpec) -> list[EnvRequirement]:
    out: list[EnvRequirement] = []
    candidates = [p for p in root.glob("**/Dockerfile*") if p.is_file()]
    # 只取根目录和一级子目录，避免扫到依赖里的
    candidates = [p for p in candidates if len(p.relative_to(root).parts) <= 2]

    for f in candidates[:5]:
        text = _read(f)
        if not text:
            continue
        rel = str(f.relative_to(root))
        if rel not in spec.scanned_files:
            spec.scanned_files.append(rel)

        for idx, line in enumerate(text.splitlines(), 1):
            s = line.strip()
            if not s or s.startswith("#"):
                continue

            # FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu20.04
            m = re.match(r"^FROM\s+(\S+)", s, re.IGNORECASE)
            if m:
                image = m.group(1)
                mc = re.search(r"cuda[:\-]?(\d+)\.(\d+)", image, re.IGNORECASE)
                if mc:
                    out.append(EnvRequirement(
                        id="cuda", category=Category.CUDA, op="==",
                        value=f"{mc.group(1)}.{mc.group(2)}",
                        why="基础镜像决定了运行时 CUDA 版本",
                        evidence=Evidence(EvidenceLevel.L2, rel, idx, s),
                    ))
                mp = re.search(r"python[:\-]?(\d+)\.(\d+)", image, re.IGNORECASE)
                if mp:
                    out.append(EnvRequirement(
                        id="python.version", category=Category.PYTHON, op="==",
                        value=f"{mp.group(1)}.{mp.group(2)}",
                        why="基础镜像决定了 Python 版本",
                        evidence=Evidence(EvidenceLevel.L2, rel, idx, s),
                    ))
                if "cuda" in image.lower():
                    out.append(EnvRequirement(
                        id="cuda.required", category=Category.SYSTEM, op="exists",
                        value="CUDA", hard=True,
                        why="镜像基于 CUDA 基础镜像，必须能访问 NVIDIA 设备",
                        evidence=Evidence(EvidenceLevel.L2, rel, idx, s),
                    ))
                continue

            # ENV CUDA_VERSION=11.7 / pip install torch==1.13
            mi = re.search(r"pip3?\s+install\s+([^\n&|]+)", s)
            if mi:
                for token in mi.group(1).split():
                    parsed = _parse_req_line(token)
                    if not parsed:
                        continue
                    name, constraint = parsed
                    if name in _FRAMEWORKS and constraint:
                        out.append(EnvRequirement(
                            id=f"{name}.version", category=Category.FRAMEWORK,
                            op=_op_of(constraint),
                            value=constraint.lstrip("=<>!~="), 
                            why="Dockerfile 中的安装命令指定了框架版本",
                            evidence=Evidence(EvidenceLevel.L2, rel, idx, s[:200]),
                            extra={"package": name, "raw_spec": constraint},
                        ))
    return out


def _scan_python_version(root: Path, spec: PaperEnvSpec) -> list[EnvRequirement]:
    out: list[EnvRequirement] = []
    for name, pattern in (
        (".python-version", r"(\d+\.\d+(?:\.\d+)?)"),
        ("runtime.txt", r"python-(\d+\.\d+(?:\.\d+)?)"),
    ):
        f = root / name
        if not f.is_file():
            continue
        text = _read(f).strip()
        if not text:
            continue
        m = re.search(pattern, text)
        if m:
            spec.scanned_files.append(name)
            out.append(EnvRequirement(
                id="python.version", category=Category.PYTHON, op="==",
                value=m.group(1),
                why=f"{name} 固定了 Python 版本",
                evidence=Evidence(EvidenceLevel.L2, name, 1, text[:100]),
            ))
    return out


def _scan_pyproject(root: Path, spec: PaperEnvSpec) -> list[EnvRequirement]:
    out: list[EnvRequirement] = []
    f = root / "pyproject.toml"
    if not f.is_file():
        return out
    text = _read(f)
    if not text:
        return out
    spec.scanned_files.append("pyproject.toml")

    for idx, line in enumerate(text.splitlines(), 1):
        if "requires-python" in line:
            m = re.search(r'requires-python\s*=\s*["\']([^"\']+)["\']', line)
            if m:
                out.append(EnvRequirement(
                    id="python.version", category=Category.PYTHON,
                    op=_op_of(m.group(1).strip()),
                    value=m.group(1).lstrip("=<>!~= ").strip(),
                    why="pyproject.toml 声明了 requires-python",
                    evidence=Evidence(EvidenceLevel.L2, "pyproject.toml", idx, line.strip()),
                    extra={"raw_spec": m.group(1)},
                ))
    return out


def _scan_readme(root: Path, spec: PaperEnvSpec) -> list[EnvRequirement]:
    """从 README 提取 GPU 型号/数量、显存、显式声明的框架版本。"""
    out: list[EnvRequirement] = []
    readme = None
    for name in ("README.md", "README.rst", "README.txt", "README"):
        p = root / name
        if p.is_file():
            readme = p
            break
    if readme is None:
        return out

    text = _read(readme)
    if not text:
        return out
    rel = readme.name
    if rel not in spec.scanned_files:
        spec.scanned_files.append(rel)
    lines = text.splitlines()

    # 8×A100 / 8 x A100 / 8 A100 GPUs
    count_pat = re.compile(r"(\d+)\s*[×xX\*]\s*(H100|H200|A100|A800|A6000|A40|L40S?|V100|T4|RTX\s?\d{4})", re.IGNORECASE)
    # A100 80GB / 80 GB GPU
    vram_pat = re.compile(r"(H100|H200|A100|A800|A6000|A40|L40S?|V100|T4|RTX\s?\d{4})\D{0,12}(\d{2,3})\s*GB", re.IGNORECASE)

    for idx, line in enumerate(lines, 1):
        low = line.lower()
        if not any(k in low for k in ("gpu", "a100", "h100", "v100", "显卡", "device")):
            continue

        mc = count_pat.search(line)
        if mc:
            out.append(EnvRequirement(
                id="gpu.count", category=Category.HARDWARE, op=">=",
                value=mc.group(1), unit="卡", hard=True,
                why="README 明确写了训练使用的 GPU 数量",
                evidence=Evidence(EvidenceLevel.L2, rel, idx, line.strip()[:160]),
            ))

        mv = vram_pat.search(line)
        if mv:
            out.append(EnvRequirement(
                id="gpu.vram", category=Category.HARDWARE, op=">=",
                value=mv.group(2), unit="GB", hard=True,
                why=f"README 写明了 {mv.group(1)} 的显存需求",
                evidence=Evidence(EvidenceLevel.L2, rel, idx, line.strip()[:160]),
            ))
        else:
            # 只给了型号，用型号映射出显存（此时是推断，标 L3）
            mm = re.search(r"(H100|H200|A100|A800|A6000|A40|L40S?|V100|T4|RTX\s?\d{4})", line, re.IGNORECASE)
            if mm:
                key = re.sub(r"\s+", "", mm.group(1).lower())
                vram = GPU_VRAM.get(key) or GPU_VRAM.get(mm.group(1).lower())
                if vram:
                    out.append(EnvRequirement(
                        id="gpu.vram", category=Category.HARDWARE, op=">=",
                        value=str(int(vram)), unit="GB", hard=True,
                        why=f"README 提到 {mm.group(1)}，按其标准显存折算（推算值）",
                        evidence=Evidence(EvidenceLevel.L3, rel, idx, line.strip()[:160]),
                    ))
    return out


# ---------------------------------------------------------------------------
# 合并与冲突
# ---------------------------------------------------------------------------

def _merge(reqs: list[EnvRequirement], spec: PaperEnvSpec) -> list[EnvRequirement]:
    """同一需求 id 出现多次时合并约束，保留最可信的证据。"""
    merged: dict[str, EnvRequirement] = {}
    for r in reqs:
        old = merged.get(r.id)
        if old is None:
            merged[r.id] = r
            continue
        # 合并约束串，形成交集
        old_spec = old.extra.get("raw_spec") or f"{old.op}{old.value}"
        new_spec = r.extra.get("raw_spec") or f"{r.op}{r.value}"
        if old.op == "exists" or r.op == "exists":
            continue
        combined = f"{old_spec},{new_spec}"
        if _is_contradictory(old_spec, new_spec):
            spec.warnings.append(
                f"{r.id} 存在冲突声明：{old.evidence.describe()} 要求 {old_spec}，"
                f"{r.evidence.describe()} 要求 {new_spec}"
            )
        # 证据取级别更高（L1>L2>L3）的那条
        keep = r if _level_rank(r) < _level_rank(old) else old
        keep.extra["raw_spec"] = combined
        merged[r.id] = keep
    return list(merged.values())


def _level_rank(r: EnvRequirement) -> int:
    return {EvidenceLevel.L1: 0, EvidenceLevel.L2: 1, EvidenceLevel.L3: 2}[r.evidence.level]


def _is_contradictory(a: str, b: str) -> bool:
    """两个 == 约束指向不同值即判定矛盾。"""
    if not (a.startswith("==") and b.startswith("==")):
        return False
    return parse_version(a[2:]) != parse_version(b[2:])


def _detect_cross_conflicts(reqs: list[EnvRequirement], spec: PaperEnvSpec) -> None:
    """跨字段冲突：torch 版本与 Dockerfile 的 CUDA 版本不匹配。"""
    by_id = {r.id: r for r in reqs}
    torch_req = by_id.get("torch.version")
    cuda_req = by_id.get("cuda")
    if not (torch_req and cuda_req):
        return

    raw = torch_req.extra.get("raw_spec", "")
    first = raw.split(",")[0].strip()
    if not first.startswith("=="):
        return
    ver_text = first[2:]
    ver = parse_version(ver_text)
    if not ver or len(ver) < 2:
        return
    builds = TORCH_CUDA_BUILDS.get((ver[0], ver[1]))
    if not builds:
        return

    cuda_ver = parse_version(cuda_req.value)
    if not cuda_ver or len(cuda_ver) < 2:
        return
    target = f"{cuda_ver[0]}.{cuda_ver[1]}"
    if target not in builds:
        spec.warnings.append(
            f"torch {ver_text} 官方只提供 CUDA {'/'.join(builds)} 构建，"
            f"但 {cuda_req.evidence.describe()} 指定了 CUDA {target}"
        )


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

def scan_repo(repo_path: str | Path) -> PaperEnvSpec:
    """扫描仓库，产出 PaperEnvSpec。"""
    root = Path(repo_path).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f"仓库目录不存在：{root}")

    spec = PaperEnvSpec(repo_path=str(root))

    collected: list[EnvRequirement] = []
    collected += _scan_dockerfile(root, spec)
    collected += _scan_environment_yml(root, spec)
    collected += _scan_requirements(root, spec)
    collected += _scan_pyproject(root, spec)
    collected += _scan_python_version(root, spec)
    collected += _scan_readme(root, spec)

    spec.requirements = _merge(collected, spec)
    _detect_cross_conflicts(spec.requirements, spec)

    # 丢弃无证据来源的需求
    for r in spec.rejected:
        spec.warnings.append(f"需求 {r.id} 缺少来源，已丢弃")

    if not spec.scanned_files:
        spec.warnings.append("没有找到任何可识别的依赖声明文件")

    return spec
