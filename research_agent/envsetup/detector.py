"""环境检测器：把探测到的事实判定成「阻塞 / 注意 / 建议 / 正常」。

对外只有两个入口：
- ``detect()``           跑完整检测，返回 EnvReport
- ``load_knowledge()``   读知识库条目，供 advisor / agent 使用
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

from . import probes
from .models import CheckResult, EnvReport, Status

KB_PATH = Path(__file__).resolve().parent.parent / "knowledge" / "env_kb.json"

# openJiuwen agent-core 支持的 Python 区间
PY_MIN = (3, 11)
PY_MAX = (3, 13)


def load_knowledge(path: Path | None = None) -> dict:
    """读取环境配置知识库，返回 {entry_id: entry}。"""
    p = path or KB_PATH
    data = json.loads(p.read_text(encoding="utf-8"))
    return {e["id"]: e for e in data.get("entries", [])}


def _ver_tuple(text: str) -> tuple[int, ...]:
    parts = []
    for seg in str(text).replace("-", ".").split("."):
        digits = "".join(ch for ch in seg if ch.isdigit())
        if digits == "":
            break
        parts.append(int(digits))
    return tuple(parts) if parts else (0,)


def _same_env(pip_location: str, python_prefix: str) -> bool:
    if not pip_location or not python_prefix:
        return True  # 信息不足时不做负面判定
    a = os.path.normcase(os.path.abspath(pip_location))
    b = os.path.normcase(os.path.abspath(python_prefix))
    return a.startswith(b)


def detect(check_network: bool = True, check_torch: bool = True) -> EnvReport:
    """跑一次完整检测。

    check_network  是否探测 GitHub 可达性（会多花几秒，默认开）
    check_torch    是否在解释器里 import torch（首次可能 5-30 秒）
    """
    info = probes.platform_summary()
    results: list[CheckResult] = []

    # ---------------- Python -------------------------------------------
    interpreters = probes.python_interpreters()
    main = interpreters[0] if interpreters else None

    if main is None:
        results.append(CheckResult(
            id="python.installed", domain="python", name="Python 解释器",
            status=Status.BLOCKER, value="未找到",
            message="命令行找不到可用的 Python 解释器",
            kb_id="python.missing",
        ))
        report = EnvReport(
            platform=f"{info['system']} {info['release']} ({info['machine']})",
            python_executable="",
            python_version="",
            results=results,
            generated_at=datetime.now().astimezone().isoformat(timespec="seconds"),
        )
        return report

    py_exe = main["executable"]
    py_ver = main["version"]
    # 只比较 major.minor：3.13.4 属于 3.13，不能判成「高于 3.13」
    vt = (main["major"], main["minor"])

    if PY_MIN <= vt <= PY_MAX:
        py_status, py_msg = Status.OK, "版本在推荐区间内"
    elif vt < PY_MIN:
        py_status = Status.BLOCKER if vt < (3, 10) else Status.WARN
        py_msg = f"低于推荐下限 {PY_MIN[0]}.{PY_MIN[1]}，部分科研库会装不上"
    else:
        py_status = Status.WARN
        py_msg = f"高于 {PY_MAX[0]}.{PY_MAX[1]}，科学计算栈的轮子覆盖可能不全"

    results.append(CheckResult(
        id="python.version", domain="python", name="Python 版本",
        status=py_status, value=py_ver, message=py_msg,
        kb_id=None if py_status is Status.OK else "python.version.unsupported",
        detail={"executable": py_exe, "via": main.get("via"),
                "abiflags": main.get("abiflags", "")},
    ))

    # Windows 上没有 sys.abiflags，用是否启用 GIL 判断自由线程构建
    if main.get("gil_enabled") is False:
        results.append(CheckResult(
            id="python.freethreaded", domain="python", name="解释器构建",
            status=Status.WARN, value="free-threaded（GIL 已关闭）",
            message="自由线程构建，C 扩展轮子支持不全，不建议作为科研默认解释器",
            kb_id="python.version.freethreaded",
        ))
    else:
        results.append(CheckResult(
            id="python.freethreaded", domain="python", name="解释器构建",
            status=Status.OK, value="standard",
            message="标准 GIL 构建，兼容性最好",
        ))

    distinct = {os.path.normcase(os.path.abspath(i["executable"])): i for i in interpreters}
    others = [i for k, i in distinct.items()
              if os.path.normcase(os.path.abspath(i["executable"]))
              != os.path.normcase(os.path.abspath(py_exe))]
    if len(distinct) > 1:
        # 其他解释器里若有自由线程版，重点提示——这几乎必然导致装包踩坑
        ft_others = [i for i in others if i.get("gil_enabled") is False]
        msg = "装了多个 Python，注意 pip 与 python 可能不是同一套"
        if ft_others:
            msg += f"；其中 {ft_others[0]['executable']} 是 free-threaded 版，装科学计算库易失败"
        results.append(CheckResult(
            id="python.multi_install", domain="python", name="多解释器并存",
            status=Status.WARN, value=f"{len(distinct)} 个",
            message=msg, kb_id="python.multi_install",
            detail={"interpreters": [
                {"exe": i["executable"], "version": i["version"],
                 "via": i.get("via"), "gil": i.get("gil_enabled")}
                for i in distinct.values()
            ]},
        ))

    # ---------------- pip ----------------------------------------------
    pip = probes.pip_info(py_exe)
    if not pip.get("available"):
        results.append(CheckResult(
            id="pip.available", domain="python", name="pip",
            status=Status.BLOCKER, value="不可用",
            message="无法安装第三方库", kb_id="pip.missing",
        ))
    else:
        results.append(CheckResult(
            id="pip.available", domain="python", name="pip",
            status=Status.OK, value=pip["version"],
            message="可用", detail={"location": pip.get("location", "")},
        ))
        if not _same_env(pip.get("location", ""), main.get("prefix", "")):
            results.append(CheckResult(
                id="pip.mismatch", domain="python", name="pip 与解释器一致",
                status=Status.WARN, value="不一致",
                message="pip 装包的位置和当前 python 不是同一环境，会出现装了却 import 不到",
                kb_id="pip.mismatch",
                detail={"pip_location": pip.get("location", ""),
                        "python_prefix": main.get("prefix", "")},
            ))

        conf = probes.pip_config(py_exe)
        index = next((v for k, v in conf.items() if k.endswith("index-url")), "")
        if not index:
            results.append(CheckResult(
                id="pip.index", domain="python", name="pip 镜像源",
                status=Status.WARN, value="默认 PyPI",
                message="国内直连 PyPI 下载大包很慢，建议配国内镜像",
                kb_id="pip.index.default",
            ))
        else:
            results.append(CheckResult(
                id="pip.index", domain="python", name="pip 镜像源",
                status=Status.OK, value=index, message="已配置加速源",
            ))

    if not probes.has_venv_module(py_exe):
        results.append(CheckResult(
            id="venv.available", domain="python", name="虚拟环境支持",
            status=Status.WARN, value="不可用",
            message="无法创建 venv，容易污染系统 Python", kb_id="venv.missing",
        ))
    else:
        results.append(CheckResult(
            id="venv.available", domain="python", name="虚拟环境支持",
            status=Status.OK, value="可用",
        ))

    conda_exe = probes.which("conda") or probes.which("mamba")
    if not conda_exe:
        results.append(CheckResult(
            id="conda.available", domain="python", name="conda",
            status=Status.INFO, value="未安装",
            message="非必需项；需要多 CUDA 版本共存或复现 environment.yml 时再装",
            kb_id="conda.missing",
        ))
    else:
        results.append(CheckResult(
            id="conda.available", domain="python", name="conda",
            status=Status.OK, value=probes.run([conda_exe, "--version"], 15).out,
        ))

    # ---------------- Git ----------------------------------------------
    git_ver = probes.git_version()
    if not git_ver:
        results.append(CheckResult(
            id="git.installed", domain="git", name="Git",
            status=Status.BLOCKER, value="未安装",
            message="无法进行版本管理与拉取开源仓库", kb_id="git.missing",
        ))
    else:
        results.append(CheckResult(
            id="git.installed", domain="git", name="Git",
            status=Status.OK, value=git_ver,
        ))
        name = probes.git_global("user.name")
        email = probes.git_global("user.email")
        if not name or not email:
            missing = [k for k, v in (("user.name", name), ("user.email", email)) if not v]
            results.append(CheckResult(
                id="git.identity", domain="git", name="Git 身份",
                status=Status.WARN, value="未配置",
                message=f"缺少 {', '.join(missing)}，无法正确提交",
                kb_id="git.identity.missing", detail={"missing": missing},
            ))
        else:
            results.append(CheckResult(
                id="git.identity", domain="git", name="Git 身份",
                status=Status.OK, value=f"{name} <{email}>",
            ))

        keys = probes.ssh_public_keys()
        if keys:
            results.append(CheckResult(
                id="git.ssh", domain="git", name="SSH 密钥",
                status=Status.OK, value=f"{len(keys)} 个公钥",
                detail={"keys": keys},
            ))
        else:
            results.append(CheckResult(
                id="git.ssh", domain="git", name="SSH 密钥",
                status=Status.INFO, value="未发现",
                message="用 SSH 拉私有仓库免密，后续连 GPU 服务器也会用到",
                kb_id="git.ssh.missing",
            ))

        if check_network:
            reachable = probes.git_remote_reachable(
                "https://gitcode.com/openJiuwen/agent-core.git", timeout=20)
            results.append(CheckResult(
                id="git.network", domain="git", name="代码托管连通性",
                status=Status.OK if reachable else Status.WARN,
                value="可达" if reachable else "不可达",
                message="" if reachable else "拉取开源仓库可能超时，建议走国内镜像或配代理",
                kb_id=None if reachable else "git.github.unreachable",
                detail={"probe": "git ls-remote gitcode/openJiuwen"},
            ))

    # ---------------- GPU / 深度学习 -----------------------------------
    gpus = probes.nvidia_smi_gpus()
    drv_cuda = probes.driver_cuda_version() if gpus else None

    if not gpus:
        results.append(CheckResult(
            id="gpu.nvidia", domain="gpu", name="NVIDIA GPU",
            status=Status.INFO, value="未检测到",
            message="无卡时可先用 CPU 跑小实验，或用云 GPU",
            kb_id="gpu.nvidia_smi.missing",
        ))
    else:
        g0 = gpus[0]
        vram_gb = round(g0["memory_mib"] / 1024, 1)
        results.append(CheckResult(
            id="gpu.nvidia", domain="gpu", name="NVIDIA GPU",
            status=Status.OK, value=f"{g0['name']} / {vram_gb}GB",
            message=f"驱动 {g0['driver_version']}，支持 CUDA {drv_cuda or '未知'}",
            detail={"gpus": gpus, "driver_cuda": drv_cuda},
        ))

        drv = _ver_tuple(g0["driver_version"])
        if drv and drv < (550, 0):
            results.append(CheckResult(
                id="gpu.driver", domain="gpu", name="GPU 驱动",
                status=Status.WARN, value=g0["driver_version"],
                message=f"驱动较旧，最高支持 CUDA {drv_cuda}，装新版 torch 可能失败",
                kb_id="gpu.driver.old",
            ))
        else:
            results.append(CheckResult(
                id="gpu.driver", domain="gpu", name="GPU 驱动",
                status=Status.OK, value=g0["driver_version"],
            ))

        if vram_gb <= 8:
            results.append(CheckResult(
                id="gpu.vram", domain="gpu", name="显存",
                status=Status.INFO, value=f"{vram_gb} GB",
                message="入门任务够用；训练大模型会 OOM，需要时减小 batch 或上云",
                kb_id="gpu.vram.small",
            ))
        else:
            results.append(CheckResult(
                id="gpu.vram", domain="gpu", name="显存",
                status=Status.OK, value=f"{vram_gb} GB",
            ))

    cuda_tk = probes.nvcc_version()
    results.append(CheckResult(
        id="gpu.cuda_toolkit", domain="gpu", name="CUDA Toolkit (nvcc)",
        status=Status.OK if cuda_tk else Status.INFO,
        value=cuda_tk or "未安装",
        message="" if cuda_tk else "用 pip 版 torch 时不需要；只有编译自定义算子才要",
        kb_id=None if cuda_tk else "gpu.cuda_toolkit.missing",
    ))

    torch = probes.torch_info(py_exe) if check_torch else {"installed": False, "error": "跳过"}
    if not torch.get("installed"):
        results.append(CheckResult(
            id="torch.installed", domain="gpu", name="PyTorch",
            status=Status.INFO, value="未安装",
            message="深度学习科研的第一步大件",
            kb_id="gpu.torch.missing",
            detail={"error": torch.get("error", "")},
        ))
    else:
        compiled = torch.get("cuda_compiled")
        available = bool(torch.get("cuda_available"))
        if compiled is None:
            results.append(CheckResult(
                id="torch.installed", domain="gpu", name="PyTorch",
                status=Status.WARN, value=f"{torch.get('version')} (CPU)",
                message="装的是 CPU 版，GPU 完全用不上",
                kb_id="gpu.torch.cpu_only",
                detail={"version": torch.get("version")},
            ))
        elif not available:
            worse = False
            if drv_cuda:
                worse = _ver_tuple(compiled) > _ver_tuple(drv_cuda)
            results.append(CheckResult(
                id="torch.installed", domain="gpu", name="PyTorch",
                status=Status.WARN, value=f"{torch.get('version')} (cu{compiled})",
                message=("torch 的 CUDA 版本高于驱动支持，建议升级驱动"
                         if worse else "torch 是 CUDA 版但当前不可用，通常是驱动问题"),
                kb_id="gpu.torch.cuda_mismatch",
                detail={"cuda_compiled": compiled, "driver_cuda": drv_cuda},
            ))
        else:
            results.append(CheckResult(
                id="torch.installed", domain="gpu", name="PyTorch",
                status=Status.OK, value=f"{torch.get('version')} (cu{compiled})",
                message=f"CUDA 可用，{torch.get('device_count', 1)} 个设备",
            ))

    results.sort(key=lambda r: (r.status.order, r.domain, r.id))

    return EnvReport(
        platform=f"{info['system']} {info['release']} ({info['machine']})",
        python_executable=py_exe,
        python_version=py_ver,
        results=results,
        generated_at=datetime.now().astimezone().isoformat(timespec="seconds"),
    )
