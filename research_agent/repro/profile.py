"""本机环境画像。

复用 envsetup 的探测原语，补充复现场景必需的维度：
磁盘余量、内存、CPU 核数、容器运行时能力。
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

from ..envsetup import probes
from .spec import LocalEnvProfile


def _memory_gb() -> float:
    """物理内存总量（GB），跨平台且不依赖 psutil。"""
    try:
        if os.name == "nt":
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                return round(stat.ullTotalPhys / 1024**3, 1)
        else:
            meminfo = Path("/proc/meminfo")
            if meminfo.is_file():
                for line in meminfo.read_text(encoding="utf-8", errors="replace").splitlines():
                    if line.startswith("MemTotal:"):
                        kb = int(line.split()[1])
                        return round(kb / 1024**2, 1)
    except Exception:  # noqa: BLE001 - 探测失败不影响整体
        pass
    return 0.0


def _disk(path: Path) -> tuple[float, float]:
    """目标路径所在分区的 (剩余 GB, 总 GB)。"""
    try:
        usage = shutil.disk_usage(str(path))
        return round(usage.free / 1024**3, 1), round(usage.total / 1024**3, 1)
    except OSError:
        return 0.0, 0.0


def collect_profile(repo_path: str | Path | None = None) -> LocalEnvProfile:
    """采集本机画像。

    repo_path 用于定位磁盘分区——数据集和权重通常放在项目所在盘，
    用项目盘而不是系统盘更能反映真实可用空间。
    """
    info = probes.platform_summary()

    interpreters = probes.python_interpreters()
    main = interpreters[0] if interpreters else {}
    py_exe = main.get("executable", sys.executable)
    py_ver = main.get("version", "")

    torch = probes.torch_info(py_exe) if py_exe else {"installed": False}
    gpus = probes.nvidia_smi_gpus()
    driver_cuda = probes.driver_cuda_version() if gpus else ""

    target = Path(repo_path).expanduser() if repo_path else Path.home()
    if not target.exists():
        target = Path.home()
    free_gb, total_gb = _disk(target)

    return LocalEnvProfile(
        python_version=py_ver,
        python_executable=py_exe,
        torch_version=torch.get("version") if torch.get("installed") else None,
        torch_cuda=torch.get("cuda_compiled") if torch.get("installed") else None,
        cuda_available=bool(torch.get("cuda_available")) if torch.get("installed") else False,
        gpu_name=gpus[0]["name"] if gpus else "",
        gpu_count=len(gpus),
        gpu_vram_gb=round(gpus[0]["memory_mib"] / 1024, 1) if gpus else 0.0,
        driver_version=gpus[0]["driver_version"] if gpus else "",
        driver_cuda=driver_cuda or "",
        disk_free_gb=free_gb,
        disk_total_gb=total_gb,
        ram_gb=_memory_gb(),
        cpu_count=os.cpu_count() or 0,
        platform=f"{info['system']} {info['release']} ({info['machine']})",
        os_name=info["system"],
        has_docker=probes.which("docker") is not None,
        has_apptainer=probes.which("apptainer") is not None or probes.which("singularity") is not None,
        has_conda=probes.which("conda") is not None,
    )
