"""探测原语：只做「观察」，绝不修改系统。

上层 detector 只调用这里的函数拿事实，判定逻辑放在 detector 里，
这样「怎么测」和「怎么判断」可以分开演进。
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


DEFAULT_TIMEOUT = 15


@dataclass
class ProcResult:
    cmd: str
    code: int
    stdout: str
    stderr: str
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        return self.code == 0 and not self.timed_out

    @property
    def out(self) -> str:
        """合并 stdout/stderr 并去首尾空白，方便直接解析。"""
        return (self.stdout or self.stderr).strip()


def _decode(raw: bytes) -> str:
    """中文 Windows 控制台常输出 GBK，这里依次尝试多种编码。"""
    for enc in ("utf-8", sys.getfilesystemencoding() or "utf-8", "gbk", "latin-1"):
        try:
            return raw.decode(enc)
        except (UnicodeDecodeError, LookupError):
            continue
    return raw.decode("utf-8", "replace")


def _wrap_windows(args: Sequence[str]) -> list[str]:
    """Windows 下 .bat/.cmd 不能直接 CreateProcess，需要 cmd /c 包裹。"""
    if os.name != "nt":
        return list(args)
    exe = shutil.which(args[0])
    if exe and exe.lower().endswith((".bat", ".cmd")):
        return ["cmd", "/c", *args]
    return list(args)


def _kill_tree(proc: subprocess.Popen) -> None:
    """杀掉整个进程树。

    Windows 上 subprocess 的 timeout 只杀直接子进程，git 派生的
    git-remote-https 会继续持有管道，导致 communicate 永久阻塞。
    """
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
            capture_output=True,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    else:
        proc.kill()


def run(args: Sequence[str], timeout: int = DEFAULT_TIMEOUT) -> ProcResult:
    """执行命令并捕获输出；任何异常都转成 ProcResult，不向上抛。

    注意：不使用 subprocess.run(timeout=...)，因为它在 Windows 上无法结束
    孙进程，遇到网络类命令会永久挂住。
    """
    cmd = " ".join(args)
    try:
        proc = subprocess.Popen(
            _wrap_windows(args),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0,
        )
    except FileNotFoundError:
        return ProcResult(cmd, 127, "", f"命令不存在：{args[0]}")
    except Exception as exc:  # noqa: BLE001
        return ProcResult(cmd, -1, "", f"{type(exc).__name__}: {exc}")

    try:
        out, err = proc.communicate(timeout=timeout)
        return ProcResult(cmd, proc.returncode, _decode(out), _decode(err))
    except subprocess.TimeoutExpired:
        _kill_tree(proc)
        try:
            out, err = proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            out, err = b"", b""
        return ProcResult(cmd, -1, _decode(out),
                          _decode(err) + f"\n超时（>{timeout}s）", timed_out=True)


def which(name: str) -> str | None:
    return shutil.which(name)


# ---------------------------------------------------------------------------
# Python
# ---------------------------------------------------------------------------

_PY_PROBE = (
    "import json,sys;"
    "print(json.dumps({"
    "'executable':sys.executable,"
    "'version':'.'.join(map(str,sys.version_info[:3])),"
    "'major':sys.version_info[0],'minor':sys.version_info[1],"
    "'abiflags':getattr(sys,'abiflags',''),"
    "'gil_enabled':(sys._is_gil_enabled() if hasattr(sys,'_is_gil_enabled') else True),"
    "'prefix':sys.prefix,"
    "'base_prefix':getattr(sys,'base_prefix',sys.prefix)"
    "}))"
)


def python_interpreters() -> list[dict]:
    """探测可用的 Python 解释器。

    返回每个解释器的 executable/version/abiflags 等。探测多个入口
    （python、python3、py -3.x）是因为 Windows 上它们可能不是同一个。
    """
    found: list[dict] = []
    seen: set[str] = set()

    # 顺序即优先级：用户敲什么就用什么当「主解释器」。
    # ``py -3`` 放最后，因为 Windows 上它可能指向与 ``python`` 不同的构建。
    candidates: list[list[str]] = [["python"], ["python3"]]
    if os.name == "nt":
        candidates.append(["py", "-3"])

    for cand in candidates:
        res = run([*cand, "-c", _PY_PROBE], timeout=20)
        if not res.ok:
            continue
        try:
            info = json.loads(res.out.splitlines()[-1])
        except (json.JSONDecodeError, IndexError):
            continue
        key = os.path.normcase(os.path.abspath(info.get("executable", "")))
        if key in seen:
            continue
        seen.add(key)
        info["via"] = " ".join(cand)
        found.append(info)
    return found


def py_launcher_list() -> list[str]:
    """Windows 下 ``py -0p`` 列出所有注册的解释器。"""
    if os.name != "nt":
        return []
    res = run(["py", "-0p"], timeout=15)
    if not res.ok:
        return []
    return [line.strip() for line in res.out.splitlines() if line.strip()]


def pip_info(python_exe: str) -> dict:
    """探测给定解释器的 pip 版本与所在环境。"""
    res = run([python_exe, "-m", "pip", "--version"], timeout=30)
    if not res.ok:
        return {"available": False, "raw": res.out}
    raw = res.out.splitlines()[0]
    # 形如：pip 25.1.1 from D:\PYTHON\Lib\site-packages\pip (python 3.13)
    version = raw.split()[1] if len(raw.split()) > 1 else ""
    from_path = ""
    if " from " in raw:
        from_path = raw.split(" from ", 1)[1].split(" (")[0]
    return {"available": True, "version": version, "location": from_path, "raw": raw}


def pip_config(python_exe: str) -> dict:
    """读取 pip 生效的配置（镜像源、超时等）。"""
    res = run([python_exe, "-m", "pip", "config", "list"], timeout=30)
    conf: dict[str, str] = {}
    if res.ok:
        for line in res.out.splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                conf[k.strip()] = v.strip().strip("'\"")
    return conf


def has_venv_module(python_exe: str) -> bool:
    return run([python_exe, "-m", "venv", "--help"], timeout=30).ok


# ---------------------------------------------------------------------------
# Git
# ---------------------------------------------------------------------------

def git_version() -> str | None:
    res = run(["git", "--version"], timeout=15)
    if not res.ok:
        return None
    # git version 2.53.0.windows.1
    parts = res.out.split()
    return parts[2] if len(parts) >= 3 else res.out


def git_global(key: str) -> str | None:
    res = run(["git", "config", "--global", "--get", key], timeout=15)
    val = res.out.strip() if res.ok else ""
    return val or None


def ssh_public_keys() -> list[str]:
    """返回 ~/.ssh 下的公钥文件路径。"""
    ssh_dir = Path.home() / ".ssh"
    if not ssh_dir.is_dir():
        return []
    return sorted(str(p) for p in ssh_dir.glob("*.pub"))


def git_remote_reachable(url: str, timeout: int = 20) -> bool:
    """轻量探测仓库可达性（只列分支，不下载对象）。"""
    return run(["git", "ls-remote", "--heads", url], timeout=timeout).ok


# ---------------------------------------------------------------------------
# GPU / 深度学习
# ---------------------------------------------------------------------------

def nvidia_smi_gpus() -> list[dict]:
    """解析 nvidia-smi，返回每个 GPU 的名称/驱动/显存(MiB)。"""
    res = run(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
         "--format=csv,noheader"],
        timeout=20,
    )
    if not res.ok:
        return []
    gpus = []
    for line in res.out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        mem = "".join(ch for ch in parts[2] if ch.isdigit())
        gpus.append({
            "name": parts[0],
            "driver_version": parts[1],
            "memory_mib": int(mem) if mem else 0,
        })
    return gpus


def driver_cuda_version() -> str | None:
    """读取驱动支持的最高 CUDA 版本（nvidia-smi 头部 ``CUDA Version: 12.7``）。"""
    res = run(["nvidia-smi"], timeout=20)
    if not res.ok:
        return None
    for line in res.out.splitlines():
        if "CUDA Version" in line:
            tail = line.split("CUDA Version", 1)[1]
            token = tail.replace(":", "").strip().split()[0]
            if token and token[0].isdigit():
                return token.rstrip(",")
    return None


def nvcc_version() -> str | None:
    """CUDA Toolkit 编译器版本；未装 Toolkit 时为 None。"""
    res = run(["nvcc", "--version"], timeout=20)
    if not res.ok:
        return None
    for line in res.out.splitlines():
        if "release" in line:
            # Cuda compilation tools, release 12.4, V12.4.131
            seg = line.split("release", 1)[1]
            token = seg.replace(",", " ").strip().split()[0]
            return token
    return None


_TORCH_PROBE = (
    "import json;d={'installed':False};"
    "\ntry:\n"
    " import torch\n"
    " d={'installed':True,'version':torch.__version__,"
    "'cuda_compiled':torch.version.cuda,"
    "'cuda_available':torch.cuda.is_available(),"
    "'device_count':torch.cuda.device_count()}\n"
    "except Exception as e:\n"
    " d={'installed':False,'error':str(e)}\n"
    "print(json.dumps(d))"
)


def torch_info(python_exe: str, timeout: int = 90) -> dict:
    """在指定解释器里探测 torch；import torch 可能较慢，给足超时。"""
    res = run([python_exe, "-c", _TORCH_PROBE], timeout=timeout)
    if not res.ok and not res.out:
        return {"installed": False, "error": res.stderr or "探测超时"}
    for line in reversed(res.out.splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    return {"installed": False, "error": res.out[:200]}


# ---------------------------------------------------------------------------
# 平台信息
# ---------------------------------------------------------------------------

def platform_summary() -> dict:
    return {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "os_name": os.name,
        "python": platform.python_version(),
        "is_windows": os.name == "nt",
    }
