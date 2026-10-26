"""修复执行器：把知识库里的步骤变成「可确认后执行」的动作。

安全模型（三级）：
- safe    只读或幂等的查询，不改系统
- config  改用户级配置（pip/git config），影响面小、可回退
- system  安装/卸载软件、改系统环境，必须显式确认

默认 dry-run：只打印将要执行的命令，不真正动手。
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

from . import probes
from .models import EnvReport

RISK_ORDER = {"safe": 0, "config": 1, "system": 2}

# 命令前缀 -> 风险级别。未登记的命令一律按 system 处理。
_RISK_RULES: list[tuple[tuple[str, ...], str]] = [
    (("git", "config"), "config"),
    (("git", "ls-remote"), "safe"),
    (("git", "status"), "safe"),
    (("git", "log"), "safe"),
    (("python", "-m", "pip", "config"), "config"),
    (("python", "-m", "pip", "list"), "safe"),
    (("python", "-m", "pip", "show"), "safe"),
    (("python", "-m", "pip", "uninstall"), "system"),
    (("python", "-m", "pip", "install"), "system"),
    (("python", "-m", "pip", "--version"), "safe"),
    (("nvidia-smi",), "safe"),
    (("nvcc",), "safe"),
    (("py", "-0p"), "safe"),
    (("py", "-0"), "safe"),
    (("py", "--version"), "safe"),
    (("winget",), "system"),
    (("sudo",), "system"),
    (("apt",), "system"),
    (("conda",), "system"),
    (("pkill",), "system"),
    (("ssh-keygen",), "config"),
]

# 纯只读命令：出现在任何位置都按 safe 处理
_READ_ONLY = {
    "cat", "ls", "pwd", "echo", "which", "where", "whoami", "type",
    "head", "tail", "grep", "find", "du", "df", "env", "printenv",
}


def _strip_quoted(command: str) -> str:
    """去掉引号内的内容，避免把 Python 代码里的分号当成 shell 连接符。"""
    return re.sub(r'"[^"]*"|\'[^\']*\'', '""', command)


def _is_read_only_code(command: str) -> bool:
    """``python -c`` / ``py -c`` 执行的是内联代码，默认按只读看待。

    这是权衡：内联代码理论上可以删文件，但知识库里的 -c 全是验证类命令，
    误判为 system 会让用户每次都要确认，反而促使他们无脑加 --yes。
    """
    tokens = _strip_quoted(command).split()
    return len(tokens) >= 2 and tokens[0] in ("python", "python3", "py") and "-c" in tokens[:3]


def _classify(command: str) -> str:
    tokens = command.strip().split()
    if not tokens:
        return "system"

    unquoted = _strip_quoted(command)

    # 已登记的前缀优先级最高（避免 pip install 被当成 python 只读命令）
    for prefix, risk in _RISK_RULES:
        if tuple(tokens[: len(prefix)]) == prefix:
            return risk

    if _is_read_only_code(command):
        return "safe"

    if tokens[0] in _READ_ONLY:
        # cat a | grep b 这类管道是只读的；但带重定向的写文件不是
        if ">" not in unquoted:
            return "safe"

    # 剩下的复合命令按最危险的一段算
    for sep in ("&&", "||", "|", ";"):
        if sep in unquoted:
            return "system"

    return "system"


def _platform_ok(step_platform: str) -> bool:
    if step_platform in ("", "any", None):
        return True
    if step_platform == "win":
        return os.name == "nt"
    if step_platform == "linux":
        return os.name != "nt"
    return True


@dataclass
class FixAction:
    check_id: str
    kb_id: str
    title: str
    command: str
    risk: str
    platform: str
    reason: str = ""
    extra_steps: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "check_id": self.check_id,
            "kb_id": self.kb_id,
            "title": self.title,
            "command": self.command,
            "risk": self.risk,
            "platform": self.platform,
            "reason": self.reason,
            "extra_steps": self.extra_steps,
        }


def build_plan(report: EnvReport, kb: dict, max_risk: str = "system") -> list[FixAction]:
    """根据检测报告，生成按风险排序的待执行动作清单。"""
    limit = RISK_ORDER.get(max_risk, 2)
    actions: list[FixAction] = []

    for result in report.results:
        if not result.kb_id or result.kb_id not in kb:
            continue
        entry = kb[result.kb_id]
        for step in entry.get("steps", []):
            cmd = step.get("command")
            plat = step.get("platform", "any")
            if not cmd or not _platform_ok(plat):
                continue
            risk = _classify(cmd)
            if RISK_ORDER[risk] > limit:
                continue
            actions.append(FixAction(
                check_id=result.id,
                kb_id=result.kb_id,
                title=entry["title"],
                command=cmd,
                risk=risk,
                platform=plat,
                reason=step.get("text", ""),
                extra_steps=[s.get("text", "") for s in entry.get("steps", [])
                             if not s.get("command") and _platform_ok(s.get("platform", "any"))],
            ))

    # 去重：同一条命令只保留一次
    seen: set[str] = set()
    unique: list[FixAction] = []
    for a in actions:
        if a.command in seen:
            continue
        seen.add(a.command)
        unique.append(a)

    unique.sort(key=lambda a: (RISK_ORDER[a.risk], a.check_id))
    return unique


def execute(action: FixAction, approve_system: bool = False, timeout: int = 600) -> dict:
    """执行单个动作。

    approve_system 为 False 时拒绝执行 system 级命令，返回明确的原因，
    避免 agent 在用户没确认的情况下改动系统。
    """
    if action.risk == "system" and not approve_system:
        return {
            "command": action.command,
            "executed": False,
            "reason": "system 级操作需要用户显式确认",
        }

    if action.risk == "system":
        # 系统级安装用 shell 执行，允许 && / 重定向等写法
        proc = probes.run(["cmd", "/c", action.command] if os.name == "nt" else
                          ["bash", "-lc", action.command], timeout=timeout)
    else:
        proc = probes.run(action.command.split(), timeout=timeout)

    return {
        "command": action.command,
        "executed": True,
        "ok": proc.ok,
        "code": proc.code,
        "output": proc.out[-2000:],
    }
