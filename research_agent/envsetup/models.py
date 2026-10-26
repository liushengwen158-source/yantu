"""环境检测的数据结构。

设计原则：所有诊断结果都序列化为普通 dict，方便：
1. CLI 直接打印成表格；
2. openJiuwen agent 把结果塞进 prompt 或作为 tool 返回值；
3. 落盘成 JSON 供后续对比（比如「上次检测 vs 这次检测」）。
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any


class Status(str, Enum):
    """单项检测的状态。"""

    OK = "ok"                 # 正常
    INFO = "info"             # 可选建议，不影响使用
    WARN = "warn"             # 能跑但会踩坑
    BLOCKER = "blocker"       # 阻塞，必须修
    UNKNOWN = "unknown"       # 检测失败/无法判断

    @property
    def order(self) -> int:
        return {"blocker": 0, "warn": 1, "info": 2, "unknown": 3, "ok": 4}[self.value]

    @property
    def label(self) -> str:
        return {
            "ok": "OK",
            "info": "建议",
            "warn": "注意",
            "blocker": "阻塞",
            "unknown": "未知",
        }[self.value]


@dataclass
class CheckResult:
    """一项检测的结果。

    id      检测项标识，如 ``python.version``
    domain  所属领域 python / git / gpu
    name    人类可读名称
    status  状态
    value   检测到的实际值，如 "3.13.4"
    message 一句话说明
    kb_id   关联知识库条目 id，None 表示无需建议
    detail  附加信息（命令路径、版本元组等），供 agent 深入解释
    """

    id: str
    domain: str
    name: str
    status: Status
    value: str = ""
    message: str = ""
    kb_id: str | None = None
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        d["status_label"] = self.status.label
        return d


@dataclass
class EnvReport:
    """一次完整环境检测的报告。"""

    platform: str
    python_executable: str
    python_version: str
    results: list[CheckResult] = field(default_factory=list)
    generated_at: str = ""

    # --- 聚合视图 -----------------------------------------------------

    def by_domain(self) -> dict[str, list[CheckResult]]:
        out: dict[str, list[CheckResult]] = {}
        for r in self.results:
            out.setdefault(r.domain, []).append(r)
        return out

    def blockers(self) -> list[CheckResult]:
        return [r for r in self.results if r.status is Status.BLOCKER]

    def warnings(self) -> list[CheckResult]:
        return [r for r in self.results if r.status is Status.WARN]

    def suggestions(self) -> list[CheckResult]:
        return [r for r in self.results if r.status is Status.INFO]

    def score(self) -> int:
        """0-100 的就绪度评分，供 agent 用一句话概括「环境能用了没」。"""
        if not self.results:
            return 0
        weight = {
            Status.OK: 1.0,
            Status.INFO: 0.85,
            Status.WARN: 0.5,
            Status.BLOCKER: 0.0,
            Status.UNKNOWN: 0.5,
        }
        total = sum(weight[r.status] for r in self.results)
        return round(100 * total / len(self.results))

    def summary(self) -> dict[str, Any]:
        return {
            "score": self.score(),
            "blockers": len(self.blockers()),
            "warnings": len(self.warnings()),
            "suggestions": len(self.suggestions()),
            "ok": sum(1 for r in self.results if r.status is Status.OK),
            "total": len(self.results),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "platform": self.platform,
            "python_executable": self.python_executable,
            "python_version": self.python_version,
            "generated_at": self.generated_at,
            "summary": self.summary(),
            "results": [r.to_dict() for r in self.results],
        }
