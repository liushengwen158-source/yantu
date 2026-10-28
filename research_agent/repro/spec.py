"""论文复现的数据契约。

两个对称的对象：
- ``PaperEnvSpec``     论文仓库要求什么
- ``LocalEnvProfile``  本机有什么

比对后产出 ``FeasibilityReport``。所有需求项必须带 ``Evidence``，
证据里没有来源文件的需求不允许进入报告——这是防止编造的地基。
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any


class Category(str, Enum):
    HARDWARE = "hardware"   # GPU 型号/显存/数量、CPU、内存、磁盘
    PYTHON = "python"       # 解释器版本
    FRAMEWORK = "framework" # torch / tf / jax 及其版本
    CUDA = "cuda"           # CUDA / cuDNN
    SYSTEM = "system"       # 系统库、容器运行时、OS
    DATA = "data"           # 数据集大小

    @property
    def label(self) -> str:
        return {
            "hardware": "硬件",
            "python": "解释器",
            "framework": "框架",
            "cuda": "CUDA",
            "system": "系统",
            "data": "数据",
        }[self.value]


class EvidenceLevel(str, Enum):
    """证据可信度。报告里必须显式展示，用户据此决定信多少。"""

    L1 = "L1"  # 论文正文 / 官方文档明确声明
    L2 = "L2"  # 仓库文件可确定性推断
    L3 = "L3"  # 经验推测，必须标注

    @property
    def label(self) -> str:
        return {
            "L1": "论文明确",
            "L2": "代码推断",
            "L3": "经验推测",
        }[self.value]


class Verdict(str, Enum):
    """单项判定四档。"""

    OK = "ok"              # 满足
    ADD = "add"            # 装一下/改版本即可
    DEGRADE = "degrade"    # 需要改配置或算法才能跑
    BLOCK = "block"        # 物理上不可能
    CONFLICT = "conflict"  # 需求本身自相矛盾，需人工确认
    UNKNOWN = "unknown"    # 信息不足，无法判定

    @property
    def label(self) -> str:
        return {
            "ok": "满足",
            "add": "可补齐",
            "degrade": "需降级",
            "block": "阻塞",
            "conflict": "冲突",
            "unknown": "待确认",
        }[self.value]

    @property
    def order(self) -> int:
        return {"block": 0, "conflict": 1, "degrade": 2, "add": 3, "unknown": 4, "ok": 5}[self.value]


@dataclass
class Evidence:
    """一条需求的证据来源。source_file 为空时该需求不应被采纳。"""

    level: EvidenceLevel
    source_file: str = ""
    line: int = 0
    excerpt: str = ""

    @property
    def is_valid(self) -> bool:
        return bool(self.source_file) and self.level != EvidenceLevel.L3 or bool(self.source_file)

    def describe(self) -> str:
        if not self.source_file:
            return "无来源"
        loc = f"{self.source_file}:{self.line}" if self.line else self.source_file
        return f"{loc} ({self.level.value})"

    def to_dict(self) -> dict[str, Any]:
        return {
            "level": self.level.value,
            "level_label": self.level.label,
            "source_file": self.source_file,
            "line": self.line,
            "excerpt": self.excerpt,
        }


@dataclass
class EnvRequirement:
    """单条环境需求。"""

    id: str                      # gpu.vram / torch.version / cuda / python.version
    category: Category
    op: str                      # >= <= == != ~= exists
    value: str
    unit: str = ""
    hard: bool = True            # True=硬约束（不足即阻塞） False=可降级
    why: str = ""                # 为什么需要这一条
    evidence: Evidence | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def accepted(self) -> bool:
        """是否允许进入报告：必须有证据来源。"""
        return self.evidence is not None and bool(self.evidence.source_file)

    def requirement_text(self) -> str:
        if self.op == "exists":
            return "需要"
        unit = f" {self.unit}" if self.unit else ""
        return f"{self.op} {self.value}{unit}".strip()

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "category": self.category.value,
            "category_label": self.category.label,
            "op": self.op,
            "value": self.value,
            "unit": self.unit,
            "hard": self.hard,
            "why": self.why,
            "requirement_text": self.requirement_text(),
            "evidence": self.evidence.to_dict() if self.evidence else None,
        }


@dataclass
class PaperEnvSpec:
    """论文仓库的环境需求集合。"""

    repo_path: str
    requirements: list[EnvRequirement] = field(default_factory=list)
    scanned_files: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def accepted(self) -> list[EnvRequirement]:
        return [r for r in self.requirements if r.accepted]

    @property
    def rejected(self) -> list[EnvRequirement]:
        return [r for r in self.requirements if not r.accepted]

    def by_category(self) -> dict[str, list[EnvRequirement]]:
        out: dict[str, list[EnvRequirement]] = {}
        for r in self.accepted:
            out.setdefault(r.category.value, []).append(r)
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "repo_path": self.repo_path,
            "scanned_files": self.scanned_files,
            "warnings": self.warnings,
            "notes": self.notes,
            "requirements": [r.to_dict() for r in self.requirements],
            "accepted_count": len(self.accepted),
            "rejected_count": len(self.rejected),
        }


@dataclass
class LocalEnvProfile:
    """本机环境画像。字段与 PaperEnvSpec 同构，便于直接比对。"""

    python_version: str = ""
    python_executable: str = ""
    torch_version: str | None = None
    torch_cuda: str | None = None
    cuda_available: bool = False
    gpu_name: str = ""
    gpu_count: int = 0
    gpu_vram_gb: float = 0.0
    driver_version: str = ""
    driver_cuda: str = ""
    disk_free_gb: float = 0.0
    disk_total_gb: float = 0.0
    ram_gb: float = 0.0
    cpu_count: int = 0
    platform: str = ""
    os_name: str = ""
    has_docker: bool = False
    has_apptainer: bool = False
    has_conda: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DiffItem:
    """单条需求与本机的对比结果。"""

    requirement: EnvRequirement
    local_value: str
    verdict: Verdict
    reason: str = ""
    remediation: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "requirement": self.requirement.to_dict(),
            "local_value": self.local_value,
            "verdict": self.verdict.value,
            "verdict_label": self.verdict.label,
            "reason": self.reason,
            "remediation": self.remediation,
        }


@dataclass
class FeasibilityReport:
    """整体可行性报告。"""

    spec: PaperEnvSpec
    profile: LocalEnvProfile
    items: list[DiffItem] = field(default_factory=list)

    @property
    def counts(self) -> dict[str, int]:
        c = {v.value: 0 for v in Verdict}
        for it in self.items:
            c[it.verdict.value] += 1
        return c

    @property
    def overall(self) -> Verdict:
        """整体结论取最严重的单项判定。"""
        if not self.items:
            return Verdict.UNKNOWN
        return min((it.verdict for it in self.items), key=lambda v: v.order)

    def headline(self) -> str:
        blockers = [i for i in self.items if i.verdict is Verdict.BLOCK]
        if not blockers:
            return "可以跑通"
        names = "、".join(i.requirement.id for i in blockers[:2])
        return f"存在物理阻塞（{names}），需更换算力"

    def to_dict(self) -> dict[str, Any]:
        return {
            "spec": self.spec.to_dict(),
            "profile": self.profile.to_dict(),
            "items": [i.to_dict() for i in self.items],
            "counts": self.counts,
            "overall": self.overall.value,
            "overall_label": self.overall.label,
            "headline": self.headline(),
        }
