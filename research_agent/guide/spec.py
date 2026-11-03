"""新手引导的数据契约。

和 ``repro/spec.py`` 同一条原则：**任何进入结论的判断都必须能指回来源**。

这里的来源不是文件行号，而是用户的原话：
- ``Signal``         一句话 → 一条证据（维度 + 档位 + 原话 + 判定理由）
- ``DimensionState`` 一个维度的证据汇总（档位 + 置信度 + 冲突）
- ``SkillProfile``   六维画像（谁、现在在哪、还缺什么）

刻意不给「总分」一个独立字段：分数永远由档位推出，旁边必须带证据，
否则 70 分是个没法核对、也没法反驳的数字。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, IntEnum
from statistics import median
from typing import Any


class Dimension(str, Enum):
    """能力维度。刻意只留六维：再多就变成考试科目表，而不是画像。"""

    PYTHON = "python"
    SHELL = "shell"
    MATH = "math"
    ML = "ml"
    DL = "dl"
    RESEARCH = "research"

    @property
    def label(self) -> str:
        return {
            "python": "Python 工程",
            "shell": "命令行 / Linux",
            "math": "数学（线代·概率）",
            "ml": "机器学习",
            "dl": "深度学习与框架",
            "research": "科研流程",
        }[self.value]


class Level(IntEnum):
    """档位。四档足够定位，再多就没人看得懂。"""

    UNKNOWN = 0
    BEGINNER = 1     # 从零：没碰过或用不起来
    FOUNDATION = 2   # 有基础：跟着教程能跑，独立做还吃力
    WORKING = 3      # 能干活：能自己写、自己调、自己查
    READY = 4        # 可独立：能改进、能复现、能交付出东西

    @property
    def label(self) -> str:
        return {
            0: "未知",
            1: "从零",
            2: "有基础",
            3: "能干活",
            4: "可独立",
        }[int(self)]

    @property
    def score(self) -> int:
        """只用于展示；真实判断以档位和证据为准。"""
        return {0: 0, 1: 25, 2: 50, 3: 75, 4: 100}[int(self)]


class Confidence(str, Enum):
    UNKNOWN = "unknown"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

    @property
    def label(self) -> str:
        return {"unknown": "未知", "low": "低", "medium": "中", "high": "高"}[self.value]


@dataclass
class Signal:
    """一条来自对话的证据。

    quote  必须是用户原话的**完整句子**，不改写、不摘要——用户要能一眼认出来这是自己说的。
    why    这条话为什么能说明水平，写给人看，也写给模型看。
    """

    rule_id: str
    dimension: Dimension
    level: Level
    strength: str          # strong（明确陈述）/ weak（含糊表述）
    quote: str
    why: str
    polarity: str = "positive"   # positive / negative。矛盾判定靠它，不靠档位差

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "dimension": self.dimension.value,
            "dimension_label": self.dimension.label,
            "level": int(self.level),
            "level_label": self.level.label,
            "strength": self.strength,
            "polarity": self.polarity,
            "quote": self.quote,
            "why": self.why,
        }


@dataclass
class DimensionState:
    """一个维度的汇总状态。"""

    dimension: Dimension
    level: Level = Level.UNKNOWN
    confidence: Confidence = Confidence.UNKNOWN
    signals: list[Signal] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)
    caveats: list[str] = field(default_factory=list)   # 与结论档位不同、但仍值得看的证据
    note: str = ""

    @property
    def supporting(self) -> Signal | None:
        """支撑当前档位的那条证据——表格里展示的必须是它，不能随便挑第一条。"""
        if self.level is Level.UNKNOWN:
            return self.signals[0] if self.signals else None
        for s in self.signals:
            if s.level == self.level:
                return s
        return self.signals[0] if self.signals else None

    @property
    def score(self) -> int:
        return self.level.score

    @property
    def is_known(self) -> bool:
        return self.level is not Level.UNKNOWN

    @property
    def needs_probe(self) -> bool:
        """缺信息、或证据打架，都该再问一句。"""
        return not self.is_known or bool(self.conflicts) or self.confidence is Confidence.LOW

    def to_dict(self) -> dict[str, Any]:
        return {
            "dimension": self.dimension.value,
            "dimension_label": self.dimension.label,
            "level": int(self.level),
            "level_label": self.level.label,
            "score": self.score,
            "confidence": self.confidence.value,
            "confidence_label": self.confidence.label,
            "signals": [s.to_dict() for s in self.signals],
            "conflicts": self.conflicts,
            "caveats": self.caveats,
            "note": self.note,
            "needs_probe": self.needs_probe,
        }


@dataclass
class SkillProfile:
    """六维画像。信息不够就说不够，绝不拿 0 分冒充「不会」。"""

    dimensions: dict[Dimension, DimensionState] = field(default_factory=dict)
    text_length: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def known(self) -> list[DimensionState]:
        return [d for d in self.dimensions.values() if d.is_known]

    @property
    def unknown(self) -> list[DimensionState]:
        """完全没有证据的维度。证据打架的算「待确认」，不混进这一栏。"""
        return [d for d in self.dimensions.values() if not d.signals]

    @property
    def pending(self) -> list[DimensionState]:
        """有证据但结论待确认的维度。"""
        return [d for d in self.dimensions.values() if d.signals and not d.is_known]

    @property
    def coverage(self) -> str:
        return f"{len(self.known)}/{len(self.dimensions)}"

    @property
    def conflicts(self) -> list[DimensionState]:
        return [d for d in self.dimensions.values() if d.conflicts]

    @property
    def overall(self) -> Level:
        """整体档位取已知维度的中位数——不给平均值，避免一维拉高整体。"""
        levels = [d.level for d in self.known]
        if not levels:
            return Level.UNKNOWN
        return Level(int(round(median(levels))))

    def to_dict(self) -> dict[str, Any]:
        return {
            "coverage": self.coverage,
            "known_count": len(self.known),
            "unknown_count": len(self.unknown),
            "overall": int(self.overall),
            "overall_label": self.overall.label,
            "text_length": self.text_length,
            "notes": self.notes,
            "dimensions": [d.to_dict() for d in self.dimensions.values()],
        }
