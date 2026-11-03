"""新手引导的数据契约。

和 ``repro/spec.py`` 同一条原则：**任何进入结论的判断都必须能指回来源**。
这里的来源是用户的原话，或用户授权扫描到的文件。

和第一版最大的区别：**维度不是固定的六维**。
计算机的研究方向差别太大——做系统和做理论要补的东西几乎没有交集，
所以维度分成两层：

- 核心维：编程与调试、命令行与工程工具、数学基础、科研方法（谁都要）
- 方向维：系统底层、数据与数据库、安全、算法理论、机器学习、深度学习
  只在选定方向时才激活

方向未确定时只评估核心维，``ML`` / ``DL`` 不会出现在报告里——
不把「不需要掌握的东西」摆到用户面前，是这套设计的前提。

刻意不给「总分」一个独立字段：分数永远由档位推出，旁边必须带证据，
否则 70 分是个没法核对、也没法反驳的数字。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, IntEnum
from statistics import median
from typing import Any


class Domain(str, Enum):
    """研究方向。用于决定「哪些维度需要评估、哪些阶段需要走」。"""

    UNKNOWN = "unknown"
    SYSTEMS = "systems"          # 系统、体系结构、分布式、存储
    NETWORKS = "networks"        # 网络
    SECURITY = "security"        # 系统与网络安全
    DATABASE = "database"        # 数据库与数据管理
    THEORY = "theory"            # 理论与算法
    SOFTWARE = "software"        # 软件工程与程序语言
    GRAPHICS = "graphics"        # 图形学与多媒体
    AI = "ai"                    # 机器学习 / 深度学习 / CV / NLP
    AI4SCIENCE = "ai4science"    # AI for Science

    @property
    def label(self) -> str:
        return {
            "unknown": "未定方向",
            "systems": "系统与体系结构",
            "networks": "网络",
            "security": "安全",
            "database": "数据库与数据管理",
            "theory": "理论与算法",
            "software": "软件工程与程序语言",
            "graphics": "图形学与多媒体",
            "ai": "人工智能",
            "ai4science": "AI for Science",
        }[self.value]


class Dimension(str, Enum):
    """能力维度。前四个是核心维，后面按方向激活。"""

    CODING = "coding"
    TOOLS = "tools"
    MATH = "math"
    RESEARCH = "research"
    SYSTEMS = "systems"
    DATA = "data"
    SECURITY = "security"
    THEORY = "theory"
    ML = "ml"
    DL = "dl"

    @property
    def label(self) -> str:
        return {
            "coding": "编程与调试",
            "tools": "命令行与工程工具",
            "math": "数学基础",
            "research": "科研方法",
            "systems": "系统与底层",
            "data": "数据与数据库",
            "security": "安全",
            "theory": "算法与理论",
            "ml": "机器学习",
            "dl": "深度学习",
        }[self.value]

    @property
    def is_core(self) -> bool:
        return self in CORE_DIMENSIONS


CORE_DIMENSIONS: tuple[Dimension, ...] = (
    Dimension.CODING,
    Dimension.TOOLS,
    Dimension.MATH,
    Dimension.RESEARCH,
)


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


class Source(str, Enum):
    """证据从哪来。三种来源必须分开展示，不混算成一个分数。

    - ``SELF``      用户自述，最直接，但可能高估
    - ``FILE``      用户授权扫描到的材料，只能证明「材料在」，不能证明「掌握了」
    - ``INFERENCE`` 由规则/模型推断，可信度最低，必须标出来
    """

    SELF = "self"
    FILE = "file"
    INFERENCE = "inference"

    @property
    def label(self) -> str:
        return {"self": "自述", "file": "文件", "inference": "推断"}[self.value]


@dataclass
class Signal:
    """一条证据。

    quote  用户原话或文件里的片段，**不改写、不摘要**——用户要能一眼认出来。
    why    这条话为什么能说明水平，写给人看，也写给模型看。
    source 证据来源；``FILE`` 的证据一定带 ``source_ref``（文件路径）。
    """

    rule_id: str
    dimension: Dimension
    level: Level
    strength: str          # strong（明确陈述）/ weak（含糊表述或文件旁证）
    quote: str
    why: str
    polarity: str = "positive"   # positive / negative。矛盾判定靠它，不靠档位差
    source: Source = Source.SELF
    source_ref: str = ""         # 文件证据的相对路径

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "dimension": self.dimension.value,
            "dimension_label": self.dimension.label,
            "level": int(self.level),
            "level_label": self.level.label,
            "strength": self.strength,
            "polarity": self.polarity,
            "source": self.source.value,
            "source_label": self.source.label,
            "source_ref": self.source_ref,
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
    required: int = 0                                  # 该方向要求的最低档位，0 表示无要求

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
    def required_label(self) -> str:
        """该方向对这一维的最低要求，0 表示没有要求。"""
        if not self.required:
            return ""
        try:
            return Level(int(self.required)).label
        except ValueError:
            return ""

    @property
    def is_known(self) -> bool:
        return self.level is not Level.UNKNOWN

    @property
    def gap(self) -> bool:
        """已知但没达到该方向要求——这才是路线要补的地方。"""
        return self.required > 0 and (not self.is_known or int(self.level) < self.required)

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
            "required": self.required,
            "required_label": Level(int(self.required)).label if self.required else "",
            "gap": self.gap,
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
    """能力画像。只装「这个方向用得上」的维度。

    方向未定时只有核心四维；方向确定后再加上该方向需要的维度。
    不在适用范围内的能力，不评估、不展示、不催着补。
    """

    domain: Domain = Domain.UNKNOWN
    domain_evidence: str = ""                                 # 从哪句话认定方向的
    dimensions: dict[Dimension, DimensionState] = field(default_factory=dict)
    text_length: int = 0
    notes: list[str] = field(default_factory=list)
    scanned: dict[str, Any] = field(default_factory=dict)     # 授权扫描的摘要（目录/计数）

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
    def gaps(self) -> list[DimensionState]:
        """低于方向要求的维度，缺口大的排前面。"""
        items = [d for d in self.dimensions.values() if d.gap]
        items.sort(key=lambda d: (not d.is_known, int(d.level)))
        return items

    @property
    def coverage(self) -> str:
        return f"{len(self.known)}/{len(self.dimensions)}"

    @property
    def conflicts(self) -> list[DimensionState]:
        return [d for d in self.dimensions.values() if d.conflicts]

    @property
    def file_evidence(self) -> list[Signal]:
        """来自授权文件的证据，单独列出来，不跟自述混算。"""
        return [
            s
            for d in self.dimensions.values()
            for s in d.signals
            if s.source is Source.FILE
        ]

    @property
    def overall(self) -> Level:
        """整体档位取已知维度的中位数——不给平均值，避免一维拉高整体。"""
        levels = [d.level for d in self.known]
        if not levels:
            return Level.UNKNOWN
        return Level(int(round(median(levels))))

    def to_dict(self) -> dict[str, Any]:
        return {
            "domain": self.domain.value,
            "domain_label": self.domain.label,
            "domain_evidence": self.domain_evidence,
            "coverage": self.coverage,
            "known_count": len(self.known),
            "unknown_count": len(self.unknown),
            "overall": int(self.overall),
            "overall_label": self.overall.label,
            "text_length": self.text_length,
            "notes": self.notes,
            "scanned": self.scanned,
            "dimensions": [d.to_dict() for d in self.dimensions.values()],
            "gaps": [d.dimension.value for d in self.gaps],
        }
