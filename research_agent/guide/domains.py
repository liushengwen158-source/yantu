"""方向知识：从对话里识别方向，并决定这个方向需要哪些维度。

单独成一个模块，是因为它同时被画像（决定评估哪些维度）和路线（决定编排
哪些阶段）使用；放在任何一边都会造成循环依赖。

识别策略刻意保守：
- 只认 domains.json 里列出的关键词，不做语义猜测
- 一句话里同时命中多个方向时，取命中关键词总长度最大的；打平就报「未定」
- 用户明确说「还没定 / 都可以 / 不知道」时，直接返回 UNKNOWN，不去硬猜

方向没定的代价只是「先走核心四维」，比猜错方向、给一堆用不上的任务要好。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from .spec import CORE_DIMENSIONS, Dimension, Domain

KNOWLEDGE_DIR = Path(__file__).resolve().parent.parent / "knowledge"
DOMAINS_PATH = KNOWLEDGE_DIR / "domains.json"

# 用户明确表示「方向还没定」的说法。命中这些就不猜。
# 旧方向 id 与常见叫法都能对上，避免用户按记忆里的名字输入时被拒。
DOMAIN_ALIASES = {
    "database": Domain.DATA,
    "db": Domain.DATA,
    "data": Domain.DATA,
    "networks": Domain.SYSTEMS,
    "network": Domain.SYSTEMS,
    "system": Domain.SYSTEMS,
    "systems": Domain.SYSTEMS,
    "graphics": Domain.SOFTWARE,
    "hci": Domain.SOFTWARE,
    "software-engineering": Domain.SOFTWARE,
    "ai4science": Domain.AI,
    "ml": Domain.AI,
    "dl": Domain.AI,
    "machine-learning": Domain.AI,
}

NO_DOMAIN_HINTS = (
    "还没定",
    "没定方向",
    "不确定方向",
    "不知道方向",
    "都可以",
    "都行",
    "还没选",
    "先看看",
    "还没想好",
)


@dataclass
class DomainSpec:
    domain: Domain
    name: str
    why: str = ""
    keywords: list[str] = field(default_factory=list)
    dims: dict[Dimension, int] = field(default_factory=dict)
    skills: list[str] = field(default_factory=list)
    reading_hint: str = ""
    papers: list[dict[str, Any]] = field(default_factory=list)
    stage_tasks: list[dict[str, Any]] = field(default_factory=list)
    first_project: dict[str, Any] = field(default_factory=dict)

    @property
    def active_dimensions(self) -> tuple[Dimension, ...]:
        """核心维在前，方向维在后。顺序即展示顺序。"""
        extra = [d for d in self.dims if d not in CORE_DIMENSIONS]
        return CORE_DIMENSIONS + tuple(extra)

    @property
    def required(self) -> dict[Dimension, int]:
        return dict(self.dims)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.domain.value,
            "name": self.name,
            "why": self.why,
            "keywords": self.keywords,
            "dims": {d.value: v for d, v in self.dims.items()},
            "skills": self.skills,
            "reading_hint": self.reading_hint,
            "papers": self.papers,
            "stage_tasks": self.stage_tasks,
            "first_project": self.first_project,
        }


def _parse_dimension(name: str) -> Dimension | None:
    try:
        return Dimension(name)
    except ValueError:
        return None


@lru_cache(maxsize=1)
def load_domains() -> dict[Domain, DomainSpec]:
    """读方向库。缓存一份，避免每次评估都重新解析 JSON。"""
    if not DOMAINS_PATH.exists():
        return {}
    raw = json.loads(DOMAINS_PATH.read_text(encoding="utf-8"))
    out: dict[Domain, DomainSpec] = {}
    for item in raw.get("domains", []):
        try:
            domain = Domain(item["id"])
        except (KeyError, ValueError):
            continue
        dims: dict[Dimension, int] = {}
        for key, level in (item.get("dims") or {}).items():
            dim = _parse_dimension(key)
            if dim is None:
                continue
            dims[dim] = int(level)
        # 核心维永远在，即使方向库漏写
        for dim in CORE_DIMENSIONS:
            dims.setdefault(dim, 1)
        out[domain] = DomainSpec(
            domain=domain,
            name=item.get("name", domain.label),
            why=item.get("why", ""),
            keywords=list(item.get("keywords") or []),
            dims=dims,
            skills=list(item.get("skills") or []),
            reading_hint=item.get("reading_hint", ""),
            papers=list(item.get("papers") or []),
            stage_tasks=list(item.get("stage_tasks") or []),
            first_project=item.get("first_project") or {},
        )
    return out


def get_domain(domain: Domain) -> DomainSpec | None:
    return load_domains().get(domain)


def list_domains() -> list[DomainSpec]:
    """按固定顺序列出方向，供 CLI 与 agent 展示。"""
    order = [
        Domain.SYSTEMS,
        Domain.SECURITY,
        Domain.DATA,
        Domain.THEORY,
        Domain.SOFTWARE,
        Domain.AI,
    ]
    specs = load_domains()
    return [specs[d] for d in order if d in specs]


def active_dimensions(domain: Domain) -> tuple[Dimension, ...]:
    """该方向需要评估的维度。方向未定 → 只有核心四维。"""
    spec = get_domain(domain)
    if spec is None:
        return CORE_DIMENSIONS
    return spec.active_dimensions


def required_levels(domain: Domain) -> dict[Dimension, int]:
    spec = get_domain(domain)
    if spec is None:
        return {}
    return spec.required


def _clauses(text: str) -> list[str]:
    return [c for c in re.split(r"[。！？!?；;\n]", text) if c.strip()]


def _keyword_hit(keyword: str, text: str, lowered: str) -> bool:
    """关键词匹配。

    纯英文关键词（AI、TCP、SQL）必须按词边界匹配，否则 ``ai`` 会在
    ``train``、``detail`` 里被命中；中文关键词直接按子串匹配。
    """
    if not keyword:
        return False
    if re.fullmatch(r"[a-z0-9+#.\- ]+", keyword):
        pattern = rf"(?<![a-z0-9]){re.escape(keyword)}(?![a-z0-9])"
        return re.search(pattern, lowered) is not None
    return keyword.lower() in lowered


def resolve_domain(value: str) -> Domain:
    """把用户给的方向（id 或中文名）解析成 ``Domain``。

    认不出来时抛 ``ValueError``，让调用方把可选值回给用户，
    而不是猜一个方向继续算。
    """
    text = (value or "").strip()
    if not text:
        return Domain.UNKNOWN
    if text.lower() in DOMAIN_ALIASES:
        return DOMAIN_ALIASES[text.lower()]
    for spec in load_domains().values():
        if text == spec.domain.value or text == spec.name:
            return spec.domain
        # 允许简称：「系统」「数据库」「图形学」都能对上全名
        if len(text) >= 2 and text in spec.name:
            return spec.domain
    detected, _ = detect_domain(text)
    if detected is not Domain.UNKNOWN:
        return detected
    options = "、".join(f"{s.name}（{s.domain.value}）" for s in list_domains())
    raise ValueError(f"认不出方向「{value}」；可选：{options}")


def detect_domain(text: str) -> tuple[Domain, str]:
    """从对话里识别方向，返回 (方向, 依据原话)。

    依据只用来解释「为什么认为你在做这个方向」，用户不认同可以直接改。
    """
    if not text or not text.strip():
        return Domain.UNKNOWN, ""
    for hint in NO_DOMAIN_HINTS:
        if hint in text:
            return Domain.UNKNOWN, ""

    scores: dict[Domain, tuple[int, str]] = {}
    lowered = text.lower()
    for spec in load_domains().values():
        total = 0
        best = ""
        for keyword in spec.keywords:
            if _keyword_hit(keyword, text, lowered):
                total += len(keyword)
                if len(keyword) > len(best):
                    best = keyword
        if total:
            scores[spec.domain] = (total, best)

    if not scores:
        return Domain.UNKNOWN, ""

    ranked = sorted(scores.items(), key=lambda kv: -kv[1][0])
    top_domain, (top_score, keyword) = ranked[0]
    if len(ranked) > 1 and ranked[1][1][0] == top_score:
        # 打平就当没识别出来，不硬选一个
        return Domain.UNKNOWN, ""

    evidence = ""
    for clause in _clauses(text):
        if _keyword_hit(keyword, clause, clause.lower()):
            evidence = clause.strip()
            break
    return top_domain, evidence
