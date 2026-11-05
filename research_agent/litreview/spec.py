"""学术检索与文献关联的数据契约。

和 ``repro/spec.py``、``document`` 同一套原则，这里只是换了对象：

1. **每条结论都能指回来源**。一篇文献必须带 ``source`` + ``source_id`` + ``fetched_at``，
   否则不许进结果；图上的每条边必须带 ``provenance``（哪个源、什么关系、什么时候取的）。
2. **对齐宁可不合并**。两个来源的同一篇论文没把握时，只标 ``possible_duplicates``，
   不硬合并——合并错了会让"被引 3 次"变成"被引 300 次"，而且不可逆。
3. **覆盖度必须显式**。查不到、被限流、超时的源要写进 ``Coverage``，
   报告里明写"本次 S2 不可用，引用网络不完整"，不能假装检索过整个领域。

刻意不做的事：这里不解析 PDF（那是 ``document`` 的活）、不判断能不能复现
（那是 ``repro`` 的活）。本包只负责**找得到、对得上、排得出**。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Source(str, Enum):
    """数据来源。``local`` 指本机已下载的 PDF / 笔记（后续接入）。"""

    ARXIV = "arxiv"
    OPENALEX = "openalex"
    CROSSREF = "crossref"
    S2 = "s2"
    LOCAL = "local"

    @property
    def label(self) -> str:
        return {
            "arxiv": "arXiv",
            "openalex": "OpenAlex",
            "crossref": "Crossref",
            "s2": "Semantic Scholar",
            "local": "本地",
        }[self.value]


class Relation(str, Enum):
    """论文之间的关系。一条边只能有一种关系，混着画就没人能解释它。"""

    CITES = "cites"          # 有向：A 引用了 B
    CITED_BY = "cited_by"    # 有向：A 被 B 引用
    COUPLING = "coupling"    # 无向：共享参考文献（文献耦合）
    SIMILAR = "similar"      # 无向：源站给出的相似推荐，不等于因果

    @property
    def label(self) -> str:
        return {
            "cites": "引用",
            "cited_by": "被引",
            "coupling": "文献耦合",
            "similar": "相似",
        }[self.value]

    @property
    def directed(self) -> bool:
        return self in (Relation.CITES, Relation.CITED_BY)


class Verdict(str, Enum):
    """阅读建议四档。刻意不用"重要/一般"这种没法核对的词。"""

    READ_NOW = "read_now"   # 现在就读：入门必经或当前最相关
    SKIM = "skim"           # 扫一遍：知道结论即可
    PARK = "park"           # 先放着：重要但现在读不动/读不起
    DROP = "drop"           # 不用读：偏题、重复、或质量信号差

    @property
    def label(self) -> str:
        return {
            "read_now": "现在精读",
            "skim": "快速略读",
            "park": "先搁置",
            "drop": "不必读",
        }[self.value]

    @property
    def order(self) -> int:
        return {"read_now": 0, "skim": 1, "park": 2, "drop": 3}[self.value]


class Confidence(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

    @property
    def label(self) -> str:
        return {"high": "高", "medium": "中", "low": "低"}[self.value]


@dataclass
class PaperRef:
    """一篇文献。``source``/``source_id`` 与 ``fetched_at`` 缺一不可。"""

    source: Source
    source_id: str
    title: str
    authors: list[str] = field(default_factory=list)
    year: int | None = None
    venue: str = ""
    abstract: str = ""
    url: str = ""
    doi: str = ""
    arxiv_id: str = ""
    openalex_id: str = ""
    cited_by: int = 0
    cited_known: bool = True   # arXiv 不提供被引数：未知不能当成 0，否则会被系统性地压到后面
    is_survey: bool = False
    fields: list[str] = field(default_factory=list)
    fetched_at: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def valid(self) -> bool:
        """能不能进结果：必须有来源 id 和标题。"""
        return bool(self.source_id) and bool(self.title.strip())

    @property
    def author_text(self) -> str:
        if not self.authors:
            return ""
        if len(self.authors) <= 3:
            return ", ".join(self.authors)
        return f"{', '.join(self.authors[:3])} 等"

    def short(self, width: int = 60) -> str:
        t = " ".join(self.title.split())
        return t if len(t) <= width else t[: width - 1] + "…"

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source.value,
            "source_label": self.source.label,
            "source_id": self.source_id,
            "title": self.title,
            "authors": self.authors,
            "year": self.year,
            "venue": self.venue,
            "abstract": self.abstract[:1200],
            "url": self.url,
            "doi": self.doi,
            "arxiv_id": self.arxiv_id,
            "openalex_id": self.openalex_id,
            "cited_by": self.cited_by,
            "cited_known": self.cited_known,
            "is_survey": self.is_survey,
            "fields": self.fields,
            "fetched_at": self.fetched_at,
        }


@dataclass
class Identity:
    """跨源对齐后的规范身份。``match_method`` 记录它是怎么被认出来的。"""

    canonical_key: str
    doi: str = ""
    arxiv_id: str = ""
    openalex_id: str = ""
    s2_id: str = ""
    match_method: str = ""      # doi | arxiv | title_exact | title_fuzzy | single_source
    confidence: Confidence = Confidence.MEDIUM
    possible_duplicates: list[str] = field(default_factory=list)
    merged: list[str] = field(default_factory=list)   # 被合并进来的 source:source_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "canonical_key": self.canonical_key,
            "doi": self.doi,
            "arxiv_id": self.arxiv_id,
            "openalex_id": self.openalex_id,
            "s2_id": self.s2_id,
            "match_method": self.match_method,
            "confidence": self.confidence.value,
            "confidence_label": self.confidence.label,
            "possible_duplicates": self.possible_duplicates,
            "merged": self.merged,
        }


@dataclass
class SourceStatus:
    """单个源的调用结果。失败不是异常，是要展示的事实。"""

    source: Source
    ok: bool = False
    attempts: int = 0
    count: int = 0
    error: str = ""
    elapsed_ms: int = 0
    cached: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source.value,
            "source_label": self.source.label,
            "ok": self.ok,
            "attempts": self.attempts,
            "count": self.count,
            "error": self.error,
            "elapsed_ms": self.elapsed_ms,
            "cached": self.cached,
        }


@dataclass
class Coverage:
    """一次检索的覆盖度。报告里必须显示，用户据此决定信几分。"""

    statuses: list[SourceStatus] = field(default_factory=list)

    @property
    def ok_sources(self) -> list[str]:
        return [s.source.value for s in self.statuses if s.ok]

    @property
    def failed_sources(self) -> list[str]:
        return [s.source.value for s in self.statuses if not s.ok]

    @property
    def degraded(self) -> bool:
        """有一个源失败就算降级——不掩饰。"""
        return any(not s.ok for s in self.statuses)

    def note(self) -> str:
        if not self.statuses:
            return "没有可用来源"
        if not self.failed_sources:
            return f"全部来源可用（{len(self.ok_sources)} 个）"
        return (
            f"降级：{'、'.join(self.ok_sources) or '无'} 可用，"
            f"{'、'.join(self.failed_sources)} 不可用，结果可能不完整"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "statuses": [s.to_dict() for s in self.statuses],
            "ok_sources": self.ok_sources,
            "failed_sources": self.failed_sources,
            "degraded": self.degraded,
            "note": self.note(),
        }


@dataclass
class Retrieval:
    """一次检索的完整现场。可重放是它存在的理由。"""

    question: str
    queries: list[str] = field(default_factory=list)
    filters: dict[str, Any] = field(default_factory=dict)
    started_at: str = ""
    finished_at: str = ""
    coverage: Coverage = field(default_factory=Coverage)
    papers: list[PaperRef] = field(default_factory=list)
    identities: dict[str, Identity] = field(default_factory=dict)
    dropped: int = 0
    warnings: list[str] = field(default_factory=list)

    def fingerprint(self) -> str:
        """结果指纹：同样的检索式 + 同样的命中，指纹就一样。"""
        h = hashlib.sha1()
        h.update(self.question.encode("utf-8"))
        for q in self.queries:
            h.update(q.encode("utf-8"))
        for p in self.papers:
            h.update(f"{p.source.value}:{p.source_id}".encode("utf-8"))
        return h.hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "queries": self.queries,
            "filters": self.filters,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "fingerprint": self.fingerprint(),
            "coverage": self.coverage.to_dict(),
            "count": len(self.papers),
            "dropped": self.dropped,
            "papers": [p.to_dict() for p in self.papers],
            "identities": {k: v.to_dict() for k, v in self.identities.items()},
            "warnings": self.warnings,
        }


@dataclass
class Edge:
    """图上的一条边。``provenance`` 为空说明这条边来路不明，不许画。"""

    src: str
    dst: str
    relation: Relation
    provenance: str = ""
    context: str = ""

    @property
    def valid(self) -> bool:
        return bool(self.src) and bool(self.dst) and bool(self.provenance) and self.src != self.dst

    def to_dict(self) -> dict[str, Any]:
        return {
            "src": self.src,
            "dst": self.dst,
            "relation": self.relation.value,
            "relation_label": self.relation.label,
            "provenance": self.provenance,
            "context": self.context,
        }


@dataclass
class PaperGraph:
    """文献关系图。节点用 canonical_key，边的两端必须是图里的节点。"""

    seed: str = ""
    nodes: dict[str, PaperRef] = field(default_factory=dict)
    edges: list[Edge] = field(default_factory=list)
    coverage: Coverage = field(default_factory=Coverage)
    warnings: list[str] = field(default_factory=list)

    def add_edge(self, edge: Edge) -> None:
        if not edge.valid:
            return
        if edge.src not in self.nodes or edge.dst not in self.nodes:
            return
        self.edges.append(edge)

    def degree(self) -> dict[str, int]:
        out = {k: 0 for k in self.nodes}
        for e in self.edges:
            out[e.src] = out.get(e.src, 0) + 1
            out[e.dst] = out.get(e.dst, 0) + 1
        return out

    def to_dict(self) -> dict[str, Any]:
        deg = self.degree()
        return {
            "seed": self.seed,
            "nodes": [
                {**p.to_dict(), "key": k, "degree": deg.get(k, 0)}
                for k, p in self.nodes.items()
            ],
            "edges": [e.to_dict() for e in self.edges],
            "coverage": self.coverage.to_dict(),
            "warnings": self.warnings,
            "node_count": len(self.nodes),
            "edge_count": len(self.edges),
            "isolated": sum(1 for d in deg.values() if d == 0),
        }


@dataclass
class Triage:
    """一篇文献的阅读建议。``signals`` 让每句话都能被核对。"""

    key: str
    paper: PaperRef
    verdict: Verdict
    score: float = 0.0
    signals: dict[str, float] = field(default_factory=dict)
    reason: str = ""
    read_for: str = ""
    prereq: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "title": self.paper.title,
            "year": self.paper.year,
            "cited_by": self.paper.cited_by,
            "source": self.paper.source.value,
            "url": self.paper.url,
            "verdict": self.verdict.value,
            "verdict_label": self.verdict.label,
            "score": round(self.score, 3),
            "signals": {k: round(v, 3) for k, v in self.signals.items()},
            "reason": self.reason,
            "read_for": self.read_for,
            "prereq": self.prereq,
        }
