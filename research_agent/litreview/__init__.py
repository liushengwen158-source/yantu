"""论文检索与文献关联：找到 → 对齐 → 排优先级 → 画出来。

对外只暴露五件事：

    search()          问题 → Retrieval（多源召回 + 本地过滤 + 身份对齐 + 覆盖度）
    expand()          种子论文 → PaperGraph（上游 / 下游 / 近邻 / 文献耦合）
    triage()          一批文献 → 四档阅读建议（每条都带可核对的信号）
    render.*          图与报告（Markdown / 单文件 HTML）
    search.save_snapshot / load_snapshot   检索现场落盘与重放

刻意**不认识** ``EnvRequirement``（那是 repro 的）与 ``PaperDoc``（那是 document 的）。
三者之间靠外层拼装，而不是互相 import——否则任何一层的改动都会拖垮另外两层。
"""

from __future__ import annotations

from .graph import expand
from .identity import key_of, normalize_arxiv, normalize_doi, resolve, title_fingerprint
from .search import load_snapshot, relevance, save_snapshot, search, terms_of
from .spec import (
    Confidence,
    Coverage,
    Edge,
    Identity,
    PaperGraph,
    PaperRef,
    Relation,
    Retrieval,
    Source,
    SourceStatus,
    Triage,
    Verdict,
)
from .triage import triage

__all__ = [
    "Confidence",
    "Coverage",
    "Edge",
    "Identity",
    "PaperGraph",
    "PaperRef",
    "Relation",
    "Retrieval",
    "Source",
    "SourceStatus",
    "Triage",
    "Verdict",
    "expand",
    "key_of",
    "load_snapshot",
    "normalize_arxiv",
    "normalize_doi",
    "relevance",
    "resolve",
    "save_snapshot",
    "search",
    "terms_of",
    "title_fingerprint",
    "triage",
]
