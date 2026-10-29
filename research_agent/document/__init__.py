"""文档 → 可定位文本 → 环境约束候选。

本包只负责「读」，不负责「判」：

    extract()          文档 → PaperDoc（页 / 块 / 文本 / 章节）
    extract_candidates() PaperDoc → [Candidate]（值 + 引文 + 是否接地）
    verify_quote()      引文 → 能否在该页原文定位到

它**不认识** ``EnvRequirement`` / ``EvidenceLevel``，也不知道什么叫「复现可行性」。
把候选投影成仓库契约对象是 ``research_agent.repro.paper_adapter`` 的事。
这条单向依赖是刻意的：另外两条分支（学习导读 / 论文调研）可以复用本包而不拖进判定链。

典型用法：

    from research_agent.document import extract, extract_candidates

    doc = extract("paper.pdf")
    cands, warnings = extract_candidates(doc)      # 无 API Key 时只用规则路
"""

from __future__ import annotations

from .extract import (
    BLOCK_KINDS,
    CONSTRAINT_SECTIONS,
    Block,
    DocumentError,
    PaperDoc,
    classify,
    extract,
    is_heading,
)
from .llm_extract import Candidate, ALLOWED_IDS, extract_candidates, extract_by_rules, group_by_id
from .verify import (
    STATE_ABSENT,
    STATE_FAILED,
    STATE_HIT,
    STATE_NA,
    STATES,
    Coverage,
    FieldOutcome,
    VerifyResult,
    absent_or_failed,
    build_coverage,
    locate_quote,
    normalize,
    suspect_blocks,
    verify_quote,
)

__all__ = [
    "ALLOWED_IDS",
    "BLOCK_KINDS",
    "CONSTRAINT_SECTIONS",
    "Block",
    "Candidate",
    "Coverage",
    "DocumentError",
    "FieldOutcome",
    "PaperDoc",
    "STATES",
    "STATE_ABSENT",
    "STATE_FAILED",
    "STATE_HIT",
    "STATE_NA",
    "VerifyResult",
    "absent_or_failed",
    "build_coverage",
    "classify",
    "extract",
    "extract_by_rules",
    "extract_candidates",
    "group_by_id",
    "is_heading",
    "locate_quote",
    "normalize",
    "suspect_blocks",
    "verify_quote",
]
