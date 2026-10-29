"""论文复现：环境需求 vs 本机环境的差距分析。

对外入口：
    from research_agent.repro import scan_repo, collect_profile, diff, render_matrix

流程：
    scan_repo(repo)        仓库 → PaperEnvSpec（论文要什么）
    collect_profile()      本机 → LocalEnvProfile（我有什么）
    diff(spec, profile)    → FeasibilityReport（差在哪）
"""

from .spec import (
    Category,
    DiffItem,
    EnvRequirement,
    Evidence,
    EvidenceLevel,
    FeasibilityReport,
    LocalEnvProfile,
    PaperEnvSpec,
    Verdict,
)
from .scanner import scan_repo
from .paper_adapter import paper_requirements, to_requirements
from .profile import collect_profile
from .diff import diff
from .render import render_matrix, render_report

__all__ = [
    "Category",
    "Evidence",
    "EvidenceLevel",
    "EnvRequirement",
    "PaperEnvSpec",
    "LocalEnvProfile",
    "DiffItem",
    "FeasibilityReport",
    "Verdict",
    "scan_repo",
    "paper_requirements",
    "to_requirements",
    "collect_profile",
    "diff",
    "render_matrix",
    "render_report",
]
