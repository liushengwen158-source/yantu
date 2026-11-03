"""新手引导：对话式基础评估、按方向排路线、概念解释。

没有问卷、没有答题。档位只从用户自己说的话里推，每条结论都能指回原话或
授权目录里的文件。

三件事共用一份画像与一套方向库：

    from research_agent.guide import build_profile, build_roadmap, render_profile

    profile = build_profile("我在做数据库，写过大作业，但没碰过 C")
    print(render_profile(profile))
    roadmap = build_roadmap(profile)
    print(render_roadmap(roadmap))

关键设计：**维度随方向激活**。方向没定时只评估核心四维（编程与调试、
命令行与工程工具、数学基础、科研方法），不会把「深度学习」摆到一个做
系统的学生面前。方向由用户自己说，或从对话里识别；识别结果只是建议。
"""

from .concepts import (
    explain,
    get as get_concept,
    list_concepts,
    render_card,
    render_miss,
    validate as validate_concepts,
)
from .domains import (
    DomainSpec,
    active_dimensions,
    detect_domain,
    get_domain,
    list_domains,
    load_domains,
    required_levels,
    resolve_domain,
)
from .materials import Material, MaterialScan, scan_materials
from .profile import build_profile, build_profile_from_signals, merge_profiles
from .probe import domain_question, next_question, probe_plan
from .render import render_brief, render_profile, render_questions, render_table
from .roadmap import (
    Roadmap,
    RoadmapStage,
    RoadmapTask,
    build_roadmap,
    list_tracks,
    load_roadmap,
    render_brief as render_roadmap_brief,
    render_roadmap,
)
from .signals import load_rules, match_signals, split_clauses
from .spec import (
    CORE_DIMENSIONS,
    Confidence,
    Dimension,
    DimensionState,
    Domain,
    Level,
    Signal,
    SkillProfile,
    Source,
)

__all__ = [
    "CORE_DIMENSIONS",
    "Confidence",
    "Dimension",
    "DimensionState",
    "Domain",
    "DomainSpec",
    "Level",
    "Material",
    "MaterialScan",
    "Roadmap",
    "RoadmapStage",
    "RoadmapTask",
    "Signal",
    "SkillProfile",
    "Source",
    "active_dimensions",
    "build_profile",
    "build_profile_from_signals",
    "build_roadmap",
    "detect_domain",
    "domain_question",
    "explain",
    "get_concept",
    "get_domain",
    "list_concepts",
    "list_domains",
    "list_tracks",
    "load_domains",
    "load_roadmap",
    "load_rules",
    "match_signals",
    "merge_profiles",
    "next_question",
    "probe_plan",
    "render_brief",
    "render_card",
    "render_miss",
    "render_profile",
    "render_questions",
    "render_roadmap",
    "render_roadmap_brief",
    "render_table",
    "required_levels",
    "resolve_domain",
    "scan_materials",
    "split_clauses",
    "validate_concepts",
]
