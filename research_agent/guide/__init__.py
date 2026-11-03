"""新手引导：对话式基础评估。

没有问卷、没有答题。等级只从用户自己说的话里推，每条结论都能指回原话。

用法：

    from research_agent.guide import build_profile, render_profile

    profile = build_profile("我写过爬虫，但没写过类。torch 装了，cuda 一直是 false。")
    print(render_profile(profile))

后续的学习路线（roadmap）以 ``SkillProfile`` 为输入，概念解释（concepts）
以路线上的卡点为触发条件，因此三者共用同一份画像，不各自维护一份判断。
"""

from .materials import Material, MaterialScan, scan_materials
from .profile import build_profile, build_profile_from_signals, merge_profiles
from .probe import next_question
from .render import render_brief, render_profile, render_table
from .signals import extract, load_rules, split_clauses, split_sentences
from .spec import (
    Confidence,
    Dimension,
    DimensionState,
    Level,
    Signal,
    SkillProfile,
    Source,
)

__all__ = [
    "Confidence",
    "Dimension",
    "DimensionState",
    "Level",
    "Material",
    "MaterialScan",
    "Signal",
    "SkillProfile",
    "Source",
    "build_profile",
    "build_profile_from_signals",
    "merge_profiles",
    "extract",
    "load_rules",
    "next_question",
    "render_brief",
    "render_profile",
    "render_table",
    "scan_materials",
    "split_clauses",
    "split_sentences",
]
