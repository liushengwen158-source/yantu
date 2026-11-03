"""一次只问一个问题。

评估不做问卷、不判题：只根据用户自己说的话下判断，缺的信息通过对话补。
所以这里只产出「下一个该问什么」，并且给每个问题附上问它的理由——
用户有权说「这个不用问」，那时按用户的说法重排。

方向**不由系统先问**。用户想说的时候自己会说，没说就按通用主干走；
``domain_question()`` 只在用户主动问「我该选什么方向」时才用。
"""

from __future__ import annotations

from typing import Any

from .domains import list_domains
from .spec import Dimension, DimensionState, Level, SkillProfile

# 每个维度一个问题。都是开放式的：答完能继续判断，不是选择题。
DIMENSION_QUESTIONS: dict[Dimension, str] = {
    Dimension.CODING: "你平时写代码写到什么程度——是照着教程抄，还是能自己从头写一个能跑的东西？",
    Dimension.TOOLS: "你在命令行里干活熟练吗，比如自己装环境、跑脚本、连服务器？",
    Dimension.MATH: "数学这块你到哪一步了，比如能不能手推一个梯度或者看懂论文里的公式？",
    Dimension.RESEARCH: "你读过论文吗，是随便翻翻还是按方法读、也复现过别人的代码？",
    Dimension.SYSTEMS: "你写过 C 或者碰过内存、多线程这类底层的东西吗？",
    Dimension.DATA: "你写过 SQL 或者用 pandas 处理过真实数据吗？",
    Dimension.SECURITY: "你接触过逆向、漏洞复现或者 CTF 这类东西吗？",
    Dimension.THEORY: "你写过证明、做过归约吗，还是主要写代码？",
    Dimension.ML: "传统机器学习你跑过吗，比如用 sklearn 做过分类或回归？",
    Dimension.DL: "深度学习这块你到哪一步了——装过框架、跑过训练，还是改过网络结构？",
}

DIMENSION_REASONS: dict[Dimension, str] = {
    Dimension.CODING: "编程是所有方向的地基，先确定这一维才能决定从哪一层开始排。",
    Dimension.TOOLS: "工具不熟的话，后面每个实验都会卡在环境上，不是能力问题。",
    Dimension.MATH: "不同方向对数学的要求差很多，这一维决定要不要先补公式。",
    Dimension.RESEARCH: "读论文和做实验的方法可以单独补，但得先知道你有没有做过。",
    Dimension.SYSTEMS: "这个方向的基本功在底层，不确认就没法判断该不该从 C 开始。",
    Dimension.DATA: "数据与查询能力是这个方向的入口，先确认再排任务。",
    Dimension.SECURITY: "安全方向的门槛在二进制与协议，需要单独确认。",
    Dimension.THEORY: "理论方向看的是证明能力，和写代码是两套本事。",
    Dimension.ML: "这个方向要先有机器学习的基本概念，之后才好谈模型。",
    Dimension.DL: "框架与训练经验决定你能不能直接上手复现，还是要先补一轮。",
}

DOMAIN_OPTIONS = [spec.name for spec in list_domains()]

DOMAIN_QUESTION = (
    "你现在大概在做哪个方向？不同方向要补的东西差很多，"
    "先确认这个，后面的路线才不会给你一堆用不上的任务。"
)
DOMAIN_REASON = "方向决定评估哪些维度、编排哪些阶段；方向没定时只评估核心四维。"


def _level_label(value: int) -> str:
    try:
        return Level(int(value)).label
    except ValueError:
        return ""


def _priority(state: DimensionState) -> tuple[int, int, int]:
    """越小越先问：矛盾 > 方向要求但缺证据 > 低于方向要求 > 核心维缺失。"""
    if state.conflicts:
        return (0, 0, int(state.level))
    if state.required and not state.is_known:
        return (1, -state.required, 0)
    if state.required and state.gap:
        return (2, -state.required, int(state.level))
    if state.dimension.is_core:
        return (3, 0, int(state.level))
    return (4, 0, int(state.level))


def _question_for(state: DimensionState) -> dict[str, Any]:
    if state.conflicts:
        question = "关于这一维，前面有两种说法，以哪个为准？"
        reason = "；".join(state.conflicts)
    else:
        question = DIMENSION_QUESTIONS[state.dimension]
        if state.required and state.gap:
            current = "没有证据" if not state.is_known else f"只到「{state.level.label}」"
            reason = (
                f"这个方向要求这一维至少到「{_level_label(state.required)}」，"
                f"现在{current}，先问清楚再决定补不补。"
            )
        else:
            reason = DIMENSION_REASONS[state.dimension]
    return {
        "target": state.dimension.value,
        "target_label": state.dimension.label,
        "question": question,
        "reason": reason,
        "options": [],
        "evidence": state.supporting.quote if state.supporting else "",
    }


def domain_question() -> dict[str, Any]:
    """方向清单。只在用户主动问「有哪些方向 / 我该选什么」时给出。"""
    return {
        "target": "domain",
        "target_label": "方向",
        "question": DOMAIN_QUESTION,
        "reason": DOMAIN_REASON,
        "options": DOMAIN_OPTIONS,
        "evidence": "",
    }


def next_question(profile: SkillProfile) -> dict[str, Any] | None:
    """返回下一个该问的问题；没有需要问的返回 None。

    方向不由系统先问：用户想说的时候自己会说。方向未定时只问核心四维，
    不拿「你在做哪个方向」拦在评估前面。
    """
    if profile is None:
        return None

    pending = [d for d in profile.dimensions.values() if d.needs_probe]
    if not pending:
        return None
    pending.sort(key=_priority)
    return _question_for(pending[0])


def probe_plan(profile: SkillProfile, limit: int = 3) -> list[dict[str, Any]]:
    """按优先级给出接下来几个问题，供 CLI 一次打印。"""
    if profile is None:
        return []
    pending = [d for d in profile.dimensions.values() if d.needs_probe]
    pending.sort(key=_priority)
    return [_question_for(state) for state in pending[: max(1, limit)]]
