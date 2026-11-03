"""从证据汇总出画像。

汇总的规则都写死在代码里，且每条都对应一句能被用户核对的话：

1. **强证据定档**：明确陈述（strong）里档位最高的那条决定档位。
2. **负向优先**：同一维度同时出现「能干活」和「完全不会」，判为冲突，
   不取平均、也不挑好看的那个，档位置为未知并让调用方去问清楚。
3. **文件证据不定档**：文件只能证明「材料在」，不能证明「掌握了」，
   单靠文件把档位停在未知，置信度给低，并在备注里写清原因。
4. **弱证据只作旁证**：自评类的话（「我数学还行」）放 caveats，不进档位。
5. **只评估用得上的维度**：方向未定时只有核心四维，方向维不出现。
"""

from __future__ import annotations

from typing import Iterable

from .domains import active_dimensions, detect_domain, required_levels
from .materials import MaterialScan
from .signals import match_signals
from .spec import (
    CORE_DIMENSIONS,
    Confidence,
    Dimension,
    DimensionState,
    Level,
    Signal,
    SkillProfile,
    Source,
)

# 冲突判定门槛：一边说能干活以上，一边说从零，才算打架。
HIGH_BAR = Level.WORKING
LOW_BAR = Level.BEGINNER

MIN_TEXT_FOR_CONFIDENCE = 30


def _aggregate(state: DimensionState, signals: list[Signal]) -> DimensionState:
    """把同一维度的证据汇总成一个状态。"""
    state.signals = list(signals)
    if not signals:
        return state

    self_strong = [
        s for s in signals if s.source is Source.SELF and s.strength == "strong"
    ]
    positives = [s for s in self_strong if s.polarity == "positive"]
    negatives = [s for s in self_strong if s.polarity == "negative"]
    weak = [s for s in signals if s.strength != "strong" or s.source is Source.FILE]

    if positives:
        top = max(positives, key=lambda s: int(s.level))
        state.level = top.level
        state.confidence = (
            Confidence.HIGH if len(positives) >= 2 else Confidence.MEDIUM
        )
    elif negatives:
        top = max(negatives, key=lambda s: int(s.level))
        state.level = top.level
        state.confidence = (
            Confidence.HIGH if len(negatives) >= 2 else Confidence.MEDIUM
        )

    # 一边「能干活」一边「完全不会」，是矛盾，不是互补
    if positives and negatives:
        hi = max(positives, key=lambda s: int(s.level))
        lo = min(negatives, key=lambda s: int(s.level))
        if int(hi.level) >= int(HIGH_BAR) and int(lo.level) <= int(LOW_BAR):
            state.level = Level.UNKNOWN
            state.confidence = Confidence.LOW
            state.conflicts.append(
                f"「{hi.quote}」与「{lo.quote}」指向不同水平，需要确认以哪次为准。"
            )

    # 只有文件证据：说明材料在，不说明水平
    if not self_strong:
        file_only = [s for s in signals if s.source is Source.FILE]
        if file_only and state.level is Level.UNKNOWN:
            state.confidence = Confidence.LOW
            state.note = (
                f"只有文件旁证（{file_only[0].source_ref}），没有你本人的说明，"
                "这一维暂不判档。"
            )
        elif state.level is Level.UNKNOWN:
            state.confidence = Confidence.LOW
            state.note = "只有含糊表述，没有明确说法，暂不判档。"

    # 与结论档位不同的证据留在旁证里，不丢掉
    for signal in weak:
        if signal.level != state.level and signal not in state.signals[:1]:
            text = f"{signal.quote}（{signal.level.label}，{signal.source.label}）"
            if signal.why:
                text += f"：{signal.why}"
            if text not in state.caveats:
                state.caveats.append(text)

    return state


def build_profile_from_signals(
    signals: Iterable[Signal],
    *,
    text: str = "",
    domain=None,
) -> SkillProfile:
    """从已有证据构建画像。用于多轮对话合并。"""
    signal_list = list(signals)
    resolved_domain = domain or detect_domain(text)[0]
    profile = SkillProfile(domain=resolved_domain, text_length=len(text or ""))
    required = required_levels(resolved_domain)
    active = active_dimensions(resolved_domain)

    for dim in active:
        state = DimensionState(dimension=dim, required=required.get(dim, 0))
        profile.dimensions[dim] = _aggregate(
            state, [s for s in signal_list if s.dimension is dim]
        )

    if resolved_domain.value == "unknown":
        profile.notes.append(
            "方向还没定，所以只评估核心四维（编程、工具、数学、科研方法）；"
            "方向相关的维度先不评估，免得让你补用不上的东西。"
        )
    else:
        profile.notes.append(
            f"按「{resolved_domain.label}」方向的要求评估，"
            f"只看这个方向用得上的 {len(active)} 个维度。"
        )
    if len(text or "") < MIN_TEXT_FOR_CONFIDENCE:
        profile.notes.append("这段描述比较短，结论可能偏保守；多说几句会准一些。")
    return profile


def build_profile(
    text: str,
    *,
    scan: MaterialScan | None = None,
    domain=None,
    extra_signals: Iterable[Signal] | None = None,
) -> SkillProfile:
    """从一段自述（可选带授权目录）构建画像。

    ``domain`` 给定时以用户选的方向为准，不再从文本猜——
    用户自己说的方向永远优先于关键词匹配。
    """
    detected, evidence = detect_domain(text)
    resolved = domain or detected
    if domain is not None:
        evidence = ""
    elif detected.value != "unknown":
        evidence = evidence

    required = required_levels(resolved)
    active = active_dimensions(resolved)

    signals = match_signals(text, dimensions=active)
    if scan is not None:
        signals.extend(s for s in scan.signals if s.dimension in active)
    if extra_signals:
        signals.extend(s for s in extra_signals if s.dimension in active)

    profile = build_profile_from_signals(signals, text=text, domain=resolved)
    profile.domain_evidence = evidence
    profile.dimensions = {
        dim: state for dim, state in profile.dimensions.items() if dim in active
    }
    if scan is not None:
        profile.scanned = scan.summary()
    for dim, level in required.items():
        state = profile.dimensions.get(dim)
        if state is not None:
            state.required = level
    return profile


def merge_profiles(profiles: Iterable[SkillProfile]) -> SkillProfile:
    """合并多轮对话的画像。

    合并的是证据，不是分数——同一句话被说两遍不该变成两票，
    所以按 ``rule_id`` 去重。
    """
    items = [p for p in profiles if p is not None]
    if not items:
        return SkillProfile()
    base = items[-1]
    seen: dict[str, Signal] = {}
    for profile in items:
        for state in profile.dimensions.values():
            for signal in state.signals:
                key = f"{signal.rule_id}|{signal.source_ref}"
                prev = seen.get(key)
                if prev is None or int(signal.level) > int(prev.level):
                    seen[key] = signal
    merged = build_profile_from_signals(
        list(seen.values()), text="", domain=base.domain
    )
    merged.notes = list(base.notes)
    merged.text_length = sum(p.text_length for p in items)
    merged.scanned = base.scanned
    merged.domain_evidence = next(
        (p.domain_evidence for p in reversed(items) if p.domain_evidence), ""
    )
    required = required_levels(base.domain)
    for dim, state in merged.dimensions.items():
        state.required = required.get(dim, 0)
    return merged


def core_dimensions() -> tuple[Dimension, ...]:
    return CORE_DIMENSIONS
