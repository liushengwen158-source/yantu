"""证据汇总成六维画像。

四条规则（都是为了「结论可核对」）：

- **强证据定档**：有明确陈述时，档位取最强的那条；弱证据只加权重、不抬档位
- **矛盾才不猜**：只有「明确说会」和「明确说不会」同时出现，才算矛盾，标待确认；
  从「写过爬虫」到「自己写过库」属于正常成长，不算冲突
- **置信看证据条数**：两条独立明确陈述才算高置信，一条算中，全是含糊表述算低
- **没有证据就是未知**：不是 0 分，也不按「不会」处理
"""

from __future__ import annotations

from . import signals as signal_mod
from .spec import Confidence, Dimension, DimensionState, Level, Signal, SkillProfile

STRONG_WEIGHT = 2
WEAK_WEIGHT = 1

# 明确说会（>= 能干活）与明确说不会（<= 从零）同时出现，才是真矛盾
HIGH_BAR = Level.WORKING
LOW_BAR = Level.BEGINNER


def _aggregate(dimension: Dimension, sigs: list[Signal]) -> DimensionState:
    state = DimensionState(dimension=dimension, signals=sigs)
    if not sigs:
        state.note = "对话里没有相关信息"
        return state

    strong = [s for s in sigs if s.strength == "strong"]
    weak = [s for s in sigs if s.strength != "strong"]

    weights: dict[Level, int] = {}
    for s in sigs:
        w = STRONG_WEIGHT if s.strength == "strong" else WEAK_WEIGHT
        weights[s.level] = weights.get(s.level, 0) + w
    total = sum(weights.values())

    high = [s for s in strong if s.level >= HIGH_BAR]
    low = [s for s in strong if s.level <= LOW_BAR and s.polarity == "negative"]
    if high and low:
        state.level = Level.UNKNOWN
        state.confidence = Confidence.LOW
        state.conflicts = [
            f"「{low[0].quote}」（{low[0].level.label}）与「{high[0].quote}」（{high[0].level.label}）"
        ]
        state.note = "证据互相矛盾，先确认再定档，不替你猜"
        return state

    # 强证据定档；没有强证据才看弱证据
    pool = strong or weak
    state.level = max(s.level for s in pool)

    # 档位不同但同样是明确陈述的，作为「补充证据」留着，不吞掉
    for s in strong:
        if s.level != state.level:
            state.caveats.append(f"「{s.quote}」（{s.level.label}）")

    if len(strong) >= 2:
        state.confidence = Confidence.HIGH
    elif strong:
        state.confidence = Confidence.MEDIUM
    elif total >= 2:
        state.confidence = Confidence.MEDIUM
        state.note = "只有含糊表述，按低档参考"
    else:
        state.confidence = Confidence.LOW
        state.note = "只有一句含糊表述，参考价值有限"

    return state


def build_profile_from_signals(sigs: list[Signal], text_length: int = 0) -> SkillProfile:
    by_dim: dict[Dimension, list[Signal]] = {d: [] for d in Dimension}
    for s in sigs:
        by_dim[s.dimension].append(s)
    profile = SkillProfile(
        dimensions={d: _aggregate(d, by_dim[d]) for d in Dimension},
        text_length=text_length,
    )
    if not sigs:
        profile.notes.append("这段对话里没有可用的能力证据：标未知，不按 0 分处理。")
    return profile


def build_profile(text: str, rules: list[dict] | None = None) -> SkillProfile:
    """从一段（可跨多轮的）对话文本构建画像。"""
    sigs = signal_mod.extract(text or "", rules)
    return build_profile_from_signals(sigs, text_length=len(text or ""))


def merge_profiles(*profiles: SkillProfile) -> SkillProfile:
    """合并多轮结果。

    后说的更接近现状：如果后面已经说清「用过、会调」，前面那句「没学过」
    就不再作为矛盾证据，只记一条说明。旧结论不会盖住新证据。
    """
    by_dim: dict[Dimension, list[tuple[int, Signal]]] = {}
    for idx, p in enumerate(profiles):
        for state in p.dimensions.values():
            for s in state.signals:
                by_dim.setdefault(s.dimension, []).append((idx, s))

    kept: list[Signal] = []
    seen: set[tuple[str, str]] = set()
    dropped = 0
    for items in by_dim.values():
        later_known = max(
            (i for i, s in items if s.strength == "strong" and s.level >= Level.FOUNDATION),
            default=-1,
        )
        for idx, s in items:
            if idx < later_known and s.level <= LOW_BAR and s.polarity == "negative":
                dropped += 1
                continue
            key = (s.rule_id, s.quote)
            if key in seen:
                continue
            seen.add(key)
            kept.append(s)

    profile = build_profile_from_signals(kept, text_length=sum(p.text_length for p in profiles))
    if dropped:
        profile.notes.append(f"{dropped} 条早期陈述已被后续对话覆盖，按最新说法为准")
    return profile
