"""证据汇总成六维画像。

四条规则（都是为了「结论可核对」）：

- **强证据定档**：有明确陈述时，档位取最强的那条；弱证据只加权重、不抬档位
- **矛盾才不猜**：只有「明确说会」和「明确说不会」同时出现，才算矛盾，标待确认；
  从「写过爬虫」到「自己写过库」属于正常成长，不算冲突
- **置信看证据条数**：两条独立明确陈述才算高置信，一条算中，全是含糊表述算低
- **文件不定档**：授权扫描到的材料只作旁证，单靠文件给的结论一律低置信，
  与自述不一致时作为「对撞」摆出来，不替用户下结论
- **没有证据就是未知**：不是 0 分，也不按「不会」处理
"""

from __future__ import annotations

from typing import Any

from . import signals as signal_mod
from .spec import Confidence, Dimension, DimensionState, Level, Signal, SkillProfile, Source

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

    # 自述与文件分开算：文件只能证明「材料在」，不参与定档，只做旁证
    self_sigs = [s for s in sigs if s.source is not Source.FILE]
    file_sigs = [s for s in sigs if s.source is Source.FILE]

    if not self_sigs:
        state.level = max(s.level for s in file_sigs)
        state.confidence = Confidence.LOW
        state.note = "只有文件证据：材料在，不等于读过或会做，按低置信参考"
        return state

    strong = [s for s in self_sigs if s.strength == "strong"]
    weak = [s for s in self_sigs if s.strength != "strong"]

    weights: dict[Level, int] = {}
    for s in self_sigs:
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

    # 文件证据与结论不一致时摆出来，让用户自己判断是「材料不对」还是「自述高估」
    for s in file_sigs:
        if s.level != state.level:
            state.caveats.append(f"「{s.quote}」（文件证据，{s.level.label}）")

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


def build_profile(
    text: str,
    rules: list[dict] | None = None,
    scan: Any | None = None,
) -> SkillProfile:
    """从一段（可跨多轮的）对话文本构建画像。

    ``scan`` 是 ``materials.MaterialScan``，只在用户授权时才传进来；
    它的证据以 ``source=FILE`` 并入，不参与定档。
    """
    sigs = signal_mod.extract(text or "", rules)
    profile = build_profile_from_signals(sigs, text_length=len(text or ""))
    if scan is not None:
        merged = list(sigs) + list(scan.signals)
        profile = build_profile_from_signals(merged, text_length=len(text or ""))
        profile.scanned = scan.summary()
        profile.notes.extend(scan.notes)
        if not text.strip() and scan.signals:
            profile.notes.append("没有对话内容，只有文件证据，档位只作参考。")
    return profile



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
