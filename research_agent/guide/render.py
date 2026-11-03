"""画像渲染。

表格里四样东西必须同时出现，缺一个就没法核对：
**维度 → 档位 / 置信 → 来源（自述还是文件） → 证据 → 为什么这么判**。
"""

from __future__ import annotations

import unicodedata

from .spec import Confidence, DimensionState, SkillProfile, Source

QUOTE_WIDTH = 34
WHY_WIDTH = 30


def _width(text: str) -> int:
    """中文按 2 列算，否则终端里表格会歪。"""
    return sum(2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1 for ch in text)


def _truncate(text: str, limit: int) -> str:
    if _width(text) <= limit:
        return text
    out = ""
    for ch in text:
        if _width(out + ch) > limit - 1:
            break
        out += ch
    return out + "…"


def _pad(text: str, limit: int) -> str:
    return text + " " * max(0, limit - _width(text))


def _source_cell(state: DimensionState, limit: int = 22) -> str:
    """来源必须单独列出来：自述和文件证据的可信度不是一回事。

    文件证据带上路径，否则用户没法回去核对「哪份材料让他这么判」。
    """
    if not state.signals:
        return "—"
    refs = [s.source_ref for s in state.signals if s.source is Source.FILE and s.source_ref]
    has_self = any(s.source is Source.SELF for s in state.signals)
    if refs and has_self:
        return "自述+文件"
    if refs:
        return "文件 " + _truncate(refs[0], limit)
    labels = [src.label for src in sorted({s.source for s in state.signals}, key=lambda x: x.value)]
    return "+".join(labels)


def _evidence_cell(state: DimensionState) -> str:
    if state.conflicts:
        quotes = [s.quote for s in state.signals[:2]]
        return " / ".join(f"「{q}」" for q in quotes)
    sig = state.supporting
    if sig is None:
        return "—"
    if len(state.signals) > 1:
        return f"「{sig.quote}」 等 {len(state.signals)} 条"
    return f"「{sig.quote}」"


def _why_cell(state: DimensionState) -> str:
    if state.conflicts:
        return "两句话矛盾，先确认再定档"
    sig = state.supporting
    if sig is None:
        return "没有证据，不猜"
    return sig.why or "—"


def render_table(profile: SkillProfile) -> str:
    rows = []
    for state in profile.dimensions.values():
        rows.append(
            [
                state.dimension.label,
                state.level.label,
                state.confidence.label,
                _source_cell(state),
                _truncate(_evidence_cell(state), QUOTE_WIDTH),
                _truncate(_why_cell(state), WHY_WIDTH),
            ]
        )

    headers = ["维度", "档位", "置信", "来源", "证据", "为什么这么判"]
    widths = [max(_width(h), max(_width(r[i]) for r in rows)) for i, h in enumerate(headers)]

    def line(cells: list[str]) -> str:
        return "  ".join(_pad(c, w) for c, w in zip(cells, widths)).rstrip()

    out = [line(headers), "  ".join("-" * w for w in widths)]
    out.extend(line(r) for r in rows)
    return "\n".join(out)


def render_profile(profile: SkillProfile) -> str:
    """完整文本报告：表格 + 整体结论 + 缺口 + 建议追问。"""
    from .probe import next_question

    parts = [render_table(profile), ""]
    parts.append(f"覆盖 {profile.coverage} 维 · 整体档位：{profile.overall.label}")

    if profile.scanned:
        sc = profile.scanned
        parts.append(
            f"授权材料：{sc.get('papers', 0)} 篇 PDF（{sc.get('papers_with_notes', 0)} 篇有同名笔记）"
            f" · {sc.get('notebooks', 0)} 个 notebook · {sc.get('notes', 0)} 份笔记"
        )

    if profile.file_evidence:
        parts.append("")
        parts.append("文件证据（只能证明材料在，不能证明掌握）：")
        for s in profile.file_evidence[:8]:
            parts.append(f"  - [{s.dimension.label}] {s.quote}")

    if profile.conflicts:
        parts.append("")
        parts.append("需要确认的矛盾：")
        for state in profile.conflicts:
            for c in state.conflicts:
                parts.append(f"  - [{state.dimension.label}] {c}")

    caveats = [(d, c) for d in profile.dimensions.values() for c in d.caveats]
    if caveats:
        parts.append("")
        parts.append("同一维度里的其他明确说法（不改档位，但值得知道）：")
        for d, c in caveats[:6]:
            parts.append(f"  - [{d.dimension.label}] {c}")

    if profile.pending:
        names = "、".join(s.dimension.label for s in profile.pending)
        parts.append(f"待确认的维度：{names}")

    if profile.unknown:
        names = "、".join(s.dimension.label for s in profile.unknown)
        parts.append(f"还没有证据的维度：{names}")

    for note in profile.notes:
        parts.append(f"注意：{note}")

    q = next_question(profile)
    if q:
        parts.append("")
        parts.append(f"建议下一句问：{q['question']}")
        parts.append(f"（问这个是因为 {q['reason']}）")

    return "\n".join(parts)


def render_brief(profile: SkillProfile, limit: int = 400) -> str:
    """给 agent 用的一句话摘要，避免把整张表塞进上下文。"""
    known = profile.known
    if not known:
        return "还不知道用户水平：对话里没有可用证据。"
    items = "、".join(f"{s.dimension.label}{s.level.label}" for s in known[:4])
    text = f"已知 {profile.coverage} 维（{items}），整体{profile.overall.label}"
    if profile.conflicts:
        text += f"，另有 {len(profile.conflicts)} 个维度证据矛盾待确认"
    if profile.unknown:
        text += f"，缺 {len(profile.unknown)} 维信息"
    return text[:limit]
