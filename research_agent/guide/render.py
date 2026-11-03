"""把画像渲染成人能读的文本。

两条排版原则：
- 每一行都带「证据」和「来源」，用户能顺着引文回到自己说过的话
- 未知就说未知，不用「暂无数据」之类含糊话遮住

方向相关的维度名不是固定的，表格按当前方向实际激活的维度生成，
不会出现「深度学习：未知」摆在一个做系统的学生面前。
"""

from __future__ import annotations

from typing import Iterable

from .spec import Dimension, DimensionState, SkillProfile, Source


def _source_cell(state: DimensionState) -> str:
    if not state.signals:
        return "—"
    kinds = {s.source.label for s in state.signals}
    return "+".join(sorted(kinds))


def _evidence_cell(state: DimensionState) -> str:
    signal = state.supporting
    if signal is None:
        return "—"
    quote = signal.quote.replace("|", "\\|").replace("\n", " ")
    if len(quote) > 40:
        quote = quote[:40] + "…"
    ref = f"（{signal.source_ref}）" if signal.source_ref else ""
    return f"「{quote}」{ref}"


def render_table(profile: SkillProfile) -> str:
    """六列表格：维度、档位、该方向要求、置信、来源、证据。"""
    lines = [
        "| 维度 | 档位 | 方向要求 | 置信 | 来源 | 证据 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for state in profile.dimensions.values():
        required = state.required_label if state.required else "—"
        if state.gap and state.is_known:
            required += " ↑"
        lines.append(
            "| {dim} | {level} | {req} | {conf} | {src} | {ev} |".format(
                dim=state.dimension.label,
                level=state.level.label if state.is_known else "未知",
                req=required,
                conf=state.confidence.label,
                src=_source_cell(state),
                ev=_evidence_cell(state),
            )
        )
    return "\n".join(lines)


def render_profile(profile: SkillProfile) -> str:
    """完整报告：方向 → 表格 → 矛盾与旁证 → 文件证据 → 下一步。"""
    out: list[str] = []
    out.append(f"# 基础评估：{profile.domain.label}")
    if profile.domain_evidence:
        out.append(f"> 方向依据你原话：「{profile.domain_evidence}」")
    out.append(
        f"整体档位 **{profile.overall.label}**（取已知维度的中位数，不给平均分）；"
        f"已确认 {profile.coverage} 个维度。"
    )
    out.append(render_table(profile))

    if profile.notes:
        out.append("")
        for note in profile.notes:
            out.append(f"- {note}")

    conflicts = [d for d in profile.dimensions.values() if d.conflicts]
    if conflicts:
        out.append("")
        out.append("## 说法不一致，需要你确认")
        for state in conflicts:
            for item in state.conflicts:
                out.append(f"- **{state.dimension.label}**：{item}")

    caveats = [d for d in profile.dimensions.values() if d.caveats]
    if caveats:
        out.append("")
        out.append("## 旁证（不参与定档）")
        for state in caveats:
            for item in state.caveats[:2]:
                out.append(f"- **{state.dimension.label}**：{item}")

    files = profile.file_evidence
    if files:
        out.append("")
        out.append("## 文件证据（来自你授权扫描的目录）")
        for signal in files[:8]:
            out.append(
                f"- {signal.source_ref} → {signal.dimension.label}"
                f"（{signal.level.label}，仅旁证）：{signal.why}"
            )
        if len(files) > 8:
            out.append(f"- 其余 {len(files) - 8} 条略。")

    out.append("")
    out.append("## 说明")
    out.append("- 这一页只依据你原话和你授权扫描的目录，没有跑测评、没有判题。")
    out.append("- 没提到的维度就是「不知道」，不等于「不会」；想补就说一句，我重排。")
    out.append("- 判定不对时直接说你的实际情况，以你说的为准。")
    return "\n".join(out)


def render_brief(profile: SkillProfile) -> str:
    """一句话摘要，给 CLI 与工具返回用。"""
    if not profile.known:
        return f"方向：{profile.domain.label}；还没有可定档的证据。"

    head = (
        f"方向：{profile.domain.label}；整体 {profile.overall.label}；"
        f"已确认 {profile.coverage}。"
    )
    below = [
        d for d in profile.dimensions.values() if d.gap and d.is_known
    ]
    unknown = [d for d in profile.dimensions.values() if d.gap and not d.is_known]
    parts: list[str] = []
    if below:
        parts.append(
            "低于要求："
            + "、".join(f"{d.dimension.label}（{d.level.label}）" for d in below[:4])
        )
    if unknown:
        parts.append(
            "还没确认：" + "、".join(d.dimension.label for d in unknown[:4])
        )
    if not parts:
        return head + "该方向要求的维度都已达到要求。"
    return head + "；".join(parts) + "。"


def render_questions(questions: Iterable[dict]) -> str:
    """把待问的问题按顺序排出来，每条都带理由。"""
    items = list(questions)
    if not items:
        return "没有需要追问的维度了。"
    lines = ["按优先级，接下来可以问："]
    for index, item in enumerate(items, 1):
        lines.append(f"{index}. {item['question']}")
        lines.append(f"   理由：{item['reason']}")
        if item.get("options"):
            lines.append(f"   可选：{'、'.join(item['options'])}")
    return "\n".join(lines)


def dimension_line(state: DimensionState) -> str:
    """单行展示，供路线里的「前置」引用。"""
    if not state.is_known:
        return f"{state.dimension.label}：未知"
    ref = state.supporting
    suffix = f"（依据「{ref.quote}」）" if ref and ref.source is Source.SELF else ""
    return f"{state.dimension.label}：{state.level.label}{suffix}"


def dim_label(dimension: Dimension | str) -> str:
    if isinstance(dimension, Dimension):
        return dimension.label
    try:
        return Dimension(dimension).label
    except ValueError:
        return str(dimension)
