"""报告渲染：差距矩阵与完整报告。

矩阵按显示宽度对齐（中文占 2 列），而不是按字符数，
否则中英混排会错位。
"""

from __future__ import annotations

import unicodedata

from .spec import Category, DiffItem, FeasibilityReport, Verdict

_VERDICT_MARK = {
    Verdict.BLOCK: "BLOCK",
    Verdict.CONFLICT: "CONFLICT",
    Verdict.DEGRADE: "DEGRADE",
    Verdict.ADD: "ADD",
    Verdict.UNKNOWN: "UNKNOWN",
    Verdict.OK: "OK",
}


def _width(text: str) -> int:
    """终端显示宽度：全角字符占 2 列。"""
    w = 0
    for ch in str(text):
        w += 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1
    return w


def _pad(text: str, width: int) -> str:
    return str(text) + " " * max(0, width - _width(str(text)))


def _truncate(text: str, limit: int) -> str:
    """按显示宽度截断，超出加省略号。"""
    if _width(text) <= limit:
        return text
    out, w = "", 0
    for ch in str(text):
        cw = 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1
        if w + cw > limit - 1:
            break
        out += ch
        w += cw
    return out + "…"


def _evidence_text(item: DiffItem) -> str:
    ev = item.requirement.evidence
    return ev.describe() if ev else "无来源"


def render_matrix(report: FeasibilityReport) -> str:
    """终端用的差距矩阵。"""
    if not report.items:
        return "没有可判定的需求项（仓库里可能没有可识别的依赖声明）。"

    rows = [
        (item.requirement.id, item.requirement.requirement_text(),
         item.local_value or "-", _VERDICT_MARK[item.verdict], _evidence_text(item))
        for item in report.items
    ]
    headers = ("需求项", "论文要求", "本机", "判定", "证据")
    widths = [
        max(_width(headers[i]), max(_width(r[i]) for r in rows)) for i in range(5)
    ]
    widths[3] = max(widths[3], 9)

    sep = "  "
    lines = [sep.join(_pad(h, widths[i]) for i, h in enumerate(headers))]
    lines.append(sep.join("-" * widths[i] for i in range(5)))
    for r in rows:
        lines.append(sep.join(_pad(v, widths[i]) for i, v in enumerate(r)))

    c = report.counts
    lines.append("")
    lines.append(
        f"结论：阻塞 {c['block']} · 冲突 {c['conflict']} · 需降级 {c['degrade']} · "
        f"可补齐 {c['add']} · 待确认 {c['unknown']} · 满足 {c['ok']}"
    )
    lines.append(f"可行性：{report.overall.label}（{_VERDICT_MARK[report.overall]}）—— {report.headline()}")
    return "\n".join(lines)


def render_report(report: FeasibilityReport) -> str:
    """Markdown 完整报告。"""
    spec, profile = report.spec, report.profile
    lines: list[str] = ["# 论文复现可行性报告", ""]

    lines.append(f"- 仓库：`{spec.repo_path}`")
    lines.append(f"- 扫描文件：{', '.join(f'`{f}`' for f in spec.scanned_files) or '无'}")
    lines.append(f"- 本机：{profile.platform}")
    lines.append(
        f"- 本机规格：{profile.cpu_count} 核 / {profile.ram_gb or '-'} GB 内存 / "
        f"{profile.gpu_name or '无 GPU'}"
        + (f" ×{profile.gpu_count}（{profile.gpu_vram_gb} GB）" if profile.gpu_count else "")
        + f" / 磁盘剩余 {profile.disk_free_gb or '-'} GB"
    )
    lines.append("")
    lines.append("## 差距矩阵")
    lines.append("")
    lines.append("```")
    lines.append(render_matrix(report))
    lines.append("```")
    lines.append("")

    # 分组给出建议
    actionable = [i for i in report.items if i.verdict is not Verdict.OK]
    if actionable:
        lines.append("## 逐项说明")
        lines.append("")
        for item in actionable:
            req = item.requirement
            lines.append(f"### {req.id}　`{_VERDICT_MARK[item.verdict]}`")
            lines.append("")
            lines.append(f"- 论文要求：`{req.requirement_text()}`")
            lines.append(f"- 本机实际：`{item.local_value or '-'}`")
            if req.why:
                lines.append(f"- 为什么需要：{req.why}")
            if item.reason:
                lines.append(f"- 判定依据：{item.reason}")
            ev = req.evidence
            if ev:
                lines.append(f"- 证据：`{ev.describe()}`" + (f" — `{ev.excerpt}`" if ev.excerpt else ""))
            if item.remediation:
                lines.append("- 处理方式：")
                for r in item.remediation:
                    lines.append(f"  - `{r}`" if r.startswith("python ") else f"  - {r}")
            lines.append("")

    if spec.warnings:
        lines.append("## 需要注意")
        lines.append("")
        for w in spec.warnings:
            lines.append(f"- {w}")
        lines.append("")

    rejected = spec.rejected
    if rejected:
        lines.append("## 已丢弃的需求（无来源）")
        lines.append("")
        for r in rejected:
            lines.append(f"- `{r.id}`：{r.why or '缺少来源文件'}")
        lines.append("")

    lines.append("## 证据可信度说明")
    lines.append("")
    lines.append("- `L2 代码推断`：来自仓库文件，可确定，行号可核对")
    lines.append("- `L3 经验推测`：由型号或惯例推算，可能偏差，需人工确认")
    lines.append("- 本报告**不包含论文正文信息**，正文中的环境描述需要人工补充或后续接入 PDF 解析")
    return "\n".join(lines)
