"""建议生成：把检测结果 + 知识库拼成新手看得懂的指引。

这里刻意不依赖大模型：规则能给出准确、可复现的答案，
大模型只在需要「个性化解释 / 学习路径」时作为增强（见 agent.py）。
"""

from __future__ import annotations

import os

from .models import CheckResult, EnvReport, Status

_STATUS_ICON = {
    Status.OK: "[OK]",
    Status.INFO: "[建议]",
    Status.WARN: "[注意]",
    Status.BLOCKER: "[阻塞]",
    Status.UNKNOWN: "[未知]",
}

_DOMAIN_TITLE = {
    "python": "Python 与包管理",
    "git": "Git 与代码托管",
    "gpu": "GPU 与深度学习",
}


def _platform_ok(step_platform: str | None) -> bool:
    """当前平台是否适用该步骤；不适用的直接不展示，避免误导。"""
    if not step_platform or step_platform == "any":
        return True
    if step_platform == "win":
        return os.name == "nt"
    if step_platform == "linux":
        return os.name != "nt"
    return True


def _entry_for(result: CheckResult, kb: dict) -> dict | None:
    return kb.get(result.kb_id) if result.kb_id else None


def build_guidance(report: EnvReport, kb: dict) -> list[dict]:
    """按「先阻塞、再注意、后建议」的顺序生成指引。"""
    out: list[dict] = []
    for result in report.results:
        entry = _entry_for(result, kb)
        if result.status is Status.OK or entry is None:
            continue
        steps = [s for s in entry.get("steps", []) if _platform_ok(s.get("platform"))]
        if not steps:
            continue
        out.append({
            "check_id": result.id,
            "domain": result.domain,
            "kb_id": result.kb_id,
            "title": entry.get("title", result.name),
            "severity": result.status.value,
            "why": entry.get("why", ""),
            "observed": result.value,
            "steps": steps,
            "verify": entry.get("verify", ""),
            "doc": entry.get("doc", ""),
        })
    out.sort(key=lambda g: (Status(g["severity"]).order, g["domain"]))
    return out


def guidance_to_text(guidance: list[dict], with_commands: bool = True) -> str:
    """给 LLM / 聊天用的纯文本块。"""
    if not guidance:
        return "环境未发现需要处理的问题。"
    blocks = []
    for i, g in enumerate(guidance, 1):
        lines = [f"{i}. [{g['severity']}] {g['title']}（当前：{g['observed'] or '未知'}）"]
        if g["why"]:
            lines.append(f"   原因：{g['why']}")
        for s in g["steps"]:
            text = s.get("text", "")
            cmd = s.get("command", "")
            if cmd and with_commands:
                lines.append(f"   - {text}")
                lines.append(f"     `{cmd}`")
            elif text:
                lines.append(f"   - {text}")
        if g["verify"]:
            lines.append(f"   验证：`{g['verify']}`")
        blocks.append("\n".join(lines))
    return "\n".join(blocks)


def render_markdown(report: EnvReport, kb: dict) -> str:
    """渲染完整体检报告。"""
    s = report.summary()
    lines: list[str] = []
    lines.append("# 环境体检报告")
    lines.append("")
    lines.append(f"- 系统：{report.platform}")
    lines.append(f"- 解释器：`{report.python_executable or '未找到'}`（{report.python_version or '-'}）")
    lines.append(f"- 检测时间：{report.generated_at}")
    lines.append(
        f"- 就绪度：**{s['score']}/100**　阻塞 {s['blockers']} · 注意 {s['warnings']} · "
        f"建议 {s['suggestions']} · 正常 {s['ok']}/{s['total']}"
    )
    lines.append("")

    for domain, results in report.by_domain().items():
        lines.append(f"## {_DOMAIN_TITLE.get(domain, domain)}")
        lines.append("")
        for r in results:
            icon = _STATUS_ICON[r.status]
            value = f"`{r.value}`" if r.value else ""
            msg = f" — {r.message}" if r.message else ""
            lines.append(f"- {icon} **{r.name}** {value}{msg}")
        lines.append("")

    guidance = build_guidance(report, kb)
    if guidance:
        lines.append("## 建议处理顺序")
        lines.append("")
        for i, g in enumerate(guidance, 1):
            lines.append(f"### {i}. {g['title']}　`{g['severity']}`")
            if g["why"]:
                lines.append("")
                lines.append(g["why"])
            lines.append("")
            for step in g["steps"]:
                text = step.get("text", "")
                cmd = step.get("command")
                if cmd:
                    lines.append(f"- {text}")
                    lines.append("")
                    lines.append("  ```bash")
                    lines.append(f"  {cmd}")
                    lines.append("  ```")
                elif text:
                    lines.append(f"- {text}")
            if g["verify"]:
                lines.append("")
                lines.append(f"验证：`{g['verify']}`")
            lines.append("")

    if s["blockers"] == 0 and s["warnings"] == 0:
        lines.append("---")
        lines.append("")
        lines.append("**基础环境已就绪**，可以进入下一步：写一个 Hello World 跑通，再开始读论文复现。")

    return "\n".join(lines)


def render_actions(actions: list[dict]) -> str:
    """渲染待执行动作清单（dry-run 用）。"""
    if not actions:
        return "没有需要执行的命令。"
    lines = ["待执行命令（按风险从低到高）：", ""]
    for i, a in enumerate(actions, 1):
        lines.append(f"{i}. [{a['risk']}] {a['reason'] or a['title']}")
        lines.append(f"   `{a['command']}`")
    lines.append("")
    lines.append("提示：config 级会改用户配置，system 级会安装/卸载软件。")
    return "\n".join(lines)
