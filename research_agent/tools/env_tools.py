"""供 agent 调用的工具函数。

每个函数都返回 JSON 可序列化的 dict，并且**不抛异常**——工具出错时把错误
放进返回值，让 agent 能读懂发生了什么，而不是整轮对话崩掉。

这几个函数是纯 Python 的，openjiuwen 的 Tool 封装只负责给它们套一层 schema。
"""

from __future__ import annotations

from ..envsetup import (
    build_guidance,
    build_plan,
    detect,
    execute,
    guidance_to_text,
    load_knowledge,
    render_actions,
    render_markdown,
)


def env_check(check_network: bool = True, check_torch: bool = True,
              as_markdown: bool = False) -> dict:
    """体检当前机器的 Python / Git / GPU 环境。

    Args:
        check_network: 是否探测代码托管连通性（多花几秒）
        check_torch: 是否探测 PyTorch（首次会比较慢）
        as_markdown: True 时额外返回人读的 markdown 报告
    """
    try:
        report = detect(check_network=check_network, check_torch=check_torch)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    kb = load_knowledge()
    data = report.to_dict()
    data["ok"] = True
    if as_markdown:
        data["markdown"] = render_markdown(report, kb)
    return data


def env_explain(check_id: str) -> dict:
    """解释某个检测项：为什么重要、怎么修、怎么验证。

    Args:
        check_id: 检测项 id，如 "python.version"、"torch.installed"
    """
    report = detect(check_network=False, check_torch=False)
    kb = load_knowledge()
    for result in report.results:
        if result.id != check_id:
            continue
        entry = kb.get(result.kb_id or "", {})
        return {
            "ok": True,
            "check_id": check_id,
            "name": result.name,
            "status": result.status.value,
            "observed": result.value,
            "message": result.message,
            "title": entry.get("title", ""),
            "why": entry.get("why", ""),
            "steps": entry.get("steps", []),
            "verify": entry.get("verify", ""),
            "doc": entry.get("doc", ""),
        }
    return {
        "ok": False,
        "error": f"没有名为 {check_id} 的检测项",
        "available": [r.id for r in report.results],
    }


def env_plan(max_risk: str = "config") -> dict:
    """生成修复计划（只生成命令，不执行）。

    Args:
        max_risk: safe / config / system，控制最多允许到哪一级风险
    """
    report = detect(check_network=False, check_torch=True)
    actions = build_plan(report, load_knowledge(), max_risk=max_risk)
    payload = [a.to_dict() for a in actions]
    return {
        "ok": True,
        "count": len(payload),
        "actions": payload,
        "preview": render_actions(payload),
    }


def env_fix(command: str, approve_system: bool = False) -> dict:
    """执行一条修复命令。

    Args:
        command: 要执行的命令，必须是 env_plan 里出现过的
        approve_system: 是否同意执行 system 级（安装/卸载软件）命令
    """
    report = detect(check_network=False, check_torch=False)
    actions = build_plan(report, load_knowledge(), max_risk="system")
    matched = next((a for a in actions if a.command.strip() == command.strip()), None)
    if matched is None:
        return {
            "ok": False,
            "error": "该命令不在当前修复计划中，拒绝执行",
            "allowed": [a.command for a in actions],
        }

    result = execute(matched, approve_system=approve_system)
    result["ok"] = bool(result.get("executed") and result.get("ok"))
    result["risk"] = matched.risk
    return result


def env_guidance() -> dict:
    """按严重程度返回当前需要处理的全部问题及处理步骤。"""
    report = detect(check_network=False, check_torch=True)
    guidance = build_guidance(report, load_knowledge())
    return {
        "ok": True,
        "count": len(guidance),
        "items": guidance,
        "text": guidance_to_text(guidance),
    }


# 供 openjiuwen Tool 注册时统一取用
TOOL_FUNCTIONS = {
    "env_check": env_check,
    "env_explain": env_explain,
    "env_plan": env_plan,
    "env_fix": env_fix,
    "env_guidance": env_guidance,
}
