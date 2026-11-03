"""新手引导的工具函数（供 agent 调用）。

和 ``env_tools`` 同一套约定：返回 JSON 可序列化的 dict，**不抛异常**，
出错时把原因和 ``hint`` 写进返回值，模型能读懂并转述。

四条边界写在代码里：

- 评估只吃「用户说的话」和「用户显式授权的目录」，不主动探测任何路径
- 扫目录必须是本地目录，传 URL 直接拒绝并提示先下载
- 方向没定时不评估方向维，也不会默认按人工智能处理
- 路线里的每个阶段都带验收标准，缺验收的条目在生成时就被丢掉
"""
from __future__ import annotations

from typing import Any

from ..guide import (
    build_profile,
    build_roadmap,
    explain,
    list_domains,
    list_tracks,
    next_question,
    probe_plan,
    render_brief,
    render_card,
    render_miss,
    render_profile,
    render_questions,
    render_roadmap,
    render_roadmap_brief,
    resolve_domain,
    scan_materials,
)
from ..guide.spec import Domain

MARKDOWN_LIMIT = 8000


def _clip(text: str, limit: int = MARKDOWN_LIMIT) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n...（已截断，原长 {len(text)} 字符）"


def _scan(scan_dir: str):
    """扫描授权目录；出错时返回 (None, 错误 dict)。"""
    if not scan_dir:
        return None, None
    if scan_dir.startswith(("http://", "https://")):
        return None, {
            "ok": False,
            "error": "只能扫描本地目录",
            "hint": "先 git clone 或下载到本地，再把本地路径传进来",
        }
    try:
        return scan_materials(scan_dir), None
    except (FileNotFoundError, NotADirectoryError) as exc:
        return None, {
            "ok": False,
            "error": str(exc),
            "hint": "确认路径存在且是目录；不要传单个文件",
        }
    except Exception as exc:  # noqa: BLE001
        return None, {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def _domain_or_error(value: str):
    """解析方向参数；无法识别时返回 (Domain.UNKNOWN, 错误 dict)。"""
    if not (value or "").strip():
        return Domain.UNKNOWN, None
    try:
        resolved = resolve_domain(value)
    except ValueError as exc:
        return Domain.UNKNOWN, {
            "ok": False,
            "error": str(exc),
            "hint": "方向可选值见 domains 字段；不确定就不要传，留空表示还没定",
            "domains": list_tracks(),
        }
    return resolved, None


def guide_assess(
    text: str = "",
    scan_dir: str = "",
    domain: str = "",
    as_markdown: bool = True,
) -> dict[str, Any]:
    """从对话内容（可选加授权目录）推断能力画像。

    Args:
        text: 用户原话，可跨多轮拼接；空字符串表示只看材料
        scan_dir: 用户显式授权的本地目录，扫描 PDF / notebook / 笔记 / 源码
        domain: 用户已说明的方向 id 或名称；留空表示从 text 里识别
        as_markdown: 是否附带人类可读的表格
    """
    if not (text or "").strip() and not scan_dir:
        return {
            "ok": False,
            "error": "没有可评估的信息",
            "hint": "给一段用户原话（text），或让用户授权一个材料目录（scan_dir）",
        }

    resolved, error = _domain_or_error(domain)
    if error:
        return error

    scan, error = _scan(scan_dir)
    if error:
        return error

    try:
        profile = build_profile(text, scan=scan, domain=resolved if domain else None)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    data = profile.to_dict()
    data["ok"] = True
    data["brief"] = render_brief(profile)
    data["next_question"] = next_question(profile)
    if profile.domain is Domain.UNKNOWN:
        data["domains"] = list_tracks()
    if as_markdown:
        data["markdown"] = _clip(render_profile(profile))
    if scan is not None:
        data["materials"] = scan.to_dict()
    return data


def guide_roadmap(
    text: str = "",
    scan_dir: str = "",
    domain: str = "",
    weekly_hours: int = 0,
    as_markdown: bool = True,
) -> dict[str, Any]:
    """按方向与能力画像生成分阶段学习路线。

    Args:
        text: 用户原话，用来识别方向与推断水平
        scan_dir: 用户授权的材料目录
        domain: 方向 id 或名称，如 systems / 系统；留空表示还没定
        weekly_hours: 每周可投入小时数，0 表示不设预算
        as_markdown: 是否附人类可读的路线
    """
    if not (text or "").strip() and not scan_dir and not domain:
        return {
            "ok": False,
            "error": "没有任何可依据的信息",
            "hint": "给一段用户原话（text），或说明方向（domain），或授权一个目录（scan_dir）",
        }

    resolved, error = _domain_or_error(domain)
    if error:
        return error

    scan, error = _scan(scan_dir)
    if error:
        return error

    try:
        profile = build_profile(text, scan=scan, domain=resolved if domain else None)
        roadmap = build_roadmap(
            profile,
            hours_limit=weekly_hours * 8 if weekly_hours else None,
        )
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    data = roadmap.to_dict()
    data["ok"] = True
    data["brief"] = render_roadmap_brief(roadmap)
    if roadmap.domain is Domain.UNKNOWN:
        data["domains"] = list_tracks()
    if as_markdown:
        data["markdown"] = _clip(render_roadmap(roadmap))
    return data


def guide_questions(
    text: str = "",
    scan_dir: str = "",
    domain: str = "",
    limit: int = 3,
) -> dict[str, Any]:
    """按优先级列出接下来该问的问题（一次只问一个，这里给出队列）。

    Args:
        text: 用户原话
        scan_dir: 用户授权目录
        domain: 方向 id 或名称
        limit: 最多给出几个问题
    """
    resolved, error = _domain_or_error(domain)
    if error:
        return error
    scan, error = _scan(scan_dir)
    if error:
        return error
    try:
        profile = build_profile(text, scan=scan, domain=resolved if domain else None)
        plan = probe_plan(profile, limit=limit)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    return {
        "ok": True,
        "domain": profile.domain.value,
        "count": len(plan),
        "questions": plan,
        "text": render_questions(plan),
    }


def guide_explain(term: str, dimension: str = "") -> dict[str, Any]:
    """解释一个科研概念：一句话定义、类比、常见误解、前置与下一步。

    Args:
        term: 概念名，如「三次握手」「栈溢出」「反向传播」「venv」
        dimension: 可选，限定维度 coding/tools/math/research/systems/data/security/theory/ml/dl
    """
    if not (term or "").strip():
        return {"ok": False, "error": "没有指定概念", "hint": "传入 term，如 反向传播"}

    try:
        result = explain(term, dimension=dimension or None)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    if not result["found"]:
        return {
            "ok": False,
            "error": f"概念库里没有「{term}」",
            "candidates": result["candidates"],
            "text": render_miss(result),
            "hint": "不要凭空编定义；可以改用候选概念，或按用户上下文口头解释",
        }

    card = result["card"]
    return {
        "ok": True,
        "match": result["match"],
        "concept": card,
        "candidates": result["candidates"],
        "text": render_card(card),
    }


def guide_tracks() -> dict[str, Any]:
    """列出可选的研究方向：需要哪些能力、读哪些论文、基本功大概多少小时。"""
    tracks = list_tracks()
    return {"ok": True, "count": len(tracks), "tracks": tracks}


def guide_domains() -> dict[str, Any]:
    """列出研究方向及其要求（完整版，含论文与首个项目）。"""
    items = [spec.to_dict() for spec in list_domains()]
    return {"ok": True, "count": len(items), "domains": items}
