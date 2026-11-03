"""新手引导的工具函数（供 agent 调用）。

和 ``env_tools`` 同一套约定：返回 JSON 可序列化的 dict，**不抛异常**，
出错时把原因和 ``hint`` 写进返回值，模型能读懂并转述。

三条边界写在代码里：

- 评估只吃「用户说的话」和「用户显式授权的目录」，不主动探测任何路径
- 扫目录必须是本地目录，传 URL 直接拒绝并提示先下载
- 路线里的每个阶段都带验收标准，缺验收的条目在生成时就被丢掉
"""
from __future__ import annotations

from typing import Any

from ..guide import (
    build_profile,
    build_roadmap,
    explain,
    list_tracks,
    next_question,
    render_brief,
    render_card,
    render_miss,
    render_profile,
    render_roadmap,
    render_roadmap_brief,
    scan_materials,
)

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


def guide_assess(
    text: str = "",
    scan_dir: str = "",
    as_markdown: bool = True,
) -> dict[str, Any]:
    """从对话内容（可选加授权目录）推断六维能力画像。

    Args:
        text: 用户原话，可跨多轮拼接；空字符串表示只看材料
        scan_dir: 用户显式授权的本地目录，扫描 PDF / notebook / 笔记
        as_markdown: 是否附带人类可读的表格
    """
    if not (text or "").strip() and not scan_dir:
        return {
            "ok": False,
            "error": "没有可评估的信息",
            "hint": "给一段用户原话（text），或让用户授权一个材料目录（scan_dir）",
        }

    scan, error = _scan(scan_dir)
    if error:
        return error

    try:
        profile = build_profile(text, scan=scan)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    data = profile.to_dict()
    data["ok"] = True
    data["brief"] = render_brief(profile)
    question = next_question(profile)
    data["next_question"] = question
    if as_markdown:
        data["markdown"] = _clip(render_profile(profile))
    if scan is not None:
        data["materials"] = scan.to_dict()
    return data


def guide_roadmap(
    text: str = "",
    scan_dir: str = "",
    track: str = "general",
    weekly_hours: int = 0,
    as_markdown: bool = True,
) -> dict[str, Any]:
    """按能力画像生成分阶段学习路线。

    Args:
        text: 用户原话，用来推断水平
        scan_dir: 用户授权的材料目录
        track: 方向 id（cv/nlp/system/data/theory/ai4science），留空表示还没定
        weekly_hours: 每周可投入小时数，0 表示用默认 10 小时
        as_markdown: 是否附人类可读的路线
    """
    if not (text or "").strip() and not scan_dir:
        return {
            "ok": False,
            "error": "没有任何可依据的信息",
            "hint": "给一段用户原话（text），或让用户授权一个材料目录（scan_dir）",
        }

    scan, error = _scan(scan_dir)
    if error:
        return error

    try:
        profile = build_profile(text, scan=scan)
        roadmap = build_roadmap(
            profile, track=track or "general", weekly_hours=weekly_hours or None
        )
    except ValueError as exc:
        return {
            "ok": False,
            "error": str(exc),
            "hint": "可选方向见 tracks 字段",
            "tracks": list_tracks(),
        }
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    data = roadmap.to_dict()
    data["ok"] = True
    data["brief"] = render_roadmap_brief(roadmap)
    if as_markdown:
        data["markdown"] = _clip(render_roadmap(roadmap))
    return data


def guide_explain(term: str, dimension: str = "") -> dict[str, Any]:
    """解释一个科研入门概念：一句话定义、类比、常见误解、前置与下一步。

    Args:
        term: 概念名，如「反向传播」「过拟合」「venv」
        dimension: 可选，限定维度 python/shell/math/ml/dl/research
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
    """列出可选的科研方向。"""
    tracks = list_tracks()
    return {"ok": True, "count": len(tracks), "tracks": tracks}
