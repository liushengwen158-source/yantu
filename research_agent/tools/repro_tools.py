"""供 agent 调用的论文复现工具。

与其他工具一样：返回 JSON 可序列化的 dict，**不抛异常**，
出错时把原因写进返回值，让 agent 能读懂并转述给用户。

三条工具对应命令行 paper scan / diff / report，
但返回给模型的内容做了裁剪——结构化数据只保留可引用的字段，
完整长文本单独放在 text 字段里。
"""

from __future__ import annotations

from pathlib import Path

from ..repro import collect_profile, diff, render_matrix, render_report, scan_repo

# 单次返回给模型的文本上限，避免撑爆上下文
_MAX_TEXT = 7000


def _fail(message: str, hint: str = "") -> dict:
    return {"ok": False, "error": message, "hint": hint}


def _resolve(repo_path: str) -> tuple[Path | None, dict | None]:
    """校验路径，返回 (Path, None) 或 (None, 错误载荷)。"""
    if not repo_path or not repo_path.strip():
        return None, _fail("缺少仓库路径", "请提供论文配套代码在本机的目录，例如 /path/to/repo")
    text = repo_path.strip()
    if text.startswith(("http://", "https://", "git@", "git://")):
        return None, _fail(
            f"需要本地路径，不能直接扫远程仓库：{text}",
            "先用 git clone 拉到本地，再把本地目录传给本工具",
        )
    path = Path(text).expanduser()
    if not path.exists():
        return None, _fail(
            f"路径不存在：{path}",
            "确认路径是否正确；如果是远程仓库，先 git clone 到本地再传入目录",
        )
    if not path.is_dir():
        return None, _fail(f"不是目录：{path}", "请传入仓库根目录")
    return path, None


def repro_scan(repo_path: str) -> dict:
    """从论文配套代码里提取环境需求。

    Args:
        repo_path: 论文配套代码在本机的目录路径
    """
    path, err = _resolve(repo_path)
    if err:
        return err
    try:
        spec = scan_repo(path)
    except Exception as exc:  # noqa: BLE001
        return _fail(f"{type(exc).__name__}: {exc}")

    items = [
        {
            "id": r.id,
            "category": r.category.label,
            "requirement": r.requirement_text(),
            "why": r.why,
            "evidence": r.evidence.describe() if r.evidence else "无来源",
            "level": r.evidence.level.value if r.evidence else "",
        }
        for r in spec.accepted
    ]
    return {
        "ok": True,
        "repo": str(path),
        "scanned_files": spec.scanned_files,
        "count": len(items),
        "requirements": items,
        "dropped": [r.id for r in spec.rejected],
        "warnings": spec.warnings,
        "note": (
            "只解析仓库文件，不包含论文正文。每条需求都可按 evidence 里的文件名:行号核对，"
            "level 为 L3 的属于推算值，需要人工确认。"
        ),
    }


def repro_diff(repo_path: str) -> dict:
    """把论文环境需求与本机环境比对，给出四档可行性结论。

    Args:
        repo_path: 论文配套代码在本机的目录路径
    """
    path, err = _resolve(repo_path)
    if err:
        return err
    try:
        spec = scan_repo(path)
        profile = collect_profile(path)
        report = diff(spec, profile)
    except Exception as exc:  # noqa: BLE001
        return _fail(f"{type(exc).__name__}: {exc}")

    if not report.items:
        return {
            "ok": False,
            "error": "仓库里没有可识别的依赖声明，无法比对",
            "repo": str(path),
            "scanned_files": spec.scanned_files,
            "hint": "检查仓库是否包含 requirements.txt / environment.yml / Dockerfile",
        }

    matrix = render_matrix(report)
    items = [
        {
            "id": it.requirement.id,
            "category": it.requirement.category.label,
            "required": it.requirement.requirement_text(),
            "local": it.local_value or "-",
            "verdict": it.verdict.value,
            "verdict_label": it.verdict.label,
            "reason": it.reason,
            "evidence": it.requirement.evidence.describe() if it.requirement.evidence else "无来源",
            "remediation": it.remediation[:3],
        }
        for it in report.items
    ]
    return {
        "ok": True,
        "repo": str(path),
        "overall": report.overall.value,
        "overall_label": report.overall.label,
        "headline": report.headline(),
        "counts": report.counts,
        "items": items,
        "text": matrix[:_MAX_TEXT],
        "warnings": spec.warnings,
        "verdict_legend": {
            "block": "物理上不可能，需更换算力",
            "degrade": "改配置（batch/精度/LoRA）才能跑",
            "add": "安装或换版本即可",
            "ok": "满足",
            "unknown": "信息不足，不要当成满足",
            "conflict": "需求本身矛盾，需人工确认",
        },
    }


def repro_report(repo_path: str) -> dict:
    """生成完整的论文复现可行性报告（Markdown）。

    Args:
        repo_path: 论文配套代码在本机的目录路径
    """
    path, err = _resolve(repo_path)
    if err:
        return err
    try:
        spec = scan_repo(path)
        profile = collect_profile(path)
        report = diff(spec, profile)
        text = render_report(report)
    except Exception as exc:  # noqa: BLE001
        return _fail(f"{type(exc).__name__}: {exc}")

    return {
        "ok": True,
        "repo": str(path),
        "overall": report.overall.value,
        "overall_label": report.overall.label,
        "headline": report.headline(),
        "counts": report.counts,
        "text": text[:_MAX_TEXT],
        "truncated": len(text) > _MAX_TEXT,
    }


# 供 openjiuwen Tool 注册时统一取用
REPRO_TOOL_FUNCTIONS = {
    "repro_scan": repro_scan,
    "repro_diff": repro_diff,
    "repro_report": repro_report,
}
