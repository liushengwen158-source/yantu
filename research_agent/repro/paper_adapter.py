"""把文档抽取结果投影成仓库既有的需求契约。

这是 ``document`` 与 ``repro`` 之间**唯一**的接缝，刻意做得很薄：

    document（只认识页/块/文本）  →  paper_adapter（这里）  →  repro（只认识需求/证据）

三条定级规则（都在这一层，不在解析层）：

1. **引文回验通过才给 L1**。回验失败一律 L3，并在 ``extra`` 里写明原因——
   L3 是「经验推测」，用户看到它会自己核一遍，这正是我们要的。
2. **多套配置不合并**。同字段多值时主值取最可信的一条，全部值进 ``extra["configs"]``。
   把多配置压成单值是语义缺失，而且压过之后不可逆。
3. **未注册的 id 直接丢弃并警告**。抽了没注册的字段只会静默落 UNKNOWN，
   属于「抽了不生效」，比不抽更糟。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..document import (
    ALLOWED_IDS,
    STATE_ABSENT,
    STATE_FAILED,
    STATE_HIT,
    FieldOutcome,
    PaperDoc,
    absent_or_failed,
    build_coverage,
    extract,
    extract_candidates,
    group_by_id,
)
from ..document.llm_extract import Candidate, prefer
from .spec import Category, EnvRequirement, Evidence, EvidenceLevel

# id → 分类。与 document.ALLOWED_IDS 一一对应。
_CATEGORY = {
    "gpu.count": Category.HARDWARE,
    "gpu.vram": Category.HARDWARE,
    "cuda": Category.CUDA,
    "cuda.required": Category.SYSTEM,
    "python.version": Category.PYTHON,
}

_WHY = {
    "gpu.count": "论文声明所需的 GPU 张数",
    "gpu.vram": "论文声明所需的单卡显存",
    "cuda": "论文声明的 CUDA 版本",
    "cuda.required": "论文要求可访问 CUDA 设备",
    "python.version": "论文声明的 Python 版本",
}


def _evidence(c: Candidate, doc: PaperDoc) -> Evidence:
    """接地的给 L1，没接地的给 L3。line 装页码（docx 装段落序号）。"""
    level = EvidenceLevel.L1 if c.grounded else EvidenceLevel.L3
    return Evidence(
        level=level,
        source_file=Path(doc.source).name,
        line=c.page,
        excerpt=(c.quote or "")[:200],
    )


def to_requirements(
    doc: PaperDoc,
    candidates: list[Candidate],
    *,
    warnings: list[str] | None = None,
) -> list[EnvRequirement]:
    """候选 → EnvRequirement 列表。按 id 聚合，多配置进 extra。"""
    warns = warnings if warnings is not None else []
    out: list[EnvRequirement] = []
    for fid, cands in group_by_id(candidates).items():
        if fid not in _CATEGORY:
            warns.append(f"未注册字段 {fid}，已丢弃（抽了不生效）")
            continue
        main = prefer(cands)
        extra: dict[str, Any] = {
            "extraction_method": main.method,
            "grounded": "verbatim" if main.grounded else "inferred",
            "confidence": main.confidence,
            "block_type": main.block_kind or "unknown",
            "source_kind": doc.source_kind,
            "locator_kind": doc.locator,
        }
        if main.section:
            extra["section"] = main.section
        if main.reason:
            extra["reason_code"] = main.reason

        unique_values = {c.value for c in cands}
        if len(unique_values) > 1:
            extra["configs"] = [
                {"value": c.value, "unit": c.unit, "page": c.page,
                 "block_type": c.block_kind, "grounded": c.grounded, "quote": c.quote}
                for c in cands
            ]
            extra["ambiguity"] = f"论文给出 {len(unique_values)} 个不同取值"

        out.append(EnvRequirement(
            id=fid,
            category=_CATEGORY[fid],
            op=main.op,
            value=main.value,
            unit=main.unit,
            hard=True,
            why=_WHY.get(fid, ""),
            evidence=_evidence(main, doc),
            extra=extra,
        ))

        if not main.grounded:
            warns.append(f"{fid} 的引文未能回验（{main.reason}），已降级为 L3")
    return out


def _outcomes(candidates: list[Candidate], coverage: Any) -> list[FieldOutcome]:
    """逐字段四态。**每个允许的 id 都必须有一条**，没抽到也要显式表态。"""
    by_id = group_by_id(candidates)
    out: list[FieldOutcome] = []
    for fid in ALLOWED_IDS:
        cands = by_id.get(fid)
        if not cands:
            out.append(absent_or_failed(coverage, fid))
            continue
        main = prefer(cands)
        out.append(FieldOutcome(
            id=fid,
            state=STATE_HIT,
            page=main.page,
            quote=main.quote,
            reason_code="" if main.grounded else "UNGROUNDED",
        ))
    return out


def paper_requirements(
    path: str | Path,
    *,
    use_llm: bool | None = None,
) -> tuple[list[EnvRequirement], dict[str, Any], list[str]]:
    """一条命令跑完：文档 → 需求 + 覆盖报告 + 警告。

    失败不抛异常，返回空需求与错误说明——与 ``tools/`` 里的工具约定一致。
    """
    warnings: list[str] = []
    try:
        doc = extract(path)
    except Exception as exc:
        return [], {"ok": False, "error": str(exc)}, [str(exc)]

    candidates, cand_warns = extract_candidates(doc, use_llm=use_llm)
    warnings.extend(cand_warns)

    reqs = to_requirements(doc, candidates, warnings=warnings)
    # 先建一次（只有页/块的扫描情况），用来判定「没抽到」到底是 ABSENT 还是 FAILED
    preview = build_coverage(doc)
    coverage = build_coverage(doc, _outcomes(candidates, preview))

    for r in reqs:
        if r.extra.get("configs"):
            warnings.append(f"{r.id} 存在多套配置，已全部写入 extra.configs，未合并")
    for o in coverage.outcomes:
        if o.state in (STATE_ABSENT, STATE_FAILED):
            warnings.append(f"{o.id} 未抽到（{o.state}），不代表论文没写——见覆盖报告")

    report = {
        "ok": True,
        "doc": doc.stats(),
        "coverage": coverage.to_dict(),
        "candidates": [c.to_dict() for c in candidates],
        "requirements": [r.to_dict() for r in reqs],
        "warnings": warnings,
    }
    return reqs, report, warnings
