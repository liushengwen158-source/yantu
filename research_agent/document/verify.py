"""excerpt 回验与覆盖报告。

两件事，都很小，但是整个方案里唯一「别的 agent 不做」的部分：

1. **回验**：模型声称引用了某段原文 → 去那一页的文本里找得到吗？
   找不到就不许升 L1。这是把「论文明确」从自述变成可机械验证的唯一办法。
2. **覆盖**：哪些页/块被扫过、哪些没被扫过。
   没有它，「这篇论文没写 batch size」与「我们没读到那一页」在输出上完全同形。

状态取值（四态，别混用）：
    HIT            命中且可回验
    ABSENT         扫过了，确实没有
    FAILED         没扫成（异常 / 未执行）
    NOT_APPLICABLE 与它无关，不占召回率分母
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Iterable

from .extract import Block, PaperDoc

STATE_HIT = "HIT"
STATE_ABSENT = "ABSENT"
STATE_FAILED = "FAILED"
STATE_NA = "NOT_APPLICABLE"

STATES = (STATE_HIT, STATE_ABSENT, STATE_FAILED, STATE_NA)

# 回验的最短长度：太短的串（"8"、"A100"）匹配上没有意义。
MIN_QUOTE_CHARS = 6

_WS_RE = re.compile(r"\s+")
_TRANS = str.maketrans({
    "×": "x", "✕": "x", "✖": "x", "⨯": "x",
    "–": "-", "—": "-", "−": "-", "‐": "-", "‑": "-",
    "“": '"', "”": '"', "‘": "'", "’": "'",
})


def normalize(text: str) -> str:
    """宽松归一：全角转半角、去空白、统一乘号与引号、小写。

    去空白是为了应付 PDF 行内换行把 ``8×A100-80GB`` 拆成两行的情况；
    代价是会放过「空格位置不同」的差异，但那种差异对证据核对无意义。
    """
    if not text:
        return ""
    s = unicodedata.normalize("NFKC", text)
    s = s.translate(_TRANS)
    s = _WS_RE.sub("", s)
    return s.lower()


@dataclass
class VerifyResult:
    ok: bool
    method: str          # exact / partial / miss / too_short
    reason: str = ""
    page: int = 0
    block_index: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "method": self.method,
            "reason": self.reason,
            "page": self.page,
            "block_index": self.block_index,
        }


def verify_quote(quote: str, page_text: str, *, min_chars: int = MIN_QUOTE_CHARS) -> VerifyResult:
    """判断 quote 是否能在 page_text 里定位到。只比对，不做模糊「意思相近」判定。"""
    nq = normalize(quote)
    if len(nq) < min_chars:
        return VerifyResult(False, "too_short", f"引文过短（{len(nq)} 字符 < {min_chars}）")
    npage = normalize(page_text)
    if not npage:
        return VerifyResult(False, "miss", "该页没有文本")
    if nq in npage:
        return VerifyResult(True, "exact")

    # 部分命中：分成若干 token 看覆盖率，用于诊断（不据此升级证据等级）
    tokens = [t for t in re.split(r"[^0-9a-z\u4e00-\u9fff.]+", nq) if len(t) >= 3]
    if tokens:
        hit = sum(1 for t in tokens if t in npage)
        ratio = hit / len(tokens)
        if ratio >= 0.8:
            return VerifyResult(False, "partial", f"部分命中 {hit}/{len(tokens)}，不足以立证")
    return VerifyResult(False, "miss", "该页找不到该片段")


def locate_quote(quote: str, doc: PaperDoc, *, pages: Iterable[int] | None = None) -> VerifyResult:
    """在文档里找 quote。找到返回页号与块序号，找不到返回 miss。"""
    allowed = set(pages) if pages is not None else None
    page_text = doc.pages()
    for page in sorted(page_text):
        if allowed is not None and page not in allowed:
            continue
        result = verify_quote(quote, page_text[page])
        if result.ok:
            result.page = page
            result.block_index = _find_block(quote, doc, page)
            return result
    return VerifyResult(False, "miss", "全篇找不到该片段")


def _find_block(quote: str, doc: PaperDoc, page: int) -> int | None:
    nq = normalize(quote)
    for b in doc.blocks:
        if b.page != page:
            continue
        if nq in normalize(b.text):
            return b.index
    return None


@dataclass
class FieldOutcome:
    """一个字段的判定结果。空列表不得表达任何状态——所以要显式带 state。"""

    id: str
    state: str
    page: int = 0
    quote: str = ""
    reason_code: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "state": self.state,
            "page": self.page,
            "quote": self.quote,
            "reason_code": self.reason_code,
            "extra": self.extra,
        }


@dataclass
class Coverage:
    """覆盖报告。回答「我们读到了什么、哪些没读到」。"""

    total_pages: int = 0
    scanned_pages: list[int] = field(default_factory=list)
    blocks_total: int = 0
    blocks_scanned: int = 0
    blocks_in_constraint_region: int = 0
    by_kind: dict[str, int] = field(default_factory=dict)
    outcomes: list[FieldOutcome] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def fully_scanned(self) -> bool:
        """整篇都扫过了，才允许输出 ABSENT。"""
        return self.total_pages > 0 and len(self.scanned_pages) >= self.total_pages

    def verdict_of(self, field_id: str) -> str:
        for o in self.outcomes:
            if o.id == field_id:
                return o.state
        return STATE_FAILED

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_pages": self.total_pages,
            "scanned_pages": len(self.scanned_pages),
            "fully_scanned": self.fully_scanned,
            "blocks_total": self.blocks_total,
            "blocks_scanned": self.blocks_scanned,
            "blocks_in_constraint_region": self.blocks_in_constraint_region,
            "by_kind": self.by_kind,
            "outcomes": [o.to_dict() for o in self.outcomes],
            "warnings": self.warnings,
        }

    def render(self) -> str:
        lines = [
            f"覆盖：{len(self.scanned_pages)}/{self.total_pages} 页，"
            f"{self.blocks_scanned}/{self.blocks_total} 块"
            f"（其中约束承载区 {self.blocks_in_constraint_region} 块）",
        ]
        by_kind = ", ".join(f"{k}={v}" for k, v in sorted(self.by_kind.items()))
        if by_kind:
            lines.append(f"块类型：{by_kind}")
        for o in self.outcomes:
            loc = f" p{o.page}" if o.page else ""
            reason = f" [{o.reason_code}]" if o.reason_code else ""
            lines.append(f"  - {o.id}: {o.state}{loc}{reason}")
        for w in self.warnings:
            lines.append(f"  ! {w}")
        return "\n".join(lines)


def build_coverage(
    doc: PaperDoc,
    outcomes: Iterable[FieldOutcome] = (),
    *,
    pages_scanned: Iterable[int] | None = None,
    blocks_scanned: int | None = None,
) -> Coverage:
    """按文档与字段结果组装覆盖报告。

    默认认为整篇都被扫过（本模块的抽取是全文逐块跑的）；
    调用方若做了分页/预算截断，必须显式传 ``pages_scanned`` 与 ``blocks_scanned``。
    """
    by_kind: dict[str, int] = {}
    for b in doc.blocks:
        by_kind[b.kind] = by_kind.get(b.kind, 0) + 1

    all_pages = sorted({b.page for b in doc.blocks})
    scanned = sorted(set(pages_scanned)) if pages_scanned is not None else all_pages

    cov = Coverage(
        total_pages=doc.page_count or len(all_pages),
        scanned_pages=scanned,
        blocks_total=len(doc.blocks),
        blocks_scanned=blocks_scanned if blocks_scanned is not None else len(doc.blocks),
        blocks_in_constraint_region=len(doc.constraint_blocks()),
        by_kind=by_kind,
        outcomes=list(outcomes),
        warnings=list(doc.warnings),
    )

    # 没扫完就不许出现 ABSENT——否则「没读到」会冒充「没写」
    if not cov.fully_scanned:
        for o in cov.outcomes:
            if o.state == STATE_ABSENT:
                o.state = STATE_FAILED
                o.reason_code = o.reason_code or "COVERAGE_INCOMPLETE"
        cov.warnings.append("覆盖不完整：ABSENT 已降级为 FAILED，不得据此判断「论文没写」")
    return cov


def absent_or_failed(coverage: Coverage, field_id: str) -> FieldOutcome:
    """抽取层没产出某字段时的默认结果：扫全了才算 ABSENT，否则 FAILED。"""
    state = STATE_ABSENT if coverage.fully_scanned else STATE_FAILED
    code = "" if coverage.fully_scanned else "COVERAGE_INCOMPLETE"
    return FieldOutcome(id=field_id, state=state, reason_code=code)


def suspect_blocks(doc: PaperDoc, keywords: Iterable[str] = ("gpu", "a100", "h100", "v100",
                                                              "cuda", "batch", "显存")) -> list[Block]:
    """文本里有硬件关键词、但块类型是纯段落的块。

    用途：这类块可能是「被压平成文本流的表格」，也可能是正文散文——分不清，
    所以只留痕、不当结论、不进任何门限。
    """
    out: list[Block] = []
    for b in doc.blocks:
        if b.kind != "paragraph":
            continue
        low = b.text.lower()
        if any(k in low for k in keywords):
            out.append(b)
    return out
