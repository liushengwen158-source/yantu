"""从文本块里抽环境约束。

两条路，一个出口：

- **规则路**（默认兜底）：正则匹配主流写法。它只求「不漏掉最常见的那几种」，
  上限很低——这是有意为之，规则层的失败会暴露给覆盖报告，而不是被掩盖。
- **LLM 路**（有 API Key 时）：按 section 聚合上下文，要求模型连同**原文引文**
  一起返回；抽取结果一律经 ``verify.locate_quote`` 回验，回验不过就不算 grounded。

一条纪律：**本模块不产生 EvidenceLevel。** 它只产出「候选 + 引文 + 是否接地」，
定级是 ``repro.paper_adapter`` 的事。这样解析层与判定层的质量才能分别计量。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from .. import llm
from .extract import PaperDoc
from .verify import locate_quote, normalize

# 允许抽取的字段。必须与 repro/diff.py 已注册的判定分支一致，
# 抽了没注册的 id 只会静默落 UNKNOWN，属于「抽了不生效」。
ALLOWED_IDS = ("gpu.count", "gpu.vram", "cuda", "cuda.required", "python.version")

FIELD_SPEC = {
    "gpu.count": ("GPU 张数（单机/单节点的卡数）", '">="', ""),
    "gpu.vram": ("单卡显存", '">="', "GB"),
    "cuda": ("CUDA 版本（如 12.1）", '"=="', ""),
    "cuda.required": ("是否要求 CUDA（存在性）", '"exists"', ""),
    "python.version": ("Python 版本（如 3.10）", '"=="', ""),
}

GPU_MODELS = (
    "A100", "A800", "H100", "H800", "H20", "V100", "L40S", "L40", "A6000",
    "A10", "A30", "RTX4090", "RTX3090", "4090", "3090", "TPU",
)
_MODEL_ALT = "|".join(re.escape(m) for m in GPU_MODELS)

_NUM_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "six": 6, "eight": 8,
    "twelve": 12, "sixteen": 16, "thirty-two": 32, "sixty-four": 64,
}
_NUM_ALT = "|".join(_NUM_WORDS)

# 8×A100 / 8 x A100-80GB / 8 NVIDIA A100
_CNT_TIMES = re.compile(
    rf"(?P<n>\d{{1,4}})\s*[x×]\s*(?:NVIDIA\s+)?(?P<model>{_MODEL_ALT})s?\b", re.IGNORECASE)
# A100×8 / A100 x 8
_CNT_TIMES_REV = re.compile(
    rf"(?P<model>{_MODEL_ALT})\s*[x×]\s*(?P<n>\d{{1,4}})\b", re.IGNORECASE)
# 256 H100 GPUs / 8 NVIDIA A100 GPUs
_CNT_UNIT = re.compile(
    rf"(?P<n>\d{{1,4}})\s+(?:NVIDIA\s+|AMD\s+)?(?P<model>{_MODEL_ALT})s?\s*(?:GPUs?|graphics\s+cards?|cards?)\b",
    re.IGNORECASE)
# four NVIDIA A100 / eight A100-80GB
_CNT_WORD = re.compile(
    rf"(?P<w>{_NUM_ALT})\s+(?:NVIDIA\s+)?(?P<model>{_MODEL_ALT})s?\b", re.IGNORECASE)
# 512 V100s / 8 A100（数字直接跟型号，论文里极常见）
_CNT_BARE = re.compile(
    rf"(?P<n>\d{{1,4}})\s+(?:NVIDIA\s+|AMD\s+)?(?P<model>{_MODEL_ALT})s?\b", re.IGNORECASE)

# A100-80GB / A100 80GB / H100-SXM-80GB
_VRAM_ON_MODEL = re.compile(
    rf"(?P<model>{_MODEL_ALT})[-\s]*(?:SXM\d*\s*)?[-\s]?(?P<vram>\d{{2}})\s*GB\b", re.IGNORECASE)
# 80GB A100
_VRAM_BEFORE = re.compile(
    rf"(?P<vram>\d{{2,3}})\s*GB\s+(?:of\s+)?(?:GPU\s+)?(?:memory\s+)?(?:NVIDIA\s+)?(?P<model>{_MODEL_ALT})\b",
    re.IGNORECASE)
# 80 GB of GPU memory / 40GB VRAM
_VRAM_GENERIC = re.compile(
    r"(?P<vram>\d{2,3})\s*GB\s+(?:of\s+)?(?:GPU\s+)?(?:memory|VRAM)\b", re.IGNORECASE)

_CUDA_VERSION = re.compile(r"CUDA\s*(?:Toolkit\s*)?(?:version\s*)?v?(?P<maj>\d{1,2})\.(?P<min>\d{1,2})",
                           re.IGNORECASE)
_CUDA_MENTION = re.compile(r"\bCUDA\b")
_PY_VERSION = re.compile(r"\bPython\s*(?:version\s*)?v?(?P<maj>\d)\.(?P<min>\d{1,2})\b",
                         re.IGNORECASE)


@dataclass
class Candidate:
    """一条抽取候选。还不是需求，因为还没定级。"""

    id: str
    op: str
    value: str
    unit: str = ""
    quote: str = ""
    page: int = 0
    block_kind: str = ""
    section: str = ""
    method: str = "rule"        # rule / llm
    grounded: bool = False
    confidence: str = "medium"  # high / medium / low
    reason: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def key(self) -> tuple[str, str]:
        return (self.id, self.value)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "op": self.op,
            "value": self.value,
            "unit": self.unit,
            "page": self.page,
            "block_kind": self.block_kind,
            "section": self.section,
            "method": self.method,
            "grounded": self.grounded,
            "confidence": self.confidence,
            "quote": self.quote,
            "reason": self.reason,
        }


# ---------------------------------------------------------------------------
# 规则路
# ---------------------------------------------------------------------------

def _rule_candidates(block_text: str) -> list[tuple[str, str, str, str]]:
    """返回 [(id, op, value, unit)]。只在单块内部匹配，不跨块推断。"""
    found: list[tuple[str, str, str, str]] = []

    for m in _CNT_TIMES.finditer(block_text):
        found.append(("gpu.count", ">=", str(int(m.group("n"))), ""))
    for m in _CNT_TIMES_REV.finditer(block_text):
        found.append(("gpu.count", ">=", str(int(m.group("n"))), ""))
    for m in _CNT_UNIT.finditer(block_text):
        found.append(("gpu.count", ">=", str(int(m.group("n"))), ""))
    for m in _CNT_WORD.finditer(block_text):
        found.append(("gpu.count", ">=", str(_NUM_WORDS[m.group("w").lower()]), ""))
    for m in _CNT_BARE.finditer(block_text):
        found.append(("gpu.count", ">=", str(int(m.group("n"))), ""))

    for regex in (_VRAM_ON_MODEL, _VRAM_BEFORE, _VRAM_GENERIC):
        for m in regex.finditer(block_text):
            found.append(("gpu.vram", ">=", str(int(m.group("vram"))), "GB"))

    m = _CUDA_VERSION.search(block_text)
    if m:
        found.append(("cuda", "==", f"{int(m.group('maj'))}.{int(m.group('min'))}", ""))
    if _CUDA_MENTION.search(block_text):
        found.append(("cuda.required", "exists", "CUDA", ""))

    m = _PY_VERSION.search(block_text)
    if m:
        found.append(("python.version", "==", f"{m.group('maj')}.{int(m.group('min'))}", ""))
    return found


def extract_by_rules(doc: PaperDoc) -> list[Candidate]:
    """逐块跑规则。same id + 同值去重，保留最先出现的块。"""
    out: dict[tuple[str, str], Candidate] = {}
    for block in doc.blocks:
        for fid, op, value, unit in _rule_candidates(block.text):
            key = (fid, value)
            if key in out:
                continue
            out[key] = Candidate(
                id=fid, op=op, value=value, unit=unit,
                quote=block.text.strip()[:300],
                page=block.page, block_kind=block.kind, section=block.section,
                method="rule",
                grounded=True,          # 规则命中即块内文本，天然可回验
                confidence="medium",
            )
    return list(out.values())


# ---------------------------------------------------------------------------
# LLM 路
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """你在从学术论文中抽取「复现所需的计算环境约束」。

只允许输出下列 id：
- gpu.count       GPU 张数
- gpu.vram        单卡显存（GB）
- cuda            CUDA 版本，形如 12.1
- cuda.required   是否明确要求 CUDA（值固定为 CUDA）
- python.version  Python 版本，形如 3.10

硬性要求：
1. 每条必须给 quote，quote 必须是输入文本里**逐字出现**的一段原文（不得改写、不得翻译、不得拼接两处）。
2. 原文没写就不要猜；宁可不输出，也不要补一个"常见值"。
3. 同一字段有多个值（多套配置）时全部输出，不要合并。
4. 只输出 JSON 数组，不要解释、不要 Markdown 代码块。

输出格式：
[{"id": "gpu.count", "op": ">=", "value": "8", "unit": "", "quote": "we train on 8×A100 GPUs"}]
"""


def _chunks(doc: PaperDoc, budget: int = 6000) -> list[str]:
    """按 section 聚合、带页码标记分块。约束承载区优先。"""
    blocks = sorted(doc.blocks, key=lambda b: (not b.in_constraint_region, b.page, b.index))
    chunks: list[str] = []
    buf: list[str] = []
    size = 0
    for b in blocks:
        line = f"[p{b.page}] {b.text.strip()}"
        if size + len(line) > budget and buf:
            chunks.append("\n".join(buf))
            buf, size = [], 0
        buf.append(line)
        size += len(line) + 1
    if buf:
        chunks.append("\n".join(buf))
    return chunks


def _parse_json_array(text: str) -> list[dict[str, Any]]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text).strip()
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1:
        raise ValueError(f"模型没有返回 JSON 数组：{text[:200]}")
    data = json.loads(text[start:end + 1])
    if not isinstance(data, list):
        raise ValueError("模型返回的不是数组")
    return [d for d in data if isinstance(d, dict)]


def extract_by_llm(
    doc: PaperDoc,
    *,
    chat: Callable[..., str] | None = None,
    budget: int = 6000,
) -> tuple[list[Candidate], list[str]]:
    """LLM 抽取。返回 (候选, 警告)。无 Key 时不调用，直接返回空并给警告。"""
    warnings: list[str] = []
    caller = chat or llm.chat
    if chat is None and not llm.available():
        return [], ["未配置模型 API Key，跳过 LLM 抽取（仅用规则路）"]

    out: dict[tuple[str, str], Candidate] = {}
    for idx, chunk in enumerate(_chunks(doc, budget), start=1):
        try:
            reply = caller([
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": chunk},
            ])
        except Exception as exc:  # 网络/配额/解析，一律降级而不是崩
            warnings.append(f"LLM 第 {idx} 块失败：{exc}")
            continue
        try:
            items = _parse_json_array(reply)
        except (ValueError, json.JSONDecodeError) as exc:
            warnings.append(f"LLM 第 {idx} 块返回无法解析：{exc}")
            continue
        for item in items:
            fid = str(item.get("id", "")).strip()
            quote = str(item.get("quote", "")).strip()
            if fid not in ALLOWED_IDS:
                warnings.append(f"LLM 返回未注册 id，已丢弃：{fid}")
                continue
            if not quote:
                warnings.append(f"LLM 返回无引文的 {fid}，已丢弃")
                continue
            hit = locate_quote(quote, doc)
            value = str(item.get("value", "")).strip()
            # 第二道闸门：引文回验通过还不够，取值本身必须出现在那一段原文里。
            # 少了这一步，"8×A100"（引文成立）配 "80"GB（凭空补的）也能拿 L1——
            # 这正是「值对主体错」那类最危险的错。
            value_ok = bool(value) and _value_in_text(value, quote, doc, hit.block_index)
            grounded = hit.ok and value_ok
            reason = ""
            if not hit.ok:
                reason = f"引文回验失败：{hit.reason}"
            elif not value_ok:
                reason = f"引文不含该取值 {value}，不予立证"
            cand = Candidate(
                id=fid,
                op=str(item.get("op") or FIELD_SPEC[fid][1].strip('"')),
                value=value,
                unit=str(item.get("unit", "") or FIELD_SPEC[fid][2]),
                quote=quote,
                page=hit.page,
                block_kind=_block_kind(doc, hit.block_index),
                section=_block_section(doc, hit.block_index),
                method="llm",
                grounded=grounded,
                confidence="high" if grounded else "low",
                reason=reason,
            )
            if not cand.value:
                continue
            key = cand.key()
            prev = out.get(key)
            # 同一 (id, 值) 保留更可信的那条
            if prev is None or (cand.grounded and not prev.grounded):
                out[key] = cand
    return list(out.values()), warnings


def _value_in_text(value: str, quote: str, doc: PaperDoc, block_index: int | None) -> bool:
    """取值必须出现在引文（或引文所在块）里，否则不予立证。"""
    nv = normalize(value)
    if not nv:
        return False
    if nv in normalize(quote):
        return True
    for b in doc.blocks:
        if b.index == block_index:
            return nv in normalize(b.text)
    return False


def _block_kind(doc: PaperDoc, index: int | None) -> str:
    for b in doc.blocks:
        if b.index == index:
            return b.kind
    return ""


def _block_section(doc: PaperDoc, index: int | None) -> str:
    for b in doc.blocks:
        if b.index == index:
            return b.section
    return ""


# ---------------------------------------------------------------------------
# 出口
# ---------------------------------------------------------------------------

def extract_candidates(
    doc: PaperDoc,
    *,
    use_llm: bool | None = None,
    chat: Callable[..., str] | None = None,
) -> tuple[list[Candidate], list[str]]:
    """抽候选。LLM 优先，规则补齐；两边同 (id, 值) 时保留 LLM 那条。"""
    warnings: list[str] = []
    merged: dict[tuple[str, str], Candidate] = {}

    if use_llm is None:
        use_llm = llm.available()
    if use_llm:
        cands, warns = extract_by_llm(doc, chat=chat)
        warnings.extend(warns)
        for c in cands:
            merged[c.key()] = c

    for c in extract_by_rules(doc):
        merged.setdefault(c.key(), c)

    return list(merged.values()), warnings


def group_by_id(cands: Iterable[Candidate]) -> dict[str, list[Candidate]]:
    out: dict[str, list[Candidate]] = {}
    for c in cands:
        out.setdefault(c.id, []).append(c)
    return out


def prefer(cands: list[Candidate]) -> Candidate:
    """同字段多值时取最强的一条作为主值（多配置信息不丢，进 extra）。"""
    rank = {"high": 0, "medium": 1, "low": 2}
    return sorted(cands, key=lambda c: (not c.grounded, rank.get(c.confidence, 3)))[0]
