"""文档 → 可定位文本块。

设计边界（先说清，免得后人误以为这是解析器）：

- 只做「把页面变成带定位的文本块」，**不产出任何环境需求语义**。
- 无版面模型、无表格结构还原、无公式识别、无 OCR、无 VLM。
- 定位精度是「页 + 行」，不是「页 + bbox 精确区域」。bbox 存下来只为排序与调试。

为什么不做版面还原：科研 agent 的主流做法是「逐页抽文本 + LLM + 引用回验」
（PaperQA2 / AI Scientist 审稿器都是这么干的），因为它们要的是「理解」，
不是「第 3 页第 2 个单元格」。本模块沿用同一路线，另加一件它们不做的事：
**明确记录哪些块没有被扫描到**，见 ``coverage``。

后端降级顺序：
    pdf   → pdfplumber（MIT）；缺失时抛 DocumentError，不静默返回空文本
    docx  → python-docx；缺失时退回 zipfile + 标准库 XML（无第三方依赖）
    text  → 标准库
"""

from __future__ import annotations

import re
import statistics
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

# 块类型。只区分「对证据归属有影响」的类型，不做细粒度版面分类。
KIND_HEADING = "heading"
KIND_PARAGRAPH = "paragraph"
KIND_TABLE_CAPTION = "table_caption"
KIND_TABLE = "table"
KIND_TABLE_NOTE = "table_note"
KIND_FOOTNOTE = "footnote"
KIND_EQUATION = "equation"
KIND_REFERENCE = "reference"

BLOCK_KINDS = (
    KIND_HEADING,
    KIND_PARAGRAPH,
    KIND_TABLE_CAPTION,
    KIND_TABLE,
    KIND_TABLE_NOTE,
    KIND_FOOTNOTE,
    KIND_EQUATION,
    KIND_REFERENCE,
)

# 承载环境约束的章节（小写匹配）。只用于「优先扫描」，不用于排除其它章节。
CONSTRAINT_SECTIONS = (
    "method",
    "methodology",
    "approach",
    "implementation",
    "experimental setup",
    "experiment setup",
    "experimental settings",
    "training details",
    "training setup",
    "appendix",
    "supplementary",
    "方法",
    "实验设置",
    "实现细节",
    "训练细节",
    "附录",
)

# 论文里「环境约束极可能只出现在这里」的块类型。
CONSTRAINT_KINDS = (
    KIND_TABLE_CAPTION,
    KIND_TABLE,
    KIND_TABLE_NOTE,
    KIND_FOOTNOTE,
)

_TABLE_CAPTION_RE = re.compile(r"^\s*(table|tab\.|表)\s*[0-9]+[.:：]?", re.IGNORECASE)
_FIGURE_CAPTION_RE = re.compile(r"^\s*(figure|fig\.|图)\s*[0-9]+[.:：]?", re.IGNORECASE)
_TABLE_NOTE_RE = re.compile(r"^\s*(note|注|注：|说明)[:：]", re.IGNORECASE)
_FOOTNOTE_MARK_RE = re.compile(r"^\s*[†‡§¶*]")
_HEADING_NUM_RE = re.compile(r"^\s*(\d+(\.\d+){0,3})\.?\s+\S")
_BACKREF_RE = re.compile(r"^\s*(\[1\]|\[\d+\]\s|[A-Z][a-z]+,\s+[A-Z]\.)")
_EQUATION_RE = re.compile(r"(\$\$|\\begin\{|\\frac|\\sum|\\mathbb)")


class DocumentError(RuntimeError):
    """文档无法解析（格式不支持、依赖缺失、文件损坏）。"""


@dataclass
class Block:
    """一个可定位的文本块。定位是「页 + 块序号」，不是坐标区域。"""

    page: int
    index: int
    kind: str
    text: str
    bbox: tuple[float, float, float, float] | None = None
    section: str = ""
    size: float = 0.0
    table_ref: int | None = None

    @property
    def in_constraint_region(self) -> bool:
        """是否属于「环境约束高密度区」。只影响扫描优先级，不做排除。"""
        if self.kind in CONSTRAINT_KINDS:
            return True
        sec = self.section.lower()
        return any(token in sec for token in CONSTRAINT_SECTIONS)

    def to_dict(self) -> dict[str, Any]:
        return {
            "page": self.page,
            "index": self.index,
            "kind": self.kind,
            "section": self.section,
            "text": self.text,
        }


@dataclass
class PaperDoc:
    """解析产物。只含文本与定位信息，不含任何判定结论。"""

    source: str
    source_kind: str
    blocks: list[Block] = field(default_factory=list)
    page_count: int = 0
    locator: str = "page"          # page / paragraph：Evidence.line 装的是哪一种
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return bool(self.blocks)

    def pages(self) -> dict[int, str]:
        """页号 → 该页全文。用于 excerpt 回验。"""
        out: dict[int, str] = {}
        for b in self.blocks:
            out[b.page] = out.get(b.page, "") + b.text + "\n"
        return out

    def constraint_blocks(self) -> list[Block]:
        return [b for b in self.blocks if b.in_constraint_region]

    def sections(self) -> list[str]:
        seen: list[str] = []
        for b in self.blocks:
            if b.kind == KIND_HEADING and b.section and b.section not in seen:
                seen.append(b.section)
        return seen

    def suspect_pages(self) -> list[int]:
        """字符密度过低的页——大概率是扫描件。只报警，不自动 OCR。"""
        return [int(w.split(":")[0]) for w in self.warnings if w.startswith("scan?")]

    def stats(self) -> dict[str, Any]:
        by_kind: dict[str, int] = {}
        for b in self.blocks:
            by_kind[b.kind] = by_kind.get(b.kind, 0) + 1
        return {
            "source": self.source,
            "source_kind": self.source_kind,
            "locator": self.locator,
            "page_count": self.page_count,
            "blocks": len(self.blocks),
            "by_kind": by_kind,
            "constraint_blocks": len(self.constraint_blocks()),
            "sections": len(self.sections()),
        }


# ---------------------------------------------------------------------------
# 块分类
# ---------------------------------------------------------------------------

def classify(text: str, *, size: float = 0.0, body_size: float = 0.0) -> str:
    """按文本形态判块类型。字号只用于脚注判定，其余全靠文本。"""
    stripped = text.strip()
    if not stripped:
        return KIND_PARAGRAPH
    if _TABLE_CAPTION_RE.match(stripped):
        return KIND_TABLE_CAPTION
    if _TABLE_NOTE_RE.match(stripped):
        return KIND_TABLE_NOTE
    if _FIGURE_CAPTION_RE.match(stripped):
        return KIND_PARAGRAPH
    if _FOOTNOTE_MARK_RE.match(stripped):
        return KIND_FOOTNOTE
    if body_size and size and size < body_size * 0.92 and len(stripped) < 300:
        return KIND_FOOTNOTE
    if _EQUATION_RE.search(stripped):
        return KIND_EQUATION
    if _BACKREF_RE.match(stripped) and len(stripped) < 400:
        return KIND_REFERENCE
    if is_heading(stripped):
        return KIND_HEADING
    return KIND_PARAGRAPH


def is_heading(text: str) -> bool:
    """判标题：短、无句末标点、命中编号或常见章节名。"""
    stripped = text.strip()
    if not stripped or len(stripped) > 90:
        return False
    if stripped[-1] in ".。;；,，":
        return False
    if _HEADING_NUM_RE.match(stripped):
        return True
    lowered = stripped.lower()
    return any(lowered.startswith(token) for token in (
        "abstract", "introduction", "related work", "method", "experiment",
        "result", "discussion", "conclusion", "reference", "appendix",
        "摘要", "引言", "相关工作", "方法", "实验", "结果", "结论", "参考文献", "附录",
    ))


# ---------------------------------------------------------------------------
# 后端：PDF
# ---------------------------------------------------------------------------

def _extract_pdf(path: Path) -> PaperDoc:
    try:
        import pdfplumber  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - 取决于环境
        raise DocumentError(
            "解析 PDF 需要 pdfplumber（MIT 许可）：pip install pdfplumber。"
            "未安装时 docx / txt / md 仍可解析。"
        ) from exc

    doc = PaperDoc(source=str(path), source_kind="pdf", locator="page")
    try:
        with pdfplumber.open(str(path)) as pdf:
            doc.page_count = len(pdf.pages)
            for pno, page in enumerate(pdf.pages, start=1):
                lines = _pdf_page_lines(page)
                text_len = sum(len(t) for t, _, _ in lines)
                if text_len < 200:
                    doc.warnings.append(f"scan?:{pno}:{text_len}")
                doc.blocks.extend(_lines_to_blocks(lines, pno, start_index=len(doc.blocks)))
    except DocumentError:
        raise
    except Exception as exc:  # 损坏 / 加密 / 不是 PDF
        raise DocumentError(f"打开 PDF 失败：{path.name}: {exc}") from exc

    if not doc.blocks:
        raise DocumentError(f"PDF 没有可用文本层（可能是扫描件）：{path.name}")
    return doc


def _pdf_page_lines(page: Any) -> list[tuple[str, tuple[float, float, float, float], float]]:
    """把一页切成行：按垂直坐标聚簇，行内按 x 排序。"""
    try:
        words = page.extract_words(extra_attrs=["size"], keep_blank_chars=False)
    except Exception:
        words = []
    if not words:
        raw = page.extract_text() or ""
        return [(ln, (0.0, 0.0, 0.0, 0.0), 0.0) for ln in raw.splitlines() if ln.strip()]

    rows: dict[int, list[dict[str, Any]]] = {}
    for w in words:
        key = int(round(float(w["top"]) / 3.0))
        rows.setdefault(key, []).append(w)

    lines: list[tuple[str, tuple[float, float, float, float], float]] = []
    for key in sorted(rows):
        items = sorted(rows[key], key=lambda w: float(w["x0"]))
        text = " ".join(str(w["text"]) for w in items).strip()
        if not text:
            continue
        bbox = (
            min(float(w["x0"]) for w in items),
            min(float(w["top"]) for w in items),
            max(float(w["x1"]) for w in items),
            max(float(w["bottom"]) for w in items),
        )
        size = statistics.median([float(w.get("size") or 0.0) for w in items])
        lines.append((text, bbox, size))
    return lines


def _lines_to_blocks(
    lines: list[tuple[str, tuple[float, float, float, float], float]],
    page: int,
    *,
    start_index: int = 0,
) -> list[Block]:
    sizes = [s for _, _, s in lines if s]
    body_size = statistics.median(sizes) if sizes else 0.0
    blocks: list[Block] = []
    section = ""
    for offset, (text, bbox, size) in enumerate(lines):
        kind = classify(text, size=size, body_size=body_size)
        if kind == KIND_HEADING:
            section = text.strip()
        blocks.append(Block(
            page=page,
            index=start_index + offset,
            kind=kind,
            text=text,
            bbox=bbox,
            section=section,
            size=size,
        ))
    _bind_table_refs(blocks)
    return blocks


def _bind_table_refs(blocks: list[Block]) -> None:
    """把表 caption 绑到紧随其后的表格块上。绑不上不算错，留 None。"""
    last_caption: Block | None = None
    for b in blocks:
        if b.kind == KIND_TABLE_CAPTION:
            last_caption = b
            continue
        if b.kind == KIND_TABLE and last_caption is not None:
            b.table_ref = last_caption.index
            last_caption.table_ref = b.index
            last_caption = None


# ---------------------------------------------------------------------------
# 后端：DOCX（结构直读，Word 本来就有结构，不需要模型）
# ---------------------------------------------------------------------------

_W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def _extract_docx(path: Path) -> PaperDoc:
    try:
        return _extract_docx_with_lib(path)
    except ImportError:
        return _extract_docx_stdlib(path)
    except DocumentError:
        raise
    except Exception:
        # python-docx 对不合规但能打开的文件很挑剔（缺 styles.xml、
        # Content_Types 不完整等）。标准库兜底更宽容，这里不降级就等于丢文件。
        return _extract_docx_stdlib(path)


def _extract_docx_with_lib(path: Path) -> PaperDoc:
    import io

    import docx  # noqa: PLC0415

    doc = PaperDoc(source=str(path), source_kind="docx", locator="paragraph")
    # 先读进内存再交给 python-docx：否则它在 Windows 上会一直占着文件句柄，
    # 调用方（包括临时目录清理）删不掉文件。
    document = docx.Document(io.BytesIO(path.read_bytes()))
    seq = 0
    section = ""
    for paragraph in document.paragraphs:
        text = (paragraph.text or "").strip()
        if not text:
            continue
        style = (getattr(paragraph.style, "name", "") or "").lower()
        kind = KIND_HEADING if style.startswith(("heading", "title", "标题")) else classify(text)
        if kind == KIND_HEADING:
            section = text
        doc.blocks.append(Block(page=0, index=seq, kind=kind, text=text, section=section))
        seq += 1
    for table in document.tables:
        rows = []
        for row in table.rows:
            cells = [(c.text or "").strip().replace("\n", " ") for c in row.cells]
            rows.append(" | ".join(cells))
        text = "\n".join(r for r in rows if r.strip())
        if text:
            doc.blocks.append(Block(page=0, index=seq, kind=KIND_TABLE, text=text, section=section))
            seq += 1
    doc.page_count = 1
    if not doc.blocks:
        raise DocumentError(f"docx 没有可用文本：{path.name}")
    doc.warnings.append("docx 无分页信息，Evidence.line 为段落序号")
    return doc


def _extract_docx_stdlib(path: Path) -> PaperDoc:
    """无 python-docx 时的兜底：docx 就是 zip + XML。"""
    try:
        with zipfile.ZipFile(str(path)) as zf:
            xml = zf.read("word/document.xml")
    except (KeyError, zipfile.BadZipFile) as exc:
        raise DocumentError(f"不是合法的 docx：{path.name}: {exc}") from exc

    root = ElementTree.fromstring(xml)
    body = root.find(f"{_W}body")
    if body is None:
        raise DocumentError(f"docx 结构异常（缺少 body）：{path.name}")

    doc = PaperDoc(source=str(path), source_kind="docx", locator="paragraph")
    doc.warnings.append("docx 无分页信息，Evidence.line 为段落序号")
    doc.warnings.append("使用标准库兜底解析（未安装 python-docx）")
    seq = 0
    section = ""
    for node in body:
        if node.tag == f"{_W}p":
            text = "".join(t.text or "" for t in node.iter(f"{_W}t")).strip()
            if not text:
                continue
            style = ""
            ppr = node.find(f"{_W}pPr")
            if ppr is not None:
                ps = ppr.find(f"{_W}pStyle")
                if ps is not None:
                    style = (ps.get(f"{_W}val") or "").lower()
            kind = KIND_HEADING if style.startswith(("heading", "title")) else classify(text)
            if kind == KIND_HEADING:
                section = text
            doc.blocks.append(Block(page=0, index=seq, kind=kind, text=text, section=section))
            seq += 1
        elif node.tag == f"{_W}tbl":
            rows: list[str] = []
            for tr in node.findall(f"{_W}tr"):
                cells = []
                for tc in tr.findall(f"{_W}tc"):
                    cells.append("".join(t.text or "" for t in tc.iter(f"{_W}t")).strip())
                if any(cells):
                    rows.append(" | ".join(cells))
            if rows:
                doc.blocks.append(
                    Block(page=0, index=seq, kind=KIND_TABLE, text="\n".join(rows), section=section)
                )
                seq += 1
    doc.page_count = 1
    if not doc.blocks:
        raise DocumentError(f"docx 没有可用文本：{path.name}")
    return doc


# ---------------------------------------------------------------------------
# 后端：纯文本 / Markdown
# ---------------------------------------------------------------------------

def _extract_text(path: Path, kind: str) -> PaperDoc:
    try:
        raw = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        raw = path.read_text(encoding="gbk", errors="replace")

    doc = PaperDoc(source=str(path), source_kind=kind, locator="page")
    section = ""
    for seq, line in enumerate(raw.splitlines()):
        text = line.strip()
        if not text:
            continue
        if kind == "markdown":
            heading = text.lstrip("#").strip() if text.startswith("#") else ""
            if heading:
                block_kind = KIND_HEADING
                section = heading
            else:
                block_kind = classify(text)
        else:
            block_kind = classify(text)
            if block_kind == KIND_HEADING:
                section = text
        doc.blocks.append(Block(page=1, index=seq, kind=block_kind, text=text, section=section))
    doc.page_count = 1
    if not doc.blocks:
        raise DocumentError(f"文件没有可用文本：{path.name}")
    return doc


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

_SUFFIX_KIND = {
    ".pdf": "pdf",
    ".docx": "docx",
    ".doc": "docx",      # .doc 二进制格式不支持，会明确报错
    ".md": "markdown",
    ".markdown": "markdown",
    ".txt": "text",
}


def extract(path: str | Path) -> PaperDoc:
    """把一份文档抽成可定位文本块。不抛异常之外的东西，失败抛 DocumentError。"""
    p = Path(path).expanduser()
    if not p.is_file():
        raise DocumentError(f"文件不存在：{p}")
    kind = _SUFFIX_KIND.get(p.suffix.lower())
    if kind is None:
        raise DocumentError(
            f"不支持的格式：{p.suffix or '(无扩展名)'}；"
            f"当前支持 {', '.join(sorted(_SUFFIX_KIND))}"
        )
    if kind == "pdf":
        return _extract_pdf(p)
    if kind == "docx":
        if p.suffix.lower() == ".doc":
            raise DocumentError("不支持 .doc 二进制格式，请先另存为 .docx")
        return _extract_docx(p)
    return _extract_text(p, kind)
