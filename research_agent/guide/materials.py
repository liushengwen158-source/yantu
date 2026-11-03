"""扫描用户授权的目录，把「你手上有什么材料」变成文件证据。

两条边界，写死在实现里：

1. **只读、不联网、不猜内容**：只数文件、读文本，不解析 PDF 正文
   （正文解析由 ``document-processing`` 分支负责，未合并前这里不假装能读）。
2. **文件只能证明「材料在」**，不能证明「读过、会做」。所以这里的证据一律
   ``strength="weak"``、``source=FILE``，单独一条不给维度定高档位，
   「自述熟练但目录里只有入门教程」这种情况会作为对撞提示摆出来，而不是下结论。

命名上把论文、notebook、笔记分开统计，因为它们的可信度不一样：
- 论文 PDF 在目录里 → 可能只是下载了；
- 论文旁边有同名笔记 → 大概率真读过；
- notebook 里有训练循环 → 至少跑过代码。
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .spec import Dimension, Level, Signal, Source

PAPER_EXT = {".pdf"}
NOTEBOOK_EXT = {".ipynb"}
NOTES_EXT = {".md", ".txt", ".org", ".rst"}

SKIP_DIRS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    "node_modules",
    ".idea",
    ".vscode",
    "dist",
    "build",
    ".mypy_cache",
    ".pytest_cache",
    ".ipynb_checkpoints",
}

MAX_FILES = 3000
MAX_READ_BYTES = 2_000_000

IMPORT_RE = re.compile(
    r"^\s*(?:import|from)\s+(torch|tensorflow|jax|sklearn|numpy|pandas)\b", re.MULTILINE
)
TRAIN_HINTS = ("backward(", "optimizer", "dataloader", ".fit(", "epoch", "loss")

# 笔记里出现这些词，说明这一维至少被碰过。只作弱证据。
NOTE_KEYWORDS: dict[Dimension, tuple[str, ...]] = {
    Dimension.MATH: ("梯度", "线性代数", "矩阵", "概率", "特征值", "导数", "偏导", "证明", "复杂度"),
    Dimension.RESEARCH: ("论文", "复现", "消融", "实验记录", "baseline", "arxiv", "引用"),
    # 系统、数据、安全、机器学习都算「写代码」的不同侧面，不再单设维度
    Dimension.CODING: (
        "进程", "线程", "死锁", "内存分配", "索引", "事务", "漏洞", "栈溢出", "汇编",
        "过拟合", "交叉验证", "正则化", "训练集", "验证集", "反向传播", "卷积", "注意力",
        "transformer", "学习率", "epoch", "损失函数",
    ),
}

# 目录里有这些源码，说明干活的语言不是 Python；按弱证据给编程维。
CODE_EXT_SYSTEMS = {".c", ".cc", ".cpp", ".h", ".hpp", ".rs", ".go", ".s", ".asm"}
CODE_EXT_DATA = {".sql"}


@dataclass
class Material:
    """一份材料。"""

    path: str            # 相对扫描根的路径
    kind: str            # paper / notebook / notes / other
    title: str = ""
    size_kb: float = 0.0
    hints: list[str] = field(default_factory=list)
    has_note: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "kind": self.kind,
            "title": self.title,
            "size_kb": round(self.size_kb, 1),
            "hints": self.hints,
            "has_note": self.has_note,
        }


@dataclass
class MaterialScan:
    """一次扫描的结果。``signals`` 是可以直接并进画像的证据。"""

    root: str
    materials: list[Material] = field(default_factory=list)
    signals: list[Signal] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    truncated: bool = False

    def by_kind(self, kind: str) -> list[Material]:
        return [m for m in self.materials if m.kind == kind]

    def summary(self) -> dict[str, Any]:
        return {
            "root": self.root,
            "counts": dict(self.counts),
            "papers": len(self.by_kind("paper")),
            "papers_with_notes": sum(1 for m in self.by_kind("paper") if m.has_note),
            "notebooks": len(self.by_kind("notebook")),
            "notes": len(self.by_kind("notes")),
            "truncated": self.truncated,
            "notes_text": list(self.notes),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.summary(),
            "materials": [m.to_dict() for m in self.materials],
            "signals": [s.to_dict() for s in self.signals],
        }


def _walk(root: Path) -> tuple[list[Path], bool]:
    """遍历目录。返回（文件列表, 是否被截断）。"""
    files: list[Path] = []
    truncated = False
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d.lower() not in SKIP_DIRS]
        for name in filenames:
            files.append(Path(dirpath) / name)
            if len(files) >= MAX_FILES:
                truncated = True
                return files, truncated
    return files, truncated


def _read_text(path: Path) -> str:
    try:
        with path.open("rb") as fh:
            raw = fh.read(MAX_READ_BYTES)
        return raw.decode("utf-8", errors="ignore")
    except OSError:
        return ""


def _notebook_text(path: Path) -> str:
    """只取代码单元，markdown 单元不算「写过代码」。"""
    try:
        data = json.loads(_read_text(path) or "{}")
    except json.JSONDecodeError:
        return ""
    chunks: list[str] = []
    for cell in data.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src = cell.get("source", "")
        chunks.append("".join(src) if isinstance(src, list) else str(src))
    return "\n".join(chunks)


def _note_stems(paths: list[Material]) -> set[str]:
    return {Path(m.path).stem for m in paths if len(Path(m.path).stem) > 1}


def _signal(
    rule_id: str,
    dimension: Dimension,
    level: Level,
    quote: str,
    why: str,
    ref: str,
) -> Signal:
    return Signal(
        rule_id=rule_id,
        dimension=dimension,
        level=level,
        strength="weak",
        quote=quote,
        why=why,
        polarity="positive",
        source=Source.FILE,
        source_ref=ref,
    )


def _code_signals(code_files: dict[str, list[str]]) -> list[Signal]:
    """目录里有 C/C++/Rust/Go 源码，说明动手的语言不是 Python。

    只算弱证据：有文件不等于写过、更不等于写得好；方向不同，这条证据的
    权重也不同，所以不在这里判断方向。
    """
    out: list[Signal] = []
    systems = code_files.get("systems") or []
    if len(systems) >= 3:
        out.append(
            _signal(
                "material.code_native",
                Dimension.CODING,
                Level.FOUNDATION,
                f"目录里有 {len(systems)} 个 C/C++/Rust/Go 源码（如 {systems[0]}）",
                "底层语言的文件通常意味着写过系统向的代码，但数量与质量都看不出来",
                systems[0],
            )
        )
    data = code_files.get("data") or []
    if data:
        out.append(
            _signal(
                "material.sql",
                Dimension.CODING,
                Level.FOUNDATION,
                f"目录里有 SQL 文件（如 {data[0]}）",
                "有建表或查询脚本，说明跟数据库打过交道",
                data[0],
            )
        )
    return out


def scan_materials(root: str | os.PathLike[str]) -> MaterialScan:
    """扫描授权目录，产出文件证据。

    目录不存在时抛 ``FileNotFoundError``，由调用方转成可读提示。
    """
    path = Path(root).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"目录不存在：{root}")
    if not path.is_dir():
        raise NotADirectoryError(f"不是目录：{root}")

    files, truncated = _walk(path)
    scan = MaterialScan(root=str(path), truncated=truncated)
    counts = {"paper": 0, "notebook": 0, "notes": 0, "other": 0}
    code_files: dict[str, list[str]] = {"systems": [], "data": []}

    for file in files:
        ext = file.suffix.lower()
        rel = file.relative_to(path).as_posix()
        try:
            size_kb = file.stat().st_size / 1024
        except OSError:
            size_kb = 0.0

        if ext in PAPER_EXT:
            kind = "paper"
        elif ext in NOTEBOOK_EXT:
            kind = "notebook"
        elif ext in NOTES_EXT:
            kind = "notes"
        else:
            kind = "other"
        counts[kind] += 1

        if ext in CODE_EXT_SYSTEMS:
            code_files["systems"].append(rel)
        elif ext in CODE_EXT_DATA:
            code_files["data"].append(rel)

        if kind == "other":
            continue
        scan.materials.append(
            Material(
                path=rel,
                kind=kind,
                title=file.stem.replace("_", " ").replace("-", " ").strip(),
                size_kb=size_kb,
            )
        )

    scan.counts = counts
    papers = scan.by_kind("paper")
    notebooks = scan.by_kind("notebook")
    notes = scan.by_kind("notes")

    # 论文旁边有同名笔记，才算「读过」；只有 PDF 只算「材料在」
    stems = _note_stems(notes)
    for paper in papers:
        paper.has_note = Path(paper.path).stem in stems

    scan.signals.extend(_paper_signals(papers))
    scan.signals.extend(_notebook_signals(notebooks, path))
    scan.signals.extend(_note_signals(notes, path))
    scan.signals.extend(_code_signals(code_files))

    scan.notes.append(
        "文件证据只能证明材料在，不能证明读过或会做；按弱证据处理，不定高中档。"
    )
    if truncated:
        scan.notes.append(f"文件数超过 {MAX_FILES}，只扫了前 {MAX_FILES} 个。")
    if not scan.materials:
        scan.notes.append("这个目录里没找到 PDF / notebook / 笔记。")
    return scan


def _paper_signals(papers: list[Material]) -> list[Signal]:
    if not papers:
        return []
    out: list[Signal] = []
    first = papers[0].path
    out.append(
        _signal(
            "material.papers",
            Dimension.RESEARCH,
            Level.FOUNDATION,
            f"目录里有 {len(papers)} 篇 PDF（如 {first}）",
            "手上有论文，说明至少开始看文献；但下载不等于读过，所以只给「有基础」",
            first,
        )
    )
    with_notes = [p for p in papers if p.has_note]
    if with_notes:
        out.append(
            _signal(
                "material.papers_noted",
                Dimension.RESEARCH,
                Level.WORKING,
                f"{len(with_notes)} 篇论文旁边有同名笔记（如 {with_notes[0].path}）",
                "读过并留下笔记，比单纯存放 PDF 可信，但仍不判断读得多深",
                with_notes[0].path,
            )
        )
    return out


def _notebook_signals(notebooks: list[Material], root: Path) -> list[Signal]:
    if not notebooks:
        return []
    out: list[Signal] = []
    torch_nb: list[str] = []
    train_nb: list[str] = []
    sklearn_nb: list[str] = []

    for nb in notebooks:
        code = _notebook_text(root / nb.path)
        if not code:
            continue
        libs = set(IMPORT_RE.findall(code))
        nb.hints = sorted(libs)
        if "torch" in libs or "tensorflow" in libs or "jax" in libs:
            torch_nb.append(nb.path)
            if any(hint in code for hint in TRAIN_HINTS):
                train_nb.append(nb.path)
        if "sklearn" in libs:
            sklearn_nb.append(nb.path)

    if torch_nb:
        out.append(
            _signal(
                "material.notebook_torch",
                Dimension.CODING,
                Level.FOUNDATION,
                f"notebook 里 import 了框架：{torch_nb[0]}",
                "写过框架的代码，说明装过、跑过基础 API；是否跑完训练另说",
                torch_nb[0],
            )
        )
    if train_nb:
        out.append(
            _signal(
                "material.notebook_train",
                Dimension.CODING,
                Level.WORKING,
                f"notebook 里有训练循环：{train_nb[0]}",
                "出现 optimizer/backward/epoch，说明至少走过一次完整训练流程",
                train_nb[0],
            )
        )
    if sklearn_nb:
        out.append(
            _signal(
                "material.notebook_sklearn",
                Dimension.CODING,
                Level.FOUNDATION,
                f"notebook 里用了 sklearn：{sklearn_nb[0]}",
                "调过 sklearn，走过 fit/评估这条链",
                sklearn_nb[0],
            )
        )
    if len(notebooks) >= 3:
        out.append(
            _signal(
                "material.notebooks",
                Dimension.CODING,
                Level.FOUNDATION,
                f"目录里有 {len(notebooks)} 个 notebook",
                "notebook 数量多说明常用 Python 做实验，但代码质量看不出来",
                notebooks[0].path,
            )
        )
    return out


def _note_signals(notes: list[Material], root: Path) -> list[Signal]:
    """笔记里同一维度的关键词出现 2 个以上，才给一条弱证据。"""
    if not notes:
        return []
    hits: dict[Dimension, list[str]] = {d: [] for d in NOTE_KEYWORDS}
    for note in notes:
        text = _read_text(root / note.path)
        for dim, words in NOTE_KEYWORDS.items():
            found = [w for w in words if w in text]
            if len(found) >= 2 and note.path not in hits[dim]:
                hits[dim].append(note.path)

    out: list[Signal] = []
    for dim, paths in hits.items():
        if not paths:
            continue
        words = [w for w in NOTE_KEYWORDS[dim] if w in _read_text(root / paths[0])]
        out.append(
            _signal(
                f"material.notes.{dim.value}",
                dim,
                Level.FOUNDATION,
                f"笔记里出现 {'、'.join(words[:3])}：{paths[0]}",
                "笔记里有这一维的术语，说明碰过相关概念；记笔记不等于掌握",
                paths[0],
            )
        )
    return out
