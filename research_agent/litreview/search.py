"""多源检索编排：宽召回 → 本地过滤 → 身份对齐 → 记覆盖度。

这里是全包最该"保守"的地方。本机实测的教训（2026-11-05）：

- Crossref 检索 "retrieval augmented generation"，返回的前 20 条里主题命中 **0 条**
  ——它把查询拆词后，被 ``augmented`` / ``generation`` 带偏到"投影缀加平面波方法""新一代测序"。
- OpenAlex 匿名检索直接 429；arXiv 的 http 端点 15 秒超时。

所以检索层不能"信服务端排序"。做法是三层：

1. **宽召回**：每个源多给一点，不指望第一次就准；
2. **本地过滤**：按问题词在标题/摘要里的覆盖度打分，零分的直接丢，
   并把这个数字报给用户（"Crossref 返回 20 条，19 条与本问题无关，已丢弃"）；
3. **合并去重**：交给 ``identity.resolve``，拿不准的只标不并。

任何一层出问题都写进 ``warnings`` / ``Coverage``，不做静默降级。
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Iterable

from .cache import Cache, default_root
from .identity import resolve
from .spec import Coverage, PaperRef, Retrieval, SourceStatus
from .sources import SourceError, build_sources

# 查询里常见但不携带主题信息的词，参与打分只会加噪声
STOPWORDS = {
    "a", "an", "the", "of", "for", "and", "or", "in", "on", "with", "to", "via",
    "using", "based", "towards", "toward", "from", "by", "is", "are", "how", "what",
    "why", "when", "which", "can", "do", "does", "paper", "papers", "survey", "review",
}

_CJK = re.compile(r"[\u4e00-\u9fff]")


def terms_of(text: str) -> list[str]:
    """把问题拆成检索词。英文按词，中文按二元组（不做分词，够用且不引依赖）。"""
    lowered = (text or "").lower()
    words = [w for w in re.findall(r"[a-z][a-z0-9\-]{1,}", lowered) if w not in STOPWORDS]
    cjk = re.findall(r"[\u4e00-\u9fff]+", lowered)
    grams: list[str] = []
    for chunk in cjk:
        if len(chunk) <= 2:
            grams.append(chunk)
        else:
            grams.extend(chunk[i:i + 2] for i in range(len(chunk) - 1))
    return list(dict.fromkeys(words + grams))


def relevance(question: str, paper: PaperRef) -> float:
    """0~1：问题词在标题/摘要里的覆盖度。标题命中权重是摘要的 2 倍。

    这是**过滤信号**，不是排序信号——排序还要看年份、被引、是否综述（见 ``triage``）。
    """
    ts = terms_of(question)
    if not ts:
        return 1.0
    title = " ".join(paper.title.split()).lower()
    abstract = (paper.abstract or "").lower()
    hit = 0.0
    for t in ts:
        if t in title:
            hit += 1.0
        elif t in abstract:
            hit += 0.5
        elif _CJK.search(t) and t in (title + abstract):
            hit += 0.5
    return min(1.0, hit / len(ts))


def _queries_for(question: str) -> list[str]:
    """同一问题给两个检索式：原句 + 去停用词的核心词。

    多条检索式不是"多试几个"，是为了让结果可解释——每条都写进 ``Retrieval.queries``。
    """
    core = " ".join(terms_of(question)[:8])
    out = [" ".join(question.split())]
    if core and core.lower() != question.strip().lower():
        out.append(core)
    return out


def search(
    question: str,
    *,
    limit: int = 20,
    per_source: int | None = None,
    year_from: int | None = None,
    order: tuple[str, ...] | None = None,
    cache: Cache | None = None,
    offline: bool = False,
    strict: bool = True,
    threshold: float = 0.34,
) -> Retrieval:
    """跑一次完整检索。**不抛异常**：源失败记进 coverage，结果为空也是合法结果。"""
    cache = cache or Cache(offline=offline)
    sources = build_sources(cache, order=order)
    started = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    queries = _queries_for(question)
    per_source = per_source or max(limit, 20)

    coverage = Coverage()
    collected: list[PaperRef] = []
    dropped = 0
    warnings: list[str] = []

    for src in sources:
        status = SourceStatus(source=src.name, attempts=0)
        t0 = time.time()
        for q in queries:
            try:
                status.attempts += 1
                got = src.search(q, limit=per_source, year_from=year_from)
            except SourceError as exc:
                status.ok = False
                status.error = exc.message
                if not exc.retryable:
                    break
                continue
            except Exception as exc:  # noqa: BLE001  —— 适配器 bug 也不能让整次检索崩
                status.ok = False
                status.error = f"{type(exc).__name__}: {exc}"
                continue
            status.ok = status.ok or bool(got)
            status.count += len(got)
            collected.extend(got)
        status.elapsed_ms = int((time.time() - t0) * 1000)
        status.cached = cache.hits > 0
        if not status.ok and not status.error:
            status.error = "没有返回结果"
        coverage.statuses.append(status)

    # ---- 本地过滤：零分的丢掉，并如实报告丢了多少 ----------------------
    kept: list[PaperRef] = []
    for p in collected:
        if relevance(question, p) <= 0:
            dropped += 1
            continue
        kept.append(p)

    if strict:
        kept = [p for p in kept if relevance(question, p) >= threshold]
        dropped = len(collected) - len(kept)

    if collected and dropped:
        warnings.append(
            f"共取回 {len(collected)} 条，按问题词过滤后保留 {len(kept)} 条，"
            f"丢弃 {dropped} 条偏题结果——关键词检索跑偏是常态，不是意外"
        )
    if not collected:
        warnings.append("所有来源都没有返回结果。检查网络、API key，或换个更具体的问题")

    papers, identities, dedupe_warnings = resolve(kept)
    warnings.extend(dedupe_warnings)

    # 排序：相关度 → 影响力 → 年份。被引**未知**的源给中性分（见 triage._impact 的注释），
    # 否则 arXiv 的预印本会被系统性地压到"0 引用的书章节"后面。
    def rank(p: PaperRef) -> tuple:
        impact = (p.cited_by or 0) if p.cited_known else 20
        return (-relevance(question, p), -impact, -(p.year or 0))

    papers.sort(key=rank)
    papers = papers[:limit]

    finished = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    return Retrieval(
        question=question,
        queries=queries,
        filters={
            "limit": limit, "per_source": per_source, "year_from": year_from,
            "strict": strict, "threshold": threshold,
            "order": [s.name.value for s in sources], "offline": offline,
        },
        started_at=started,
        finished_at=finished,
        coverage=coverage,
        papers=papers,
        identities={k: v for k, v in identities.items() if k in {_key(p, identities) for p in papers}},
        dropped=dropped,
        warnings=warnings,
    )


def _key(paper: PaperRef, identities: dict) -> str:
    from .identity import key_of

    return key_of(paper)


# ---------------------------------------------------------------- 快照

def snapshot_path(name: str = "last", root: Path | None = None) -> Path:
    base = root or default_root()
    return Path(base) / f"snapshot-{name}.json"


def save_snapshot(retrieval: Retrieval, path: Path | None = None) -> Path:
    """把检索现场落盘。重放靠它，审计也靠它。"""
    target = path or snapshot_path(retrieval.fingerprint())
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(retrieval.to_dict(), ensure_ascii=False, indent=1), encoding="utf-8"
    )
    return target


def load_snapshot(path: Path) -> Retrieval:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    from .spec import Confidence, Identity, Source, SourceStatus

    papers = [
        PaperRef(
            source=Source(p["source"]), source_id=p["source_id"], title=p["title"],
            authors=p.get("authors") or [], year=p.get("year"), venue=p.get("venue", ""),
            abstract=p.get("abstract", ""), url=p.get("url", ""), doi=p.get("doi", ""),
            arxiv_id=p.get("arxiv_id", ""), openalex_id=p.get("openalex_id", ""),
            cited_by=p.get("cited_by", 0), cited_known=p.get("cited_known", True),
            is_survey=p.get("is_survey", False),
            fields=p.get("fields") or [], fetched_at=p.get("fetched_at", ""),
        )
        for p in raw.get("papers") or []
    ]
    cov = raw.get("coverage") or {}
    statuses = [
        SourceStatus(
            source=Source(s["source"]), ok=s.get("ok", False), attempts=s.get("attempts", 0),
            count=s.get("count", 0), error=s.get("error", ""), elapsed_ms=s.get("elapsed_ms", 0),
        )
        for s in cov.get("statuses") or []
    ]
    identities = {
        k: Identity(
            canonical_key=v["canonical_key"], doi=v.get("doi", ""), arxiv_id=v.get("arxiv_id", ""),
            openalex_id=v.get("openalex_id", ""), s2_id=v.get("s2_id", ""),
            match_method=v.get("match_method", ""),
            confidence=Confidence(v.get("confidence", "medium")),
            possible_duplicates=v.get("possible_duplicates") or [],
            merged=v.get("merged") or [],
        )
        for k, v in (raw.get("identities") or {}).items()
    }
    return Retrieval(
        question=raw.get("question", ""), queries=raw.get("queries") or [],
        filters=raw.get("filters") or {}, started_at=raw.get("started_at", ""),
        finished_at=raw.get("finished_at", ""), coverage=Coverage(statuses=statuses),
        papers=papers, identities=identities, dropped=raw.get("dropped", 0),
        warnings=raw.get("warnings") or [],
    )
