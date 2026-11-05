"""跨源身份对齐：同一篇论文在 arXiv / OpenAlex / Crossref 里有三个 id，得先认出来是一个人。

对齐顺序（能确定的先确定，不能确定的不猜）：

    DOI 归一化  →  arXiv ID（去版本号）  →  标题指纹 + 年份 ±1  →  单源独立

三条纪律：

1. **先验 id，后验文本**。DOI 和 arXiv ID 是权威的，标题是启发式的。
2. **模糊匹配只标不并**。标题指纹撞上但年份差 >1，写进 ``possible_duplicates``，
   不合并——合并错了会把"被引 3 次"算成"被引 300 次"，而且不可逆。
3. **每步都记账**。``match_method`` 写清是靠什么认出来的，用户能复核。
"""

from __future__ import annotations

import re
import unicodedata
from typing import Iterable

from .spec import Confidence, Identity, PaperRef, Source

_DOI_PREFIXES = ("https://doi.org/", "http://doi.org/", "https://dx.doi.org/", "doi:")
_ARXIV_RE = re.compile(r"(?:arxiv[:/]|abs/)?(\d{4}\.\d{4,5})(v\d+)?", re.I)
_ARXIV_OLD = re.compile(r"([a-z\-]+(?:\.[A-Z]{2})?/\d{7})(v\d+)?")


def normalize_doi(doi: str) -> str:
    d = (doi or "").strip().lower()
    for p in _DOI_PREFIXES:
        if d.startswith(p):
            d = d[len(p):]
    return d.strip().rstrip(".")


def normalize_arxiv(value: str) -> str:
    """``arXiv:2401.01234v2`` → ``2401.01234``。版本号不属于身份。"""
    v = (value or "").strip()
    if not v:
        return ""
    v = v.replace("arXiv:", "").replace("arxiv:", "").strip()
    if "/abs/" in v:
        v = v.split("/abs/")[-1]
    m = _ARXIV_RE.search(v)
    if m:
        return m.group(1)
    m2 = _ARXIV_OLD.search(v)
    if m2:
        return m2.group(1)
    return ""


def title_fingerprint(title: str) -> str:
    """标题指纹：全角转半角、去标点、压空格、小写。用于跨源比对，不用于展示。"""
    t = unicodedata.normalize("NFKC", title or "").lower()
    t = re.sub(r"<[^>]+>", " ", t)          # 去掉 arXiv 摘要里偶尔混入的标签
    t = re.sub(r"[^0-9a-z\u4e00-\u9fff ]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def canonical_key(paper: PaperRef) -> tuple[str, str, Confidence]:
    """给出规范键与匹配方式。优先级：doi > arxiv > openalex > s2 > 标题指纹。"""
    doi = normalize_doi(paper.doi)
    if doi:
        return f"doi:{doi}", "doi", Confidence.HIGH
    arxiv = normalize_arxiv(paper.arxiv_id or (paper.source_id if paper.source is Source.ARXIV else ""))
    if arxiv:
        return f"arxiv:{arxiv}", "arxiv", Confidence.HIGH
    if paper.openalex_id:
        return f"openalex:{paper.openalex_id.split('/')[-1]}", "openalex", Confidence.MEDIUM
    sid = paper.source_id or ""
    if paper.source is Source.S2 and sid:
        return f"s2:{sid}", "s2", Confidence.MEDIUM
    fp = title_fingerprint(paper.title)
    if fp:
        year = paper.year or 0
        return f"title:{fp}|{year}", "title_exact", Confidence.LOW
    return f"{paper.source.value}:{sid}", "single_source", Confidence.LOW


def _merge_into(base: PaperRef, other: PaperRef) -> None:
    """把 ``other`` 里 base 缺的信息补上。不改标题、不改年份——那是身份的一部分。"""
    if not base.doi and other.doi:
        base.doi = normalize_doi(other.doi)
    if not base.arxiv_id and other.arxiv_id:
        base.arxiv_id = normalize_arxiv(other.arxiv_id)
    if not base.openalex_id and other.openalex_id:
        base.openalex_id = other.openalex_id
    if not base.abstract and other.abstract:
        base.abstract = other.abstract
    if not base.url and other.url:
        base.url = other.url
    if not base.venue and other.venue:
        base.venue = other.venue
    if not base.authors and other.authors:
        base.authors = other.authors
    # 被引数取最大值：不同源的统计口径不同，取最大比取最后一个诚实
    base.cited_by = max(base.cited_by, other.cited_by)
    base.extra.setdefault("sources", [])
    tag = f"{other.source.value}:{other.source_id}"
    if tag not in base.extra["sources"]:
        base.extra["sources"].append(tag)


def resolve(papers: Iterable[PaperRef]) -> tuple[list[PaperRef], dict[str, Identity], list[str]]:
    """去重 + 对齐。返回 (代表篇目, key→Identity, 警告)。

    代表篇目是"信息最全的那一个"，其余同身份的被合并进去（只补字段，不丢来源标记）。
    """
    warnings: list[str] = []
    by_key: dict[str, PaperRef] = {}
    identities: dict[str, Identity] = {}
    fuzzy_index: dict[str, list[str]] = {}

    for p in papers:
        if not p.valid:
            continue
        key, method, conf = canonical_key(p)
        ident = identities.get(key)
        if ident is None:
            identities[key] = Identity(
                canonical_key=key, match_method=method, confidence=conf,
                doi=normalize_doi(p.doi) or "",
                arxiv_id=normalize_arxiv(p.arxiv_id or (p.source_id if p.source is Source.ARXIV else "")) or "",
                openalex_id=p.openalex_id,
                s2_id=p.source_id if p.source is Source.S2 else "",
                merged=[f"{p.source.value}:{p.source_id}"],
            )
            by_key[key] = p
        else:
            _merge_into(by_key[key], p)
            ident.merged.append(f"{p.source.value}:{p.source_id}")
            if not ident.doi and p.doi:
                ident.doi = normalize_doi(p.doi)
            if not ident.arxiv_id and p.arxiv_id:
                ident.arxiv_id = normalize_arxiv(p.arxiv_id)
        fp = title_fingerprint(p.title)
        if fp:
            fuzzy_index.setdefault(fp, []).append(key)

    # 标题相同但规范键不同 → 只标疑似重复，不合并
    for fp, keys in fuzzy_index.items():
        uniq = list(dict.fromkeys(keys))
        if len(uniq) < 2:
            continue
        for k in uniq:
            others = [x for x in uniq if x != k]
            ident = identities[k]
            ident.possible_duplicates.extend(others)
            if ident.confidence is Confidence.HIGH:
                continue
            ident.confidence = Confidence.LOW
        warnings.append(f"标题指纹相同但 id 不同，已标记待人工确认：{fp[:80]}")

    return list(by_key.values()), identities, warnings


def key_of(paper: PaperRef) -> str:
    """单篇取键，供图、排序等只需要 key 的地方复用。"""
    return canonical_key(paper)[0]
