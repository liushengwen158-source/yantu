"""Crossref 适配器：DOI 权威 + 参考文献补位。

它的长处是元数据最准（DOI 就是它注册的），短处有两条，都是实测撞出来的：

1. **相关性排序不可信**。检索 "retrieval augmented generation" 时，Crossref 拆词后
   被 ``augmented`` / ``generation`` 带偏，返回"投影缀加平面波方法""新一代测序"。
   → 所以本包用本地主题过滤兜底（``search.relevance``），不把排序当真理。
2. **参考文献覆盖率参差**。期刊文章常有 ``reference`` 字段，书章节常常没有；
   而且单条查询经常 20~45s 超时。→ 所以建图时优先用 OpenAlex，Crossref 只补 DOI。
"""

from __future__ import annotations

import time
import urllib.parse

from ..spec import PaperRef, Source
from .base import Http, SourceError, join_query, looks_like_survey

API = "https://api.crossref.org"
SELECT = "DOI,title,issued,is-referenced-by-count,author,container-title,type,abstract,reference,URL"


def _year(item: dict) -> int | None:
    for key in ("issued", "published-print", "published-online", "created"):
        parts = ((item.get(key) or {}).get("date-parts") or [[None]])
        if parts and parts[0] and parts[0][0]:
            try:
                return int(parts[0][0])
            except (TypeError, ValueError):
                continue
    return None


def to_paper(item: dict, fetched_at: str) -> PaperRef:
    title = (item.get("title") or [""])[0]
    authors = [
        (a.get("family") or a.get("name") or "")
        for a in item.get("author") or []
        if (a.get("family") or a.get("name"))
    ]
    refs = []
    for ref in item.get("reference") or []:
        doi = (ref.get("DOI") or "").strip().lower()
        if doi:
            refs.append(doi.replace("https://doi.org/", ""))
    return PaperRef(
        source=Source.CROSSREF,
        source_id=(item.get("DOI") or "").lower(),
        title=" ".join(title.split()),
        authors=authors,
        year=_year(item),
        venue=(item.get("container-title") or [""])[0],
        abstract=" ".join((item.get("abstract") or "").split())[:4000],
        url=item.get("URL") or "",
        doi=(item.get("DOI") or "").lower(),
        cited_by=int(item.get("is-referenced-by-count") or 0),
        is_survey=looks_like_survey(title),
        fetched_at=fetched_at,
        extra={"type": item.get("type", ""), "references": refs},
    )


class CrossrefSource:
    name = Source.CROSSREF

    def __init__(self, http: Http | None = None, mailto: str = ""):
        self.http = http or Http(timeout=15.0)
        self.mailto = mailto

    def search(self, query: str, limit: int = 20, year_from: int | None = None) -> list[PaperRef]:
        params: dict = {
            "query.bibliographic": query,
            "rows": min(limit, 100),
            "select": SELECT,
        }
        if year_from:
            params["filter"] = f"from-pub-date:{year_from}-01-01"
        if self.mailto:
            params["mailto"] = self.mailto
        data, _ = self.http.get_json(f"{API}/works?{join_query(params)}", source=self.name.value)
        items = (((data or {}).get("message") or {}).get("items")) or []
        fetched_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        return [to_paper(it, fetched_at) for it in items][:limit]

    def by_doi(self, doi: str) -> PaperRef | None:
        from ..identity import normalize_doi

        d = normalize_doi(doi)
        if not d:
            return None
        data, _ = self.http.get_json(
            f"{API}/works/{urllib.parse.quote(d)}", source=self.name.value
        )
        message = (data or {}).get("message")
        if not isinstance(message, dict):
            raise SourceError(self.name.value, "返回缺少 message 字段", retryable=False)
        return to_paper(message, time.strftime("%Y-%m-%dT%H:%M:%S%z"))
