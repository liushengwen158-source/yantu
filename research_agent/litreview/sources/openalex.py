"""OpenAlex 适配器：引用网络的主力。

它给我们的东西正好是画图需要的：``referenced_works``（引了谁）、
``filter=cites:W...``（谁引了我）、``related_works``（站方给的近邻）、
``topics``（四级主题）。全部免费，但要守两条现实约束：

- 匿名**检索**会被限流（本机实测 429 + ``retryAfter``），规模化必须带 key：
  设环境变量 ``OPENALEX_API_KEY``；
- 单条 DOI 查询通常 200，但也会超时——所以上层要允许多源降级。

摘要字段是"倒排索引"（词 → 位置列表），这里自己还原成正常语序。
"""

from __future__ import annotations

import os
import time

from ..spec import PaperRef, Source
from .base import Http, SourceError, join_query, looks_like_survey

API = "https://api.openalex.org"
SELECT = (
    "id,doi,display_name,publication_year,cited_by_count,type,primary_location,"
    "authorships,abstract_inverted_index,referenced_works,related_works,topics,is_retracted"
)


def _abstract(inv: dict | None) -> str:
    """倒排索引 → 正文。位置从 0 开始，缺词的洞用空串补。"""
    if not inv:
        return ""
    slots: dict[int, str] = {}
    for word, positions in inv.items():
        for p in positions or []:
            slots[int(p)] = word
    if not slots:
        return ""
    return " ".join(slots.get(i, "") for i in range(max(slots) + 1)).strip()


def _short_id(value: str) -> str:
    return (value or "").rstrip("/").split("/")[-1]


def to_paper(work: dict, fetched_at: str) -> PaperRef:
    location = work.get("primary_location") or {}
    source_info = location.get("source") or {}
    authors = [
        ((a.get("author") or {}).get("display_name") or "")
        for a in work.get("authorships") or []
    ]
    doi = (work.get("doi") or "").replace("https://doi.org/", "")
    title = work.get("display_name") or (work.get("title") or "")
    types = work.get("type") or ""
    return PaperRef(
        source=Source.OPENALEX,
        source_id=_short_id(work.get("id", "")),
        title=title,
        authors=[a for a in authors if a],
        year=work.get("publication_year"),
        venue=source_info.get("display_name") or "",
        abstract=_abstract(work.get("abstract_inverted_index")),
        url=work.get("id") or "",
        doi=doi,
        openalex_id=_short_id(work.get("id", "")),
        cited_by=work.get("cited_by_count") or 0,
        is_survey=looks_like_survey(title),
        fields=[(t.get("display_name") or "") for t in (work.get("topics") or [])[:2] if t.get("display_name")],
        fetched_at=fetched_at,
        extra={
            "type": types,
            "referenced_works": [_short_id(x) for x in (work.get("referenced_works") or [])],
            "related_works": [_short_id(x) for x in (work.get("related_works") or [])],
            "is_retracted": bool(work.get("is_retracted")),
        },
    )


class OpenAlexSource:
    name = Source.OPENALEX

    def __init__(self, http: Http | None = None, api_key: str | None = None):
        self.http = http or Http(min_interval={"api.openalex.org": 0.3})
        self.api_key = api_key if api_key is not None else os.environ.get("OPENALEX_API_KEY", "")

    # -- 内部 -------------------------------------------------------------
    def _get(self, path: str, params: dict) -> tuple[dict, bool]:
        if self.api_key:
            params = {**params, "api_key": self.api_key}
        url = f"{API}{path}?{join_query(params)}"
        data, cached = self.http.get_json(url, source=self.name.value)
        if not isinstance(data, dict):
            raise SourceError(self.name.value, "返回结构不是对象", retryable=False)
        return data, cached

    # -- 对外 -------------------------------------------------------------
    def search(self, query: str, limit: int = 20, year_from: int | None = None) -> list[PaperRef]:
        params: dict = {"search": query, "per-page": min(limit, 100), "select": SELECT}
        filters = []
        if year_from:
            filters.append(f"from_publication_date:{year_from}-01-01")
        if filters:
            params["filter"] = ",".join(filters)
        data, _ = self._get("/works", params)
        fetched_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        return [to_paper(w, fetched_at) for w in data.get("results") or []][:limit]

    def by_ids(self, ids: list[str], limit: int = 50) -> list[PaperRef]:
        """批量取元数据。OpenAlex 支持 ``openalex_id:W1|W2`` 一次 50 条。"""
        clean = [_short_id(i) for i in ids if i]
        if not clean:
            return []
        out: list[PaperRef] = []
        for start in range(0, len(clean), 50):
            chunk = clean[start:start + 50]
            data, _ = self._get("/works", {
                "filter": "openalex_id:" + "|".join(chunk),
                "per-page": 50,
                "select": SELECT,
            })
            fetched_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
            out.extend(to_paper(w, fetched_at) for w in data.get("results") or [])
        return out[:limit]

    def by_doi(self, doi: str) -> PaperRef | None:
        from ..identity import normalize_doi

        d = normalize_doi(doi)
        if not d:
            return None
        data, _ = self._get("/works", {"filter": f"doi:{d}", "per-page": 1, "select": SELECT})
        results = data.get("results") or []
        if not results:
            return None
        return to_paper(results[0], time.strftime("%Y-%m-%dT%H:%M:%S%z"))

    def citing(self, work_id: str, limit: int = 25) -> list[PaperRef]:
        """谁引用了它（逆向）。"""
        wid = _short_id(work_id)
        if not wid.startswith("W"):
            return []
        data, _ = self._get("/works", {
            "filter": f"cites:{wid}",
            "per-page": min(limit, 100),
            "sort": "cited_by_count:desc",
            "select": SELECT,
        })
        fetched_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        return [to_paper(w, fetched_at) for w in data.get("results") or []][:limit]
