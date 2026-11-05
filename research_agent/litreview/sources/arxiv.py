"""arXiv 适配器：拿最新预印本。

arXiv 的价值是"比正式发表早 6~18 个月"，代价是：单源、无引用关系、约 3 秒/请求的
礼节性限速，而且国内网络时通时断（本机实测 http 直接超时，https 时好时坏）。

返回 Atom XML，这里用标准库 ``xml.etree`` 解析，不引入 feedparser。
"""

from __future__ import annotations

import re
import time
import xml.etree.ElementTree as ET

from ..spec import PaperRef, Source
from .base import Http, SourceError, join_query, looks_like_survey

API = "https://export.arxiv.org/api/query"
NS = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
_ARXIV_ID = re.compile(r"(?:arxiv:|abs/)?(\d{4}\.\d{4,5})(v\d+)?", re.I)


def _text(node: ET.Element | None, path: str) -> str:
    if node is None:
        return ""
    found = node.find(path, NS)
    return " ".join((found.text or "").split()) if found is not None else ""


def parse_feed(xml_text: str, fetched_at: str) -> list[PaperRef]:
    """把 Atom 解析成 PaperRef。解析失败抛 SourceError，不返回半截数据。"""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        raise SourceError(Source.ARXIV.value, f"返回不是合法 XML：{exc}", retryable=False) from exc

    out: list[PaperRef] = []
    for entry in root.findall("a:entry", NS):
        raw_id = _text(entry, "a:id")
        m = _ARXIV_ID.search(raw_id)
        if not m:
            continue
        aid = m.group(1)
        title = _text(entry, "a:title")
        summary = _text(entry, "a:summary")
        published = _text(entry, "a:published")
        year = int(published[:4]) if published[:4].isdigit() else None
        authors = [
            _text(a, "a:name") for a in entry.findall("a:author", NS) if _text(a, "a:name")
        ]
        primary = entry.find("arxiv:primary_category", NS)
        fields = [primary.get("term")] if primary is not None and primary.get("term") else []
        doi = _text(entry, "arxiv:doi")
        comment = _text(entry, "arxiv:comment")
        out.append(
            PaperRef(
                source=Source.ARXIV,
                source_id=aid,
                title=title,
                authors=authors,
                year=year,
                venue="arXiv",
                abstract=summary,
                url=f"https://arxiv.org/abs/{aid}",
                arxiv_id=aid,
                doi=doi,
                cited_known=False,  # arXiv API 不返回被引数：未知 ≠ 0
                is_survey=looks_like_survey(title),
                fields=fields,
                fetched_at=fetched_at,
                extra={"comment": comment[:200]} if comment else {},
            )
        )
    return out


class ArxivSource:
    name = Source.ARXIV

    def __init__(self, http: Http | None = None):
        self.http = http or Http(min_interval={"export.arxiv.org": 3.0})

    def search(self, query: str, limit: int = 20, year_from: int | None = None) -> list[PaperRef]:
        """``query`` 用 arXiv 语法（``all:xxx`` / ``ti:xxx`` / ``abs:xxx``）。"""
        q = query if ":" in query else f'all:"{query}"'
        if year_from:
            q = f"{q} AND submittedDate:[{year_from}01010000 TO 999912312359]"
        url = f"{API}?{join_query({'search_query': q, 'start': 0, 'max_results': min(limit, 100), 'sortBy': 'relevance'})}"
        fetched_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        text, _ = self._fetch(url)
        return parse_feed(text, fetched_at)[:limit]

    def _fetch(self, url: str) -> tuple[str, bool]:
        try:
            res = self.http.get_text(url, source=self.name.value, accept="application/atom+xml")
        except SourceError:
            raise
        return str(res.payload), res.cached
