"""来源适配器集合。

默认顺序不是拍脑袋定的，是按本机实测的可靠性与信息量排的：

1. **openalex** —— 覆盖面最广、自带引用网络，画图靠它；匿名检索会被限流，可配 key。
2. **crossref** —— DOI 权威、元数据准，用来补身份；相关性排序不可信，只当补充。
3. **arxiv** —— 唯一能拿到最新预印本的源，但限速 3 秒且国内网络时通时断，放最后。

上层如果发现某个源失败，不是报错，是在 ``Coverage`` 里记一笔然后继续用下一个——
这就是"降级链"。
"""

from __future__ import annotations

from ..cache import Cache
from .arxiv import ArxivSource
from .base import Http, SourceError
from .crossref import CrossrefSource
from .openalex import OpenAlexSource

DEFAULT_ORDER = ("openalex", "crossref", "arxiv")

_ADAPTERS = {
    "openalex": OpenAlexSource,
    "crossref": CrossrefSource,
    "arxiv": ArxivSource,
}


def build_sources(
    cache: Cache,
    order: tuple[str, ...] | None = None,
    timeout: float = 12.0,
) -> list:
    """按顺序构造适配器。未知名字直接忽略并返回已构造的，不抛异常。"""
    http = Http(cache=cache, timeout=timeout)
    out = []
    for name in order or DEFAULT_ORDER:
        cls = _ADAPTERS.get(name)
        if cls is None:
            continue
        out.append(cls(http=http))
    return out


__all__ = [
    "DEFAULT_ORDER",
    "ArxivSource",
    "CrossrefSource",
    "Http",
    "OpenAlexSource",
    "SourceError",
    "build_sources",
]
