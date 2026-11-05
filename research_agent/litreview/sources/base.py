"""来源适配层的公共部分：一个够用的 HTTP 客户端 + 统一的失败类型。

三条和工具层一致的约定：

- **失败不抛到用户面前**。每个源抛 ``SourceError``，由上层记进 ``Coverage``，
  变成"这个源这次不可用"，而不是让整次检索崩掉。
- **限速当礼貌，不是可选项**。arXiv 官方建议间隔 3 秒，这里按 host 做最小间隔。
- **先查缓存再发请求**。离线模式下只读缓存，读不到就报不可用——测试靠这条不联网。
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any

from ..cache import Cache

USER_AGENT = "yantu-litreview/0.1 (research prototype; zero-dependency)"

# 综述识别：标题里出现这些词就算"可能是综述"。这是一条**弱信号**，
# 只用来把综述往前排，不用来下结论——所以宁可漏判，也不要按词猜错。
SURVEY_HINTS = ("survey", "review", "overview", "tutorial", "state of the art", "systematic")

# 反向排除：同行评审记录、勘误、评论文章的标题里也带 "review"，但它们不是综述。
# 实测踩过：Crossref 的 "Review of: \"GeAR: Generation Augmented Retrieval\"" 被误判成综述，
# 直接顶到"现在精读"第一条——对新手是很坏的引导。
_SURVEY_NEVER = ("review of", "peer review", "response to review", "comment on",
                 "correction to", "erratum", "retraction")


def looks_like_survey(title: str) -> bool:
    t = " ".join((title or "").lower().split())
    if not t:
        return False
    if any(t.startswith(x) for x in _SURVEY_NEVER):
        return False
    return any(h in t for h in SURVEY_HINTS)


class SourceError(RuntimeError):
    """某个源这次不可用。message 要能直接展示给用户。"""

    def __init__(self, source: str, message: str, retryable: bool = True):
        super().__init__(f"{source}: {message}")
        self.source = source
        self.message = message
        self.retryable = retryable


@dataclass
class FetchResult:
    payload: Any
    cached: bool
    attempts: int
    elapsed_ms: int


class Http:
    """最小 HTTP 客户端。只做三件事：限速、重试退避、读写缓存。"""

    def __init__(
        self,
        cache: Cache | None = None,
        timeout: float = 12.0,
        tries: int = 2,
        min_interval: dict[str, float] | None = None,
    ):
        self.cache = cache or Cache()
        self.timeout = timeout
        self.tries = max(1, tries)
        self.min_interval = min_interval or {"export.arxiv.org": 3.0, "api.openalex.org": 0.3}
        self._last_call: dict[str, float] = {}

    # -- 限速 -------------------------------------------------------------
    def _wait(self, host: str) -> None:
        gap = self.min_interval.get(host, 0.0)
        if gap <= 0:
            return
        last = self._last_call.get(host)
        if last is not None:
            delta = time.time() - last
            if delta < gap:
                time.sleep(gap - delta)
        self._last_call[host] = time.time()

    # -- 取文本 -----------------------------------------------------------
    def get_text(self, url: str, *, source: str, accept: str = "application/json") -> FetchResult:
        cached = self.cache.get(url)
        if cached is not None:
            return FetchResult(payload=cached, cached=True, attempts=0, elapsed_ms=0)
        if self.cache.offline:
            raise SourceError(source, "离线模式且无缓存", retryable=False)

        host = urllib.parse.urlparse(url).hostname or ""
        started = time.time()
        last = ""
        for i in range(self.tries):
            self._wait(host)
            try:
                req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": accept})
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    text = resp.read().decode("utf-8", errors="replace")
                elapsed = int((time.time() - started) * 1000)
                # 文本响应也要落盘：arXiv 返回的是 Atom XML，不缓存的话离线模式永远缺它
                self.cache.put(url, text)
                return FetchResult(payload=text, cached=False, attempts=i + 1, elapsed_ms=elapsed)
            except urllib.error.HTTPError as exc:
                last = f"HTTP {exc.code}"
                if exc.code in (400, 401, 403, 404):
                    raise SourceError(source, last, retryable=False) from exc
                if exc.code == 429:
                    # 限流：记下来，别在这层死等，交给上层降级
                    raise SourceError(source, "被限流（429）", retryable=True) from exc
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last = f"{type(exc).__name__}: {exc}"
            if i + 1 < self.tries:
                time.sleep(1.0 + i)
        raise SourceError(source, f"请求失败（{last}）", retryable=True)

    def get_json(self, url: str, *, source: str) -> tuple[Any, bool]:
        res = self.get_text(url, source=source)
        if isinstance(res.payload, (dict, list)):
            return res.payload, res.cached
        try:
            data = json.loads(res.payload)
        except (TypeError, json.JSONDecodeError) as exc:
            raise SourceError(source, f"返回不是合法 JSON：{exc}", retryable=False) from exc
        self.cache.put(url, data)
        return data, res.cached


def join_query(params: dict[str, Any]) -> str:
    return urllib.parse.urlencode({k: v for k, v in params.items() if v not in (None, "")})
