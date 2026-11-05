"""本地缓存：把每次 HTTP 响应原样存下来。

存在的理由不是"提速"，是三条现实约束（本机 2026-11-05 实测）：

- arXiv ``export.arxiv.org`` 走 http 直接超时，https 也时通时断；
- OpenAlex 匿名 **检索** 会被限流（429 + ``retryAfter``），规模化要 key；
- Semantic Scholar 无 key 是共享池，429 是常态；
- Crossref 单条元数据多次 20~45s 超时。

所以：**能离线跑、能重放、测试不联网**是前提条件，不是优化项。缓存文件是普通
JSON，用户可以直接打开看当时服务端到底返回了什么——这比"相信 agent 的转述"重要。
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def default_root() -> Path:
    """缓存根目录。``YANTU_CACHE`` 可覆盖；默认放在项目下的 .cache。"""
    env = os.environ.get("YANTU_CACHE", "")
    if env:
        return Path(env)
    return Path.cwd() / ".cache" / "litreview"


@dataclass
class CacheEntry:
    key: str
    url: str
    stored_at: float
    payload: Any

    @property
    def age_days(self) -> float:
        return (time.time() - self.stored_at) / 86400


class Cache:
    """对象存储很小，用文件即可，不引入 sqlite 之外的任何依赖。"""

    def __init__(self, root: Path | None = None, ttl_days: float = 7.0, offline: bool = False):
        self.root = Path(root) if root else default_root()
        self.ttl_days = ttl_days
        self.offline = offline
        self.hits = 0
        self.misses = 0

    # -- 基本读写 ---------------------------------------------------------
    @staticmethod
    def key_for(url: str) -> str:
        return hashlib.sha1(url.encode("utf-8")).hexdigest()[:20]

    def _path(self, key: str) -> Path:
        return self.root / f"{key}.json"

    def get(self, url: str) -> Any | None:
        p = self._path(self.key_for(url))
        if not p.exists():
            self.misses += 1
            return None
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            self.misses += 1
            return None
        entry = CacheEntry(
            key=raw.get("key", ""), url=raw.get("url", ""),
            stored_at=float(raw.get("stored_at", 0)), payload=raw.get("payload"),
        )
        if not self.offline and entry.age_days > self.ttl_days:
            self.misses += 1
            return None
        self.hits += 1
        return entry.payload

    def put(self, url: str, payload: Any) -> None:
        if self.offline:
            return
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            key = self.key_for(url)
            body = {"key": key, "url": url, "stored_at": time.time(), "payload": payload}
            self._path(key).write_text(json.dumps(body, ensure_ascii=False), encoding="utf-8")
        except OSError:
            # 缓存写失败不该让整次检索失败——它只是优化
            pass

    def stats(self) -> dict[str, Any]:
        return {"root": str(self.root), "hits": self.hits, "misses": self.misses,
                "ttl_days": self.ttl_days, "offline": self.offline}


class OfflineMiss(RuntimeError):
    """离线模式下没命中缓存。调用方应把它当成"该源不可用"，而不是崩溃。"""
