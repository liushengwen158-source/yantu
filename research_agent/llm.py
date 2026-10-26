"""最小可用的 OpenAI 兼容客户端。

为什么不用 openai 官方 SDK：这个模块只需要一个 POST，用标准库能少一层依赖，
也方便在没装任何三方包的机器上先跑起来。

配置走环境变量，任一组合都可用：
    OPENAI_API_KEY / OPENAI_BASE_URL / OPENAI_MODEL
    DEEPSEEK_API_KEY / DEEPSEEK_BASE_URL / DEEPSEEK_MODEL
    QWEN_API_KEY   / QWEN_BASE_URL  / QWEN_MODEL

没有 Key 时 ``available()`` 返回 False，上层自动降级到规则问答。
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass

DEFAULT_TIMEOUT = 120


@dataclass
class LLMConfig:
    api_key: str
    base_url: str
    model: str
    source: str

    @property
    def endpoint(self) -> str:
        base = self.base_url.rstrip("/")
        if base.endswith("/chat/completions"):
            return base
        return f"{base}/chat/completions"


_PRESETS = {
    "DEEPSEEK": ("https://api.deepseek.com/v1", "deepseek-chat"),
    "QWEN": ("https://dashscope.aliyuncs.com/compatible-mode/v1", "qwen-plus"),
    "OPENAI": ("https://api.openai.com/v1", "gpt-4o-mini"),
}


def load_config() -> LLMConfig | None:
    """按预设顺序找第一个配了 Key 的服务商。"""
    for prefix, (base, model) in _PRESETS.items():
        key = os.environ.get(f"{prefix}_API_KEY", "").strip()
        if not key:
            continue
        return LLMConfig(
            api_key=key,
            base_url=os.environ.get(f"{prefix}_BASE_URL", base).strip(),
            model=os.environ.get(f"{prefix}_MODEL", model).strip(),
            source=prefix.lower(),
        )
    return None


def available() -> bool:
    return load_config() is not None


class LLMError(RuntimeError):
    pass


def chat(messages: list[dict], temperature: float = 0.2,
         max_tokens: int = 1200, timeout: int = DEFAULT_TIMEOUT) -> str:
    """发一次对话请求，返回 assistant 文本。失败抛 LLMError。"""
    cfg = load_config()
    if cfg is None:
        raise LLMError("未配置模型 API Key（可设置 DEEPSEEK_API_KEY 等环境变量）")

    payload = json.dumps({
        "model": cfg.model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }).encode("utf-8")

    req = urllib.request.Request(
        cfg.endpoint,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {cfg.api_key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")[:500]
        raise LLMError(f"模型接口返回 {exc.code}：{body}") from exc
    except urllib.error.URLError as exc:
        raise LLMError(f"连接模型接口失败：{exc.reason}") from exc
    except json.JSONDecodeError as exc:
        raise LLMError(f"模型返回不是合法 JSON：{exc}") from exc

    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMError(f"模型返回结构异常：{str(data)[:300]}") from exc
