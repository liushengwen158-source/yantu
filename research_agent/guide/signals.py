"""把用户原话变成证据。

两条硬规则（都是踩过坑之后加的）：

1. **按短句匹配，不跨短句**。
   ``没(碰过|用过).{0,8}torch`` 这种规则在整句上跑，会把
   「我没用过命令行，也没碰过 torch」里的「没用过命令行」算成深度学习的证据。
   所以先按标点切短句，每条规则只在单个短句内匹配。

2. **同一短句里出现负向证据，正向证据作废**。
   「没用过命令行」必然同时命中「用过命令行」，不能让两者都成立；
   负向优先，因为否定词是说话人真实的意思。

规则库在 ``knowledge/guide_signals.json``：改文案不用改代码。
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

from .spec import Dimension, Level, Signal, Source

KNOWLEDGE_DIR = Path(__file__).resolve().parent.parent / "knowledge"
SIGNAL_PATH = KNOWLEDGE_DIR / "guide_signals.json"

# 只切分句与逗号，不切「的」「了」这类词内字
CLAUSE_SPLIT = re.compile(r"[。！？!?；;\n，,、：:]")


def split_clauses(text: str) -> list[str]:
    return [c.strip() for c in CLAUSE_SPLIT.split(text) if c and c.strip()]


@lru_cache(maxsize=1)
def load_rules(path: str | None = None) -> tuple[dict[str, Any], ...]:
    """读规则库并预编译正则。返回元组以便缓存。"""
    target = Path(path) if path else SIGNAL_PATH
    if not target.exists():
        return ()
    raw = json.loads(target.read_text(encoding="utf-8"))
    rules: list[dict[str, Any]] = []
    for item in raw.get("rules", []):
        try:
            dimension = Dimension(item["dimension"])
        except (KeyError, ValueError):
            continue
        try:
            level = Level(int(item["level"]))
        except (KeyError, ValueError):
            continue
        patterns = item.get("patterns") or []
        if not patterns:
            continue
        try:
            compiled = [re.compile(p, re.IGNORECASE) for p in patterns]
        except re.error:
            # 单条规则写错不该拖垮整个评估
            continue
        rules.append(
            {
                "rule_id": item.get("id", ""),
                "dimension": dimension,
                "level": level,
                "strength": item.get("strength", "strong"),
                "polarity": item.get("polarity", "positive"),
                "why": item.get("why", ""),
                "compiled": compiled,
            }
        )
    return tuple(rules)


def match_signals(
    text: str,
    rules: Iterable[dict[str, Any]] | None = None,
    dimensions: Iterable[Dimension] | None = None,
) -> list[Signal]:
    """从文本里抽证据。

    ``dimensions`` 给定时只保留这些维度的证据——方向未定时不需要
    去匹配「深度学习」的规则，省得后面还要丢掉。
    """
    if not text or not text.strip():
        return []
    rule_list = list(rules) if rules is not None else list(load_rules())
    if dimensions is not None:
        allowed = set(dimensions)
        rule_list = [r for r in rule_list if r["dimension"] in allowed]

    best: dict[str, Signal] = {}
    for clause in split_clauses(text):
        # 中英文之间常带空格（「写过 SQL」），匹配时去掉空白，引文仍用原文
        compact = re.sub(r"\s+", "", clause)
        hits: list[tuple[dict[str, Any], re.Match[str]]] = []
        for rule in rule_list:
            for pattern in rule["compiled"]:
                found = pattern.search(compact) or pattern.search(clause)
                if found:
                    hits.append((rule, found))
                    break

        # 同一短句里负向优先：否定词才是说话人真实的意思
        if any(rule["polarity"] == "negative" for rule, _ in hits):
            hits = [(r, m) for r, m in hits if r["polarity"] == "negative"]

        for rule, found in hits:
            # 引文取原文短句：compact 去掉了空格，不适合当证据展示
            quote = clause.strip() or found.group(0).strip()
            signal = Signal(
                rule_id=rule["rule_id"],
                dimension=rule["dimension"],
                level=rule["level"],
                strength=rule["strength"],
                quote=quote,
                why=rule["why"],
                polarity=rule["polarity"],
                source=Source.SELF,
            )
            prev = best.get(signal.rule_id)
            # 同一条规则命中多次时留档位更高的那次（同一段话里更强的表述）
            if prev is None or int(signal.level) > int(prev.level):
                best[signal.rule_id] = signal
    return list(best.values())


def rule_count() -> int:
    return len(load_rules())
