"""从对话文本里抽能力证据。

三条硬约束（对应「解释性必须好」）：

1. **只认原话**：``quote`` 保存用户说的完整句子，不改写、不摘要——他要能一眼认出这是自己说的。
2. **否定优先**：``没写过类`` 不能被 ``写过类`` 这条规则吃成正面证据。
3. **含糊降级**：``还行 / 一般 / 大概`` 只算弱证据，单独一条不给维度定档。

规则本体在 ``knowledge/guide_signals.json``，这里只做匹配。
调整判定口径改 JSON，不改代码。
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from .spec import Dimension, Level, Signal

KB_PATH = Path(__file__).resolve().parent.parent / "knowledge" / "guide_signals.json"

SENTENCE_SPLIT = re.compile(r"[。！？!?；;\n]+")
# 引文取「匹配所在的那个短句」，而不是一整句：用户说「我写过爬虫，不过没写过类」时，
# 表格里只显示「不过没写过类」才看得懂，整句会把正反两个证据混在一起。
CLAUSE_SPLIT = re.compile(r"[，,、：:]")

# 匹配位置往前看几个字，判断这句话是不是被否定了
NEGATION = ("不", "没", "未", "无", "别", "非")
NEGATION_WINDOW = 3

# 出现这些词，说明语气是含糊的，只能当弱证据
VAGUE = ("还行", "一般", "一点", "一点点", "大概", "好像", "可能", "似乎", "了解过", "听说过", "略懂")


def load_rules(path: Path | None = None) -> list[dict]:
    """读取对话证据规则库。"""
    p = path or KB_PATH
    data = json.loads(p.read_text(encoding="utf-8"))
    return list(data.get("rules", []))


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in SENTENCE_SPLIT.split(text or "") if s.strip()]


def _negated(sentence: str, start: int) -> bool:
    window = sentence[max(0, start - NEGATION_WINDOW):start]
    return any(tok in window for tok in NEGATION)


def _strength(sentence: str) -> str:
    return "weak" if any(v in sentence for v in VAGUE) else "strong"


def _clause_at(sentence: str, pos: int) -> str:
    """取匹配所在的短句，保证引文是用户原话的一部分，不改写。"""
    start = 0
    for m in CLAUSE_SPLIT.finditer(sentence):
        if start <= pos < m.start():
            clause = sentence[start:m.start()].strip()
            return clause or sentence.strip()
        start = m.end()
    clause = sentence[start:].strip()
    return clause or sentence.strip()


def extract(text: str, rules: list[dict] | None = None) -> list[Signal]:
    """把一段对话拆成证据列表。同一句话同一维度只保留最有力的一条。"""
    rules = rules if rules is not None else load_rules()
    # 长模式更具体，先匹配，减少「写过脚本」把「写过类」吃掉这类情况
    ordered = sorted(rules, key=lambda r: -max(len(p) for p in r.get("patterns", [""])))

    signals: list[Signal] = []
    seen: set[tuple[str, str]] = set()

    for sentence in split_sentences(text):
        # (起, 止, Signal)：同维度里互相重叠的只留一条，不重叠的各自成立
        hits: list[tuple[int, int, Signal]] = []
        for rule in ordered:
            for pattern in rule.get("patterns", []):
                m = re.search(pattern, sentence)
                if not m:
                    continue
                polarity = rule.get("polarity", "positive")
                if polarity == "positive" and _negated(sentence, m.start()):
                    continue
                hits.append(
                    (
                        m.start(),
                        m.end(),
                        Signal(
                            rule_id=rule["id"],
                            dimension=Dimension(rule["dimension"]),
                            level=Level(int(rule["level"])),
                            strength=_strength(sentence),
                            quote=_clause_at(sentence, m.start()),
                            why=rule.get("why", ""),
                            polarity=polarity,
                        ),
                    )
                )
                break  # 一条规则在同一句话里只算一次

        by_dim: dict[Dimension, list[tuple[int, int, Signal]]] = {}
        for start, end, sig in hits:
            by_dim.setdefault(sig.dimension, []).append((start, end, sig))

        for items in by_dim.values():
            # 先让「更高档位」占位，再让「更长的匹配」占位，剩下的视为重复表述
            occupied: list[tuple[int, int]] = []
            for start, end, sig in sorted(items, key=lambda h: (-int(h[2].level), -(h[1] - h[0]))):
                if any(not (end <= a or start >= b) for a, b in occupied):
                    continue
                occupied.append((start, end))
                key = (sig.rule_id, sig.quote)
                if key in seen:
                    continue
                seen.add(key)
                signals.append(sig)

    return signals
