"""从对话文本里抽能力证据。

四条硬约束（对应「解释性必须好」）：

1. **只认原话**：``quote`` 保存用户说的完整短句，不改写、不摘要——他要能一眼认出这是自己说的。
2. **不跨短句**：匹配只在逗号/顿号/冒号切开的短句内进行。整句匹配会把
   「没用过命令行」记到深度学习头上，引文与结论对不上。
3. **否定优先**：``没写过类`` 不能被 ``写过类`` 这条规则吃成正面证据。
4. **含糊降级**：``还行 / 一般 / 大概`` 只算弱证据，单独一条不给维度定档。

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


def split_clauses(sentence: str) -> list[str]:
    """按逗号/顿号/冒号切短句。匹配只在短句内进行，避免跨短句编出证据。"""
    return [c.strip() for c in CLAUSE_SPLIT.split(sentence or "") if c.strip()]


def _negated(sentence: str, start: int) -> bool:
    window = sentence[max(0, start - NEGATION_WINDOW):start]
    return any(tok in window for tok in NEGATION)


def _strength(sentence: str) -> str:
    return "weak" if any(v in sentence for v in VAGUE) else "strong"


def _clause_at(sentence: str, pos: int) -> str:
    """取匹配所在的短句，保证引文是用户原话的一部分，不改写。

    保留给需要「整句上下文 + 短句引文」的调用方；``extract`` 已改为短句内匹配。
    """
    start = 0
    for m in CLAUSE_SPLIT.finditer(sentence):
        if start <= pos < m.start():
            clause = sentence[start:m.start()].strip()
            return clause or sentence.strip()
        start = m.end()
    clause = sentence[start:].strip()
    return clause or sentence.strip()


def extract(text: str, rules: list[dict] | None = None) -> list[Signal]:
    """把一段对话拆成证据列表。

    匹配单位是**短句**而不是整句。整句匹配会让
    ``没用过.{0,8}torch`` 从「没用过命令行」起头，命中「…也没碰过torch」，
    于是深度学习维度拿到一条引文是「没用过命令行」的证据——档位也许没错，
    但「为什么这么判」引用错了句子，可解释性当场失效。限定在短句内匹配，
    引文就一定说的是这个维度。
    """
    rules = rules if rules is not None else load_rules()
    # 长模式更具体，先匹配，减少「写过脚本」把「写过类」吃掉这类情况
    ordered = sorted(rules, key=lambda r: -max(len(p) for p in r.get("patterns", [""])))

    signals: list[Signal] = []
    seen: set[tuple[str, str]] = set()

    for sentence in split_sentences(text):
        for clause in split_clauses(sentence):
            # 同一短句同一维度只留最有力的一条：先比档位，再比匹配长度
            best: dict[Dimension, tuple[tuple[int, int], Signal]] = {}
            for rule in ordered:
                for pattern in rule.get("patterns", []):
                    m = re.search(pattern, clause)
                    if not m:
                        continue
                    polarity = rule.get("polarity", "positive")
                    if polarity == "positive" and _negated(clause, m.start()):
                        continue
                    sig = Signal(
                        rule_id=rule["id"],
                        dimension=Dimension(rule["dimension"]),
                        level=Level(int(rule["level"])),
                        strength=_strength(clause),
                        quote=clause,
                        why=rule.get("why", ""),
                        polarity=polarity,
                    )
                    rank = (int(sig.level), m.end() - m.start())
                    current = best.get(sig.dimension)
                    if current is None or rank > current[0]:
                        best[sig.dimension] = (rank, sig)
                    break  # 一条规则在同一短句里只算一次

            for _, sig in best.values():
                key = (sig.rule_id, sig.quote)
                if key in seen:
                    continue
                seen.add(key)
                signals.append(sig)

    return signals
