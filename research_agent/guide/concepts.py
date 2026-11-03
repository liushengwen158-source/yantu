"""概念解释：查得到就给卡，查不到就直说没有。

概念库（``knowledge/concepts.json``）是唯一事实来源，不让模型自由生成定义——
新手分不清「模型编的」和「教材写的」，而这个模块的价值就是可信。

每张卡固定五段：一句话 / 类比 / 常见误解 / 前置 / 下一步，外加一道自检题。
检索顺序：id → 名称 → 别名 → 名称包含 → 模糊（给出候选，不硬答）。
"""
from __future__ import annotations

import difflib
import json
import re
from pathlib import Path
from typing import Any

KB_PATH = Path(__file__).resolve().parent.parent / "knowledge" / "concepts.json"

# 提问时常见的赘余词，去掉后再匹配
NOISE = re.compile(r"[？?。，,\s]|是什么|啥意思|什么意思|怎么理解|教我|介绍一下|讲讲|请问")

FUZZY_THRESHOLD = 0.45


def load_concepts(path: Path | None = None) -> list[dict[str, Any]]:
    p = path or KB_PATH
    data = json.loads(p.read_text(encoding="utf-8"))
    return list(data.get("concepts", []))


def _normalize(text: str) -> str:
    return NOISE.sub("", (text or "").strip()).lower()


def _index(concepts: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for card in concepts:
        out[card["id"]] = card
    return out


def get(concept_id: str) -> dict[str, Any] | None:
    """按 id 取卡，取不到返回 None（不抛异常，调用方好写）。"""
    return _index(load_concepts()).get((concept_id or "").strip())


def _brief(card: dict[str, Any]) -> dict[str, str]:
    return {
        "id": card["id"],
        "name": card["name"],
        "one_liner": card.get("one_liner", ""),
        "dimension": card.get("dimension", ""),
        "stage": card.get("stage", ""),
    }


def _fuzzy(term: str, concepts: list[dict[str, Any]], limit: int = 3) -> list[dict[str, Any]]:
    scored: list[tuple[float, dict[str, Any]]] = []
    for card in concepts:
        names = [card["name"], card["id"], *card.get("aliases", [])]
        best = max(
            difflib.SequenceMatcher(None, term, _normalize(n)).ratio() for n in names if n
        )
        if best >= FUZZY_THRESHOLD:
            scored.append((best, card))
    scored.sort(key=lambda item: -item[0])
    return [card for _, card in scored[:limit]]


def explain(term: str, dimension: str | None = None) -> dict[str, Any]:
    """查一个概念。

    返回 ``found`` / ``match`` / ``card`` / ``candidates``；
    没命中时 ``card`` 为 None，``candidates`` 是最近邻，仍然可以引导用户。
    """
    raw = (term or "").strip()
    if not raw:
        return {"found": False, "match": None, "card": None, "candidates": [], "term": raw}

    concepts = load_concepts()
    if dimension:
        filtered = [c for c in concepts if c.get("dimension") == dimension]
        concepts = filtered or concepts

    key = _normalize(raw)

    for card in concepts:
        if key == _normalize(card["id"]) or key == _normalize(card["name"]):
            return {"found": True, "match": "exact", "card": card, "candidates": [], "term": raw}

    for card in concepts:
        if any(key == _normalize(alias) for alias in card.get("aliases", [])):
            return {"found": True, "match": "alias", "card": card, "candidates": [], "term": raw}

    # 「torch.cuda.is_available() 是 false」这类问句：别名出现在句子里也算命中。
    # 太短的别名（lr、cv）容易误伤，所以英文要求 ≥3 字符、中文 ≥2 字。
    alias_hits: list[tuple[int, dict[str, Any]]] = []
    for card in concepts:
        for alias in card.get("aliases", []):
            a = _normalize(alias)
            if not a:
                continue
            min_len = 3 if a.isascii() else 2
            if len(a) >= min_len and a in key:
                alias_hits.append((len(a), card))
                break
    if alias_hits:
        alias_hits.sort(key=lambda item: -item[0])
        best = alias_hits[0][1]
        others = [_brief(c) for _, c in alias_hits[1:] if c["id"] != best["id"]]
        return {
            "found": True,
            "match": "alias_in_text",
            "card": best,
            "candidates": others[:3],
            "term": raw,
        }

    # 「反向传播和梯度下降的关系」这类长问句，按包含匹配
    containing = [
        card
        for card in concepts
        if len(key) >= 2 and (key in _normalize(card["name"]) or _normalize(card["name"]) in key)
    ]
    if containing:
        containing.sort(key=lambda c: len(_normalize(c["name"])), reverse=True)
        best = containing[0]
        others = [_brief(c) for c in containing[1:4]]
        return {
            "found": True,
            "match": "contains",
            "card": best,
            "candidates": others,
            "term": raw,
        }

    nearest = _fuzzy(key, concepts)
    return {
        "found": False,
        "match": None,
        "card": None,
        "candidates": [_brief(c) for c in nearest],
        "term": raw,
    }


def related(concept_id: str) -> dict[str, list[dict[str, str]]]:
    """前置与下一步，按 id 换成可读卡片摘要；库里没有的 id 直接略过。"""
    card = get(concept_id)
    if card is None:
        return {"prereq": [], "next": []}
    index = _index(load_concepts())

    def resolve(ids: list[str]) -> list[dict[str, str]]:
        return [_brief(index[i]) for i in ids if i in index]

    return {"prereq": resolve(card.get("prereq", [])), "next": resolve(card.get("next", []))}


def list_concepts(dimension: str | None = None) -> list[dict[str, str]]:
    cards = load_concepts()
    if dimension:
        cards = [c for c in cards if c.get("dimension") == dimension]
    return [_brief(c) for c in cards]


def render_card(card: dict[str, Any]) -> str:
    """人类可读的卡片。缺字段就跳过，不编内容。"""
    links = related(card["id"])
    lines = [f"## {card['name']}", ""]
    aliases = [a for a in card.get("aliases", []) if _normalize(a) != _normalize(card["name"])]
    if aliases:
        lines.append(f"也叫：{'、'.join(aliases)}")
        lines.append("")
    if card.get("one_liner"):
        lines.append(f"一句话：{card['one_liner']}")
    if card.get("analogy"):
        lines.append(f"类比：{card['analogy']}")
    if card.get("misconception"):
        lines.append(f"常见误解：{card['misconception']}")
    if links["prereq"]:
        lines.append(f"前置：{'、'.join(c['name'] for c in links['prereq'])}")
    if links["next"]:
        lines.append(f"下一步：{'、'.join(c['name'] for c in links['next'])}")
    if card.get("check"):
        lines.append(f"自检：{card['check']}")
    lines.append("")
    lines.append("（来自项目概念库 knowledge/concepts.json）")
    return "\n".join(lines)


def render_miss(result: dict[str, Any]) -> str:
    """查不到时的回答：直说没有，再给最近邻，不硬凑。"""
    lines = [f"概念库里没有「{result.get('term', '')}」这张卡。"]
    if result.get("candidates"):
        lines.append("比较接近的有：")
        for item in result["candidates"]:
            lines.append(f"  - {item['name']}（{item['id']}）：{item['one_liner']}")
    else:
        lines.append("也没有足够接近的条目。可以换个说法，或者告诉我上下文，我按你的场景解释。")
    return "\n".join(lines)


def validate() -> list[str]:
    """检查前置/下一步里有没有指向不存在 id 的情况，给测试用。"""
    cards = load_concepts()
    ids = {c["id"] for c in cards}
    problems: list[str] = []
    for card in cards:
        for field in ("prereq", "next"):
            for ref in card.get(field, []):
                if ref not in ids:
                    problems.append(f"{card['id']}.{field} 指向不存在的概念 {ref}")
    return problems
