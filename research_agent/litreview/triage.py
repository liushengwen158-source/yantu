"""阅读建议：把"相关度 + 年份 + 被引 + 是不是综述"折算成四档，并说清理由。

刻意不用"重要性 8.5 分"这种没法核对的数字。每一条结论都由 ``signals`` 推出来，
理由里直接写出用的是哪个信号、取值多少——用户能反驳它：

    现在精读 · 综述：一次覆盖该问题的主要路线（相关度 0.72，被引 180）

四档的判定顺序（先命中先返回，所以顺序本身是设计）：

1. 相关度 < 0.25            → 不必读（先保证不错杀时间）
2. 综述且相关度 ≥ 0.4        → 现在精读（新手最好的入口是综述，不是最热的单篇）
3. 高被引且相关度 ≥ 0.4      → 现在精读（奠基工作，绕不过去）
4. 很新且相关度 ≥ 0.6        → 快速略读（还没被引用验证，先知道结论）
5. 既旧又少被引              → 不必读（1985 年的 0 引用文章，即使在词面上对得上）
6. 相关度 ≥ 0.6            → 现在精读
7. 其余                     → 先搁置

"先搁置"和"不必读"的区别很重要：前者是"该读但现在读不动/读不起"，
后者是"跟你没关系"。混成一个"低分"会让用户错过真正的奠基工作。
"""

from __future__ import annotations

import math
import time

from .identity import key_of
from .search import relevance
from .spec import PaperRef, Triage, Verdict

# 这四个信号就是全部输入，新增信号要同时更新 reason 文案，否则用户没法核对
SIGNAL_KEYS = ("relevance", "recency", "impact", "survey")


def _recency(year: int | None, now_year: int, half_life: float = 6.0) -> float:
    """半衰期 6 年：6 年前的算 0.5，12 年前 0.25。领域不同应可调，但先给一个诚实的默认。"""
    if not year:
        return 0.3
    age = max(0, now_year - year)
    return round(0.5 ** (age / half_life), 3)


def _impact(cited: int, max_cited: int = 0, known: bool = True) -> float:
    """绝对标尺，不用批内最大值归一。

    两次修正的痕迹都留着，因为都是踩过的坑：

    1. 早期用 ``log(1+被引)/log(1+批内最大被引)``，结果是**批内第一名永远是 1.0**——
       一篇只有 1 次引用的新论文，在"只有它自己"的批次里拿到满分，被判"奠基工作"。
       改成绝对标尺：1 万次被引 = 1.0，100 次 ≈ 0.5，10 次 ≈ 0.25，跨批次可比，
       阈值也能直接写给人看。
    2. **未知被引不能当 0**。arXiv 的 API 根本不返回被引数，当成 0 会让所有预印本
       被系统性地排到书章节后面（实测：问 RAG，前 8 条全是 2026 年的 CRC 书章节，
       真正该读的 arXiv 论文一条都没进）。未知就给中性值 0.4，并在理由里写明"未知"。
    """
    if not known:
        return 0.4
    if cited <= 0:
        return 0.0
    return round(min(1.0, math.log10(1 + cited) / math.log10(1 + 10000)), 3)


def _prereq_of(paper: PaperRef, concepts: list[dict] | None) -> list[str]:
    """概念命中：标题/摘要里出现了概念库里的词 → 认为读它之前需要这个概念。

    这是**弱推断**：断言"论文用到某概念"和"读者需要先懂某概念"不是一回事。
    所以只作为提示列出来，不参与判定，也不写进理由。
    """
    if not concepts:
        return []
    haystack = f"{paper.title} {paper.abstract}".lower()
    if not haystack.strip():
        return []
    hits: list[str] = []
    for c in concepts:
        names = [c.get("name", "")] + list(c.get("aliases") or [])
        for n in names:
            n = (n or "").strip().lower()
            if len(n) >= 3 and n in haystack:
                hits.append(c.get("name") or c.get("id", ""))
                break
        if len(hits) >= 4:
            break
    return hits


def triage(
    papers: list[PaperRef],
    question: str = "",
    *,
    concepts: list[dict] | None = None,
    now_year: int | None = None,
) -> list[Triage]:
    """给一批文献打四档建议，按建议顺序 + 分数排序返回。"""
    now_year = now_year or int(time.strftime("%Y"))
    max_cited = max([p.cited_by or 0 for p in papers], default=0)

    out: list[Triage] = []
    for p in papers:
        rel = relevance(question, p) if question else 0.5
        rec = _recency(p.year, now_year)
        imp = _impact(p.cited_by or 0, max_cited, known=p.cited_known)
        sur = 1.0 if p.is_survey else 0.0
        score = 0.45 * rel + 0.25 * imp + 0.20 * rec + 0.10 * sur

        signals = {"relevance": rel, "recency": rec, "impact": imp, "survey": sur}
        verdict, reason = _decide(p, signals, rel, rec, imp, sur)
        out.append(Triage(
            key=key_of(p), paper=p, verdict=verdict, score=score, signals=signals,
            reason=reason,
            read_for=_read_for(p, question),
            prereq=_prereq_of(p, concepts),
        ))

    out.sort(key=lambda t: (t.verdict.order, -t.score))
    return out


def _decide(
    p: PaperRef, signals: dict[str, float], rel: float, rec: float, imp: float, sur: float
) -> tuple[Verdict, str]:
    cited = f"被引 {p.cited_by}" if p.cited_known else "被引数未知（该源不提供）"
    year = p.year or "—"
    if rel < 0.25:
        return Verdict.DROP, f"相关度偏低（{rel:.2f}），与本问题关系不大"
    if sur >= 1.0 and rel >= 0.4:
        return Verdict.READ_NOW, f"综述：一次覆盖该问题的主要路线（相关度 {rel:.2f}，{cited}）"
    if imp >= 0.75 and rel >= 0.4:
        return Verdict.READ_NOW, f"奠基工作：{cited}，且与问题直接相关（相关度 {rel:.2f}）"
    if rel >= 0.6 and rec >= 0.6:
        return Verdict.SKIM, f"很新（{year}），还没有足够引用验证，先扫结论（相关度 {rel:.2f}，{cited}）"
    if imp < 0.2 and rec < 0.3:
        return Verdict.DROP, f"既旧（{year}）又少被引（{cited}），且相关度一般（{rel:.2f}）"
    if rel >= 0.6:
        return Verdict.READ_NOW, f"与问题高度相关（相关度 {rel:.2f}，{cited}）"
    return Verdict.PARK, f"相关但信号偏弱（相关度 {rel:.2f}，{cited}，{year}），需要时再读"


def _read_for(p: PaperRef, question: str) -> str:
    """读它要解决什么问题。没有问题时就不编，只给它的定位。"""
    if not p.title:
        return ""
    if p.is_survey:
        return "先建立这个方向的整体地图：有哪些路线、各自卡在哪"
    if question:
        return f"看它怎么处理「{question.strip()[:40]}」，以及它的实验设定能不能对上你的场景"
    return "看它提出的方法与自己声称解决的问题是否对得上"
