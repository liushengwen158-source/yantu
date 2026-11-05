"""供 agent 调用的文献工具。

和 ``repro_tools`` / ``env_tools`` 一样的四条纪律：

1. 返回 JSON 可序列化的 dict，**不抛异常**；出错时把原因和 ``hint`` 写进返回值，
   让模型能读懂并转述，而不是崩在工具层。
2. 结构化数据只保留可引用的字段，长文本单独放 ``text``，并且有长度上限。
3. 工具之间不互相调用，全部走 ``search`` / ``graph`` / ``triage`` 这几个纯函数。
4. 不返回"我猜的结论"——只返回检索结果和建议，判断由模型结合上下文做。

工具粒度刻意做细（search / triage / expand 分开），因为合成一个"帮我调研这个方向"
的大工具，模型就会在一次调用里顺手把答案编出来——这正是要防的。
"""

from __future__ import annotations

from pathlib import Path

from ..litreview import expand as expand_graph
from ..litreview import search as run_search
from ..litreview import triage as run_triage
from ..litreview.cache import Cache
from ..litreview.render import (
    concepts_svg,
    coverage_line,
    html_page,
    legend,
    markdown_report,
    papers_svg,
)
from ..litreview.spec import PaperRef, Source, Verdict

_MAX_TEXT = 6000


def _fail(message: str, hint: str = "") -> dict:
    return {"ok": False, "error": message, "hint": hint}


def _paper_brief(p: PaperRef) -> dict:
    return {
        "title": p.title,
        "year": p.year,
        "venue": p.venue,
        "authors": p.authors[:5],
        "source": p.source.value,
        "url": p.url,
        "doi": p.doi,
        "cited_by": p.cited_by,
        "cited_known": p.cited_known,
        "is_survey": p.is_survey,
    }


def lit_search(
    question: str,
    limit: int = 15,
    year_from: int = 0,
    offline: bool = False,
) -> dict:
    """检索文献，返回候选列表、检索式与覆盖度。

    Args:
        question: 研究问题或关键词，中英文都可以
        limit: 返回条数上限，默认 15
        year_from: 只要这一年及以后的文献，0 表示不限
        offline: 只用本地缓存，不发网络请求（做可复现演示时用）
    """
    if not question or not question.strip():
        return _fail("缺少检索问题", "给一句具体的问题，例如「扩散策略 机器人操作」")
    retrieval = run_search(
        question.strip(),
        limit=max(1, min(limit, 50)),
        year_from=year_from or None,
        cache=Cache(offline=offline),
        offline=offline,
    )
    return {
        "ok": bool(retrieval.papers),
        "question": retrieval.question,
        "queries": retrieval.queries,
        "coverage": retrieval.coverage.to_dict(),
        "coverage_line": coverage_line(retrieval),
        "count": len(retrieval.papers),
        "dropped": retrieval.dropped,
        "papers": [_paper_brief(p) for p in retrieval.papers],
        "warnings": retrieval.warnings,
        "next": "需要优先读哪几篇用 lit_triage；需要看一个方向的脉络用 lit_expand（要先有一篇种子论文）",
        "note": "只返回检索到的元数据。来源失败的已写进 coverage，不要当成'该领域没有这类工作'。",
    }


def lit_triage(question: str, limit: int = 15, year_from: int = 0, offline: bool = False) -> dict:
    """检索并给出四档阅读建议（现在精读 / 快速略读 / 先搁置 / 不必读）。

    Args:
        question: 研究问题
        limit: 参与评估的文献数，默认 15
        year_from: 年份下限，0 表示不限
        offline: 只用缓存
    """
    if not question or not question.strip():
        return _fail("缺少检索问题", "例如「视觉语言模型 机器人抓取」")
    retrieval = run_search(question.strip(), limit=max(1, min(limit, 30)),
                           year_from=year_from or None,
                           cache=Cache(offline=offline), offline=offline)
    if not retrieval.papers:
        return {
            "ok": False,
            "error": "没有可用文献，无法给出建议",
            "coverage_line": coverage_line(retrieval),
            "warnings": retrieval.warnings,
        }
    triaged = run_triage(retrieval.papers, retrieval.question)
    items = []
    for t in triaged:
        row = t.to_dict()
        row["authors"] = t.paper.authors[:3]
        row["venue"] = t.paper.venue
        items.append(row)
    counts = {v.value: sum(1 for t in triaged if t.verdict is v) for v in Verdict}
    return {
        "ok": True,
        "question": retrieval.question,
        "coverage_line": coverage_line(retrieval),
        "counts": counts,
        "papers": items,
        "how_to_read": "verdict=read_now 的先读；park 的说明'重要但现在读不动'，不要当成不相关",
        "caveat": "建议由相关度/年份/被引/是否综述四个信号推出，signals 字段可逐项核对",
    }


def lit_expand(title: str, doi: str = "", limit: int = 20, offline: bool = False) -> dict:
    """以一篇论文为种子，拉出它的引用关系图（上游引用 / 下游被引 / 近邻 / 文献耦合）。

    Args:
        title: 种子论文标题（用于匹配 OpenAlex 记录）
        doi: 种子论文 DOI，有就优先用；没有就靠标题匹配
        limit: 节点数上限提示
        offline: 只用缓存
    """
    if not title and not doi:
        return _fail("缺少种子论文", "给一个标题或 DOI")
    seed = PaperRef(
        source=Source.OPENALEX if doi else Source.CROSSREF,
        source_id=doi or title[:80],
        title=title or doi,
        doi=doi,
    )
    graph = expand_graph(seed, cache=Cache(offline=offline), offline=offline)
    if not graph.nodes:
        return {
            "ok": False,
            "error": "没能建立关系图",
            "warnings": graph.warnings,
            "hint": "OpenAlex 可能不可用；它需要能访问 api.openalex.org。也可以先不画图，只用 lit_search",
        }
    deg = graph.degree()
    top = sorted(graph.nodes.items(), key=lambda kv: -deg.get(kv[0], 0))[: limit]
    return {
        "ok": True,
        "seed": graph.seed,
        "node_count": len(graph.nodes),
        "edge_count": len(graph.edges),
        "isolated": sum(1 for d in deg.values() if d == 0),
        "edges_by_relation": _count_relations(graph.edges),
        "nodes": [
            {
                "key": k,
                "title": p.short(70),
                "year": p.year,
                "cited_by": p.cited_by,
                "degree": deg.get(k, 0),
            }
            for k, p in top
        ],
        "warnings": graph.warnings,
        "relation_legend": {
            "cites": "种子引用了它（地基）",
            "cited_by": "它引用了种子（后续）",
            "similar": "站方推荐相似，不等于因果",
            "coupling": "与另一篇共享 ≥2 篇参考文献，算同一条技术路线",
        },
    }


def _count_relations(edges) -> dict:
    out: dict[str, int] = {}
    for e in edges:
        out[e.relation.value] = out.get(e.relation.value, 0) + 1
    return out


def lit_map(question: str, limit: int = 15, year_from: int = 0, out: str = "",
            offline: bool = False) -> dict:
    """生成可离线打开的文献关系图 HTML（单文件，无外部依赖）。

    Args:
        question: 研究问题
        limit: 参与绘图的文献数
        year_from: 年份下限
        out: 输出路径，默认写到当前目录 lit-map.html
        offline: 只用本地缓存
    """
    retrieval = run_search(question.strip(), limit=max(1, min(limit, 40)),
                           year_from=year_from or None,
                           cache=Cache(offline=offline), offline=offline)
    if not retrieval.papers:
        return _fail("没有可用文献", coverage_line(retrieval))
    triaged = run_triage(retrieval.papers, retrieval.question)
    keyed = [(t.key, t.paper) for t in triaged]
    svg = papers_svg(keyed, question=retrieval.question)
    leg = legend(
        [(f"{v.label}", c) for v, c in
         ((Verdict.READ_NOW, "#3fb950"), (Verdict.SKIM, "#58a6ff"),
          (Verdict.PARK, "#d29922"), (Verdict.DROP, "#6e7681"))],
        tail="圆 = 一篇文献，越大越常被引；横轴 = 发表年份",
    )
    body = (
        f'<div class="facts">检索式：{retrieval.queries[0]}　覆盖度：{coverage_line(retrieval)}</div>'
        + svg + leg
    )
    page = html_page("研途 · 文献关系图", retrieval.question, [("论文关系图", body)])
    target = Path(out) if out else Path("lit-map.html")
    target.write_text(page, encoding="utf-8")
    return {"ok": True, "path": str(target), "count": len(keyed),
            "note": "双击用浏览器打开；不联网也能看，CSS/JS 全部内联"}


def lit_concept_map(concepts_path: str = "", out: str = "") -> dict:
    """画概念前置图（需要 learning-guide 分支的概念库）。

    Args:
        concepts_path: concepts.json 的路径，默认找 research_agent/knowledge/concepts.json
        out: 输出 HTML 路径，默认 concept-map.html
    """
    import json

    candidates = [Path(concepts_path)] if concepts_path else [
        Path("research_agent/knowledge/concepts.json"),
        Path("knowledge/concepts.json"),
    ]
    target_json = next((p for p in candidates if p.exists()), None)
    if target_json is None:
        return _fail(
            "找不到概念库 concepts.json",
            "这个文件在 learning-guide 分支：git show origin/learning-guide:research_agent/knowledge/concepts.json > research_agent/knowledge/concepts.json",
        )
    from ..litreview.render import concept_stats

    data = json.loads(target_json.read_text(encoding="utf-8"))
    concepts = data.get("concepts") or []
    stats = concept_stats(concepts)
    svg = concepts_svg(concepts)
    cyc = "；".join(" ↔ ".join(c) for c in stats["cycles"]) or "无"
    body = (
        f'<div class="facts">概念 {stats["concepts"]} 个 / 前置边 {stats["edges"]} 条 / '
        f'根节点 {stats["roots"]} 个 / 跨维度依赖 {stats["cross_dimension"]} 条 / 环形依赖：{cyc}</div>'
        + svg
        + legend([(d, c) for d, c in {"tools": "#5b8ff9", "coding": "#61ddaa", "math": "#f6bd16",
                                      "ml": "#7262fd", "dl": "#78d3f8", "research": "#9661bc",
                                      "systems": "#f6903d", "data": "#008685",
                                      "theory": "#f08bb4", "security": "#e8684a"}.items()],
                 tail="箭头 = 前置：先学 A 才能学 B")
    )
    page = html_page("研途 · 概念前置图", f"数据源：{target_json}", [("概念图", body)])
    target = Path(out) if out else Path("concept-map.html")
    target.write_text(page, encoding="utf-8")
    return {"ok": True, "path": str(target), "stats": stats}


def lit_report(question: str, limit: int = 15, year_from: int = 0, out: str = "") -> dict:
    """生成 Markdown 文献速览报告（含检索式、覆盖度、四档建议）。

    Args:
        question: 研究问题
        limit: 文献数
        year_from: 年份下限
        out: 输出路径，默认 lit-report.md
    """
    retrieval = run_search(question.strip(), limit=max(1, min(limit, 30)),
                           year_from=year_from or None)
    triaged = run_triage(retrieval.papers, retrieval.question) if retrieval.papers else []
    text = markdown_report(retrieval, triaged)
    target = Path(out) if out else Path("lit-report.md")
    target.write_text(text, encoding="utf-8")
    return {
        "ok": True,
        "path": str(target),
        "count": len(retrieval.papers),
        "text": text[:_MAX_TEXT],
        "truncated": len(text) > _MAX_TEXT,
    }


LIT_TOOL_FUNCTIONS = {
    "lit_search": lit_search,
    "lit_triage": lit_triage,
    "lit_expand": lit_expand,
    "lit_map": lit_map,
    "lit_report": lit_report,
}
