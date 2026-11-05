"""渲染：图 + Markdown 报告 + 单文件 HTML。

三档输出（成本递增，按需取用）：

| 档 | 形态 | 规模 | 用在哪 |
| --- | --- | --- | --- |
| A | 静态 SVG 字符串 | 20~60 节点 | 塞进 Markdown 报告、贴进对话 |
| B | 单文件 HTML（内联 SVG + 极少 vanilla JS，无 CDN） | 200~2000 节点 | 用户本地打开，hover 看详情 |
| C | 结构化 dict | 不限 | 交给别的工具，或以后导 GraphML |

只用标准库。不用 matplotlib / networkx / pyvis：这个项目到目前的零依赖是刻意的，
内网、离线、装不上包的环境都能跑——而"画个图"不值得破这条线。

布局全部是**确定性**的（按年份/阶段分列，不做力导向）：

- 力导向图好看，但读不出"脉络"；新手要的是"哪条线在往前走"。
- 确定性布局让同一份数据每次画出来一模一样，方便对比和复核。
"""

from __future__ import annotations

import html
from collections import defaultdict
from typing import Iterable, Sequence

from .spec import PaperGraph, PaperRef, Relation, Retrieval, Triage, Verdict

DIM_COLOR = {
    "tools": "#5b8ff9", "coding": "#61ddaa", "math": "#f6bd16",
    "ml": "#7262fd", "dl": "#78d3f8", "research": "#9661bc",
    "systems": "#f6903d", "data": "#008685", "theory": "#f08bb4", "security": "#e8684a",
}

REL_COLOR = {
    Relation.CITES: "#5a6b80",
    Relation.CITED_BY: "#4f8f6a",
    Relation.SIMILAR: "#8a6fb0",
    Relation.COUPLING: "#b08a4f",
}

VERDICT_COLOR = {
    Verdict.READ_NOW: "#3fb950",
    Verdict.SKIM: "#58a6ff",
    Verdict.PARK: "#d29922",
    Verdict.DROP: "#6e7681",
}

CSS = """
:root{--bg:#0f1216;--panel:#161b22;--line:#2a323d;--fg:#e6edf3;--dim:#8b949e}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:13px/1.6 "Segoe UI","Microsoft YaHei",sans-serif}
header{padding:18px 24px 10px;border-bottom:1px solid var(--line)}
h1{margin:0 0 4px;font-size:17px;font-weight:600}
.sub{color:var(--dim);font-size:12px}
.tabs{display:flex;gap:6px;padding:10px 24px 0;flex-wrap:wrap}
.tab{padding:7px 14px;border:1px solid var(--line);border-bottom:none;border-radius:7px 7px 0 0;
     background:#11161c;color:var(--dim);cursor:pointer;font-size:12.5px}
.tab.on{background:var(--panel);color:var(--fg);border-color:#3d4a5c}
main{padding:0 24px 28px}
.panel{display:none;background:var(--panel);border:1px solid var(--line);border-radius:0 8px 8px 8px;padding:14px}
.panel.on{display:block}
svg{display:block;width:100%;height:auto}
.node{cursor:pointer}
.node text{pointer-events:none}
.edge{fill:none}
svg.focus .node:not(.hl){opacity:.18}
svg.focus .edge:not(.hl){opacity:.06}
.legend{display:flex;flex-wrap:wrap;gap:14px;padding:10px 2px 0;color:var(--dim);font-size:12px}
.legend i{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:5px;vertical-align:-1px}
.facts{margin:12px 0;padding:10px 12px;border-left:3px solid #d29922;background:#1c1a12;border-radius:4px;font-size:12.5px}
.warn{padding:10px 12px;border-left:3px solid #e8684a;background:#20120f;border-radius:4px;font-size:12.5px;color:#ffb4a2}
#tip{position:fixed;max-width:420px;background:#0d1117;border:1px solid #3d4a5c;border-radius:6px;
     padding:8px 10px;font-size:12px;pointer-events:none;opacity:0;transition:opacity .12s;z-index:9}
#tip b{color:#79c0ff}
"""

JS = """
const tip=document.getElementById('tip');
document.querySelectorAll('.tab').forEach(t=>t.onclick=()=>{
  document.querySelectorAll('.tab').forEach(x=>x.classList.remove('on'));
  document.querySelectorAll('.panel').forEach(x=>x.classList.remove('on'));
  t.classList.add('on');
  document.getElementById(t.dataset.target).classList.add('on');
});
document.querySelectorAll('.panel svg').forEach(svg=>{
  const edges=[...svg.querySelectorAll('.edge')];
  svg.querySelectorAll('.node').forEach(n=>{
    n.addEventListener('mouseenter',()=>{
      const id=n.dataset.id;
      svg.classList.add('focus'); n.classList.add('hl');
      edges.forEach(ed=>{ if(ed.dataset.src===id||ed.dataset.dst===id) ed.classList.add('hl'); });
      tip.innerHTML=n.dataset.tip||''; tip.style.opacity=1;
    });
    n.addEventListener('mousemove',e=>{
      tip.style.left=Math.min(e.clientX+16,innerWidth-440)+'px';
      tip.style.top=(e.clientY+16)+'px';
    });
    n.addEventListener('mouseleave',()=>{
      svg.classList.remove('focus'); n.classList.remove('hl');
      edges.forEach(ed=>ed.classList.remove('hl')); tip.style.opacity=0;
    });
  });
});
"""


def esc(value: str) -> str:
    return html.escape(value or "", quote=True)


def shorten(text: str, width: int) -> str:
    t = " ".join((text or "").split())
    return t if len(t) <= width else t[: width - 1] + "…"


# ---------------------------------------------------------------- 论文图

def papers_svg(
    papers: Sequence[tuple[str, PaperRef]],
    *,
    question: str = "",
    edges: Iterable[tuple[str, str, Relation]] | None = None,
    width: int = 1460,
    row_gap: int = 62,
) -> str:
    """按年份分列的论文图。圆大小 = 被引，颜色 = 阅读建议（有的话）。"""
    nodes = [(k, p) for k, p in papers if p.year]
    if not nodes:
        return '<svg viewBox="0 0 800 120" xmlns="http://www.w3.org/2000/svg"><text x="20" y="60" fill="#8b949e">没有带年份的文献，无法按时间轴布局</text></svg>'

    nodes.sort(key=lambda kp: (kp[1].year, -(kp[1].cited_by or 0)))
    years = sorted({p.year for _, p in nodes})
    col_x = {y: 90 + i * ((width - 180) / max(1, len(years) - 1)) for i, y in enumerate(years)}

    pos: dict[str, tuple[float, float]] = {}
    rows: dict[int, int] = defaultdict(int)
    for k, p in nodes:
        idx = rows[p.year]
        rows[p.year] += 1
        pos[k] = (col_x[p.year], 92 + idx * row_gap)
    height = 170 + max(rows.values()) * row_gap

    max_cited = max([p.cited_by or 0 for _, p in nodes], default=0) or 1
    radius = lambda c: 5 + 11 * (_log10(1 + c) / _log10(1 + max_cited))  # noqa: E731

    parts = [f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">',
             '<defs><marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto">'
             '<path d="M0,0 L10,5 L0,10 z" fill="#5a6b80"/></marker></defs>']

    for y in years:
        x = col_x[y]
        parts.append(f'<line x1="{x:.1f}" y1="60" x2="{x:.1f}" y2="{height - 20}" stroke="#20262e"/>')
        parts.append(f'<text x="{x:.1f}" y="46" fill="#8b949e" font-size="12" text-anchor="middle">{y}</text>')

    by_key = dict(papers)
    for src, dst, rel in edges or []:
        if src not in pos or dst not in pos:
            continue
        x1, y1 = pos[src]
        x2, y2 = pos[dst]
        dx = max(30.0, abs(x2 - x1) * 0.45)
        d = f"M{x1:.1f},{y1:.1f} C{x1 + dx:.1f},{y1:.1f} {x2 - dx:.1f},{y2:.1f} {x2:.1f},{y2:.1f}"
        dash = "" if (rel in (Relation.CITES, Relation.CITED_BY)) else " stroke-dasharray='4 3'"
        arrow = " marker-end='url(#arr)'" if rel is Relation.CITES else ""
        parts.append(
            f'<path class="edge" data-src="{esc(src)}" data-dst="{esc(dst)}" d="{d}" '
            f'stroke="{REL_COLOR.get(rel, "#5a6b80")}" stroke-width="1.1" opacity=".45"{dash}{arrow}/>'
        )

    for k, p in nodes:
        x, y = pos[k]
        r = radius(p.cited_by or 0)
        color = "#2f6fb5" if (p.cited_by or 0) > 0 else "#3d4a5c"
        tip = (f"<b>{esc(p.title)}</b><br>{esc(p.author_text)}<br>"
               f"{esc(p.venue or p.source.label)} · {p.year or '—'} · 被引 {p.cited_by}"
               f"{' · 综述' if p.is_survey else ''}")
        parts.append(
            f'<g class="node" data-id="{esc(k)}" data-tip="{tip}">'
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" fill="{color}" stroke="#79c0ff" stroke-width="1.2"/>'
            f'<text x="{x:.1f}" y="{y + r + 13:.1f}" fill="#c9d1d9" font-size="9.5" text-anchor="middle">'
            f'{esc(shorten(p.title, 22))}</text></g>'
        )
    parts.append("</svg>")
    return "".join(parts)


def _log10(x: float) -> float:
    import math

    return math.log10(x)


# ---------------------------------------------------------------- 概念图

def cycles_of(concepts: Sequence[dict]) -> list[list[str]]:
    """找出前置关系里的环（Tarjan 强连通分量，大小 > 1 即为环）。"""
    ids = [c.get("id", "") for c in concepts if c.get("id")]
    adj: dict[str, list[str]] = defaultdict(list)
    known = set(ids)
    for c in concepts:
        for p in c.get("prereq") or []:
            if p in known and c.get("id") in known:
                adj[p].append(c["id"])

    index: dict[str, int] = {}
    low: dict[str, int] = {}
    on_stack: dict[str, bool] = {}
    stack: list[str] = []
    found: list[list[str]] = []
    counter = [0]

    def dfs(v: str) -> None:
        index[v] = low[v] = counter[0]
        counter[0] += 1
        stack.append(v)
        on_stack[v] = True
        for w in adj.get(v, []):
            if w not in index:
                dfs(w)
                low[v] = min(low[v], low[w])
            elif on_stack.get(w):
                low[v] = min(low[v], index[w])
        if low[v] == index[v]:
            comp = []
            while True:
                w = stack.pop()
                on_stack[w] = False
                comp.append(w)
                if w == v:
                    break
            if len(comp) > 1:
                found.append(comp)

    for v in ids:
        if v not in index:
            dfs(v)
    return found


def concept_stats(concepts: Sequence[dict]) -> dict:
    """概念库体检：规模、根节点、跨维度依赖、环。画图顺手把数据问题报出来。"""
    by_id = {c["id"]: c for c in concepts if c.get("id")}
    edges = 0
    roots = 0
    cross = 0
    for c in concepts:
        prereqs = [p for p in (c.get("prereq") or []) if p in by_id]
        edges += len(prereqs)
        if not prereqs:
            roots += 1
        for p in prereqs:
            if by_id[p].get("dimension") != c.get("dimension"):
                cross += 1
    cyc = cycles_of(concepts)
    return {
        "concepts": len(by_id),
        "edges": edges,
        "roots": roots,
        "cross_dimension": cross,
        "cycles": [[by_id.get(n, {}).get("name", n) for n in comp] for comp in cyc],
    }


def concepts_svg(concepts: Sequence[dict], *, width_per_stage: int = 232) -> str:
    """按学习阶段分列的概念前置图。颜色 = 能力维度，红虚线 = 检测到的环。"""
    by_id = {c["id"]: c for c in concepts if c.get("id")}
    stage_order = ["stage0", "stage1", "stage2", "stage3", "stage4", "stage5", "stage6"]
    stages = [s for s in stage_order if any(c.get("stage") == s for c in concepts)]
    if not stages:
        return '<svg viewBox="0 0 800 120" xmlns="http://www.w3.org/2000/svg"><text x="20" y="60" fill="#8b949e">概念库为空或缺少 stage 字段</text></svg>'

    width = 140 + len(stages) * width_per_stage
    col_x = {s: 90 + i * width_per_stage for i, s in enumerate(stages)}
    node_w, node_h = 178, 23

    pos: dict[str, tuple[float, float]] = {}
    bottom: dict[str, float] = {}
    for s in stages:
        group = [c for c in concepts if c.get("stage") == s and c.get("id")]
        group.sort(key=lambda c: (c.get("dimension", ""), c.get("name", "")))
        y, last_dim = 96, None
        for c in group:
            if last_dim is not None and c.get("dimension") != last_dim:
                y += 12
            pos[c["id"]] = (col_x[s], y)
            y += 33
            last_dim = c.get("dimension")
        bottom[s] = y
    height = max(bottom.values()) + 40

    cyc_nodes = {n for comp in cycles_of(concepts) for n in comp}
    parts = [f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">',
             '<defs><marker id="arr2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto">'
             '<path d="M0,0 L10,5 L0,10 z" fill="#4d5a6b"/></marker></defs>']

    for s in stages:
        x = col_x[s]
        parts.append(f'<line x1="{x - 14}" y1="66" x2="{x - 14}" y2="{height - 18}" stroke="#20262e"/>')
        parts.append(f'<text x="{x - 14}" y="56" fill="#8b949e" font-size="11.5">{esc(s)}</text>')

    for c in concepts:
        cid = c.get("id")
        if cid not in pos:
            continue
        x2, y2 = pos[cid]
        for p in c.get("prereq") or []:
            if p not in pos:
                continue
            x1, y1 = pos[p]
            if abs(x1 - x2) < 1:
                bulge = 46 if y2 > y1 else -46
                d = (f"M{x1 + node_w / 2 - 6:.1f},{y1:.1f} C{x1 + node_w / 2 + bulge:.1f},{y1:.1f} "
                     f"{x2 + node_w / 2 + bulge:.1f},{y2:.1f} {x2 + node_w / 2 - 4:.1f},{y2:.1f}")
            else:
                d = (f"M{x1 + 4:.1f},{y1:.1f} C{(x1 + x2) / 2:.1f},{y1:.1f} "
                     f"{(x1 + x2) / 2:.1f},{y2:.1f} {x2 - node_w / 2 - 2:.1f},{y2:.1f}")
            bad = cid in cyc_nodes and p in cyc_nodes
            dash2 = " stroke-dasharray='4 3'" if bad else ""
            parts.append(
                f'<path class="edge" data-src="{esc(p)}" data-dst="{esc(cid)}" d="{d}" '
                f'stroke="{"#e8684a" if bad else "#4d5a6b"}" stroke-width="{1.4 if bad else 1}"'
                f'{dash2} opacity="{0.85 if bad else 0.35}" marker-end="url(#arr2)"/>'
            )

    for c in concepts:
        cid = c.get("id")
        if cid not in pos:
            continue
        x, y = pos[cid]
        color = DIM_COLOR.get(c.get("dimension", ""), "#8b949e")
        tip = (f"<b>{esc(c.get('name', ''))}</b><br>{esc(c.get('one_liner', ''))}"
               f"<br><span style=\"color:#8b949e\">{esc(c.get('dimension', ''))} · "
               f"{esc(c.get('stage', ''))} · 前置 {len(c.get('prereq') or [])}</span>")
        parts.append(
            f'<g class="node" data-id="{esc(cid)}" data-tip="{tip}">'
            f'<rect x="{x - node_w / 2:.1f}" y="{y - node_h / 2:.1f}" width="{node_w}" height="{node_h}" rx="5" '
            f'fill="{color}" fill-opacity=".18" stroke="{color}" stroke-width="1.1"/>'
            f'<text x="{x:.1f}" y="{y + 4:.1f}" fill="#d5dde5" font-size="10.5" text-anchor="middle">'
            f'{esc(shorten(c.get("name", ""), 14))}</text></g>'
        )
    parts.append("</svg>")
    return "".join(parts)


# ---------------------------------------------------------------- 组装

def html_page(title: str, subtitle: str, panels: Sequence[tuple[str, str]]) -> str:
    """单文件 HTML。所有 CSS/JS 内联，不引任何 CDN——断网也能打开。"""
    tabs, bodies = [], []
    for i, (label, body) in enumerate(panels):
        active = " on" if i == 0 else ""
        target = f"p{i}"
        tabs.append(f'<div class="tab{active}" data-target="{target}">{esc(label)}</div>')
        bodies.append(f'<section class="panel{active}" id="{target}">{body}</section>')
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>{esc(title)}</title>
<style>{CSS}</style></head><body>
<header><h1>{esc(title)}</h1><div class="sub">{esc(subtitle)}</div></header>
<div class="tabs">{''.join(tabs)}</div>
<main>{''.join(bodies)}</main>
<div id="tip"></div><script>{JS}</script></body></html>"""


def legend(items: Sequence[tuple[str, str]], tail: str = "") -> str:
    spans = "".join(
        f'<span><i style="background:{color}"></i>{esc(label)}</span>' for label, color in items
    )
    tail_html = f"<span>{esc(tail)}</span>" if tail else ""
    return f'<div class="legend">{spans}{tail_html}</div>'


def facts(text: str, *, warn: bool = False) -> str:
    cls = "warn" if warn else "facts"
    return f'<div class="{cls}">{text}</div>'


def coverage_line(retrieval: Retrieval) -> str:
    cov = retrieval.coverage
    parts = []
    for s in cov.statuses:
        flag = "✓" if s.ok else "✗"
        detail = f"{s.count} 条" if s.ok else (s.error or "无结果")
        parts.append(f"{s.source.label} {flag} {detail}")
    return "　".join(parts) + f"　（{cov.note()}）"


def markdown_report(
    retrieval: Retrieval,
    triages: Sequence[Triage] | None = None,
    *,
    graph: PaperGraph | None = None,
    snapshot: str = "",
) -> str:
    """Markdown 报告。检索式、时间、覆盖度放在最前面——先给可信度，再给结论。"""
    lines: list[str] = []
    lines.append(f"# 文献速览：{retrieval.question}")
    lines.append("")
    lines.append("## 这次检索是怎么做的")
    lines.append("")
    lines.append(f"- 检索式：{' ；'.join(f'`{q}`' for q in retrieval.queries)}")
    lines.append(f"- 时间：{retrieval.started_at} → {retrieval.finished_at}")
    lines.append(f"- 覆盖度：{coverage_line(retrieval)}")
    if retrieval.dropped:
        lines.append(f"- 过滤：丢弃 {retrieval.dropped} 条与本问题无关的结果（关键词检索跑偏是常态）")
    if snapshot:
        lines.append(f"- 快照：`{snapshot}`（重放同一份快照可复现本次结果）")
    lines.append("")

    if retrieval.warnings:
        lines.append("## 需要你知道的局限")
        lines.append("")
        for w in retrieval.warnings:
            lines.append(f"- {w}")
        lines.append("")

    lines.append(f"## 结果（{len(retrieval.papers)} 篇）")
    lines.append("")
    if triages:
        for group in (Verdict.READ_NOW, Verdict.SKIM, Verdict.PARK, Verdict.DROP):
            bucket = [t for t in triages if t.verdict is group]
            if not bucket:
                continue
            lines.append(f"### {group.label}（{len(bucket)}）")
            lines.append("")
            for t in bucket:
                p = t.paper
                cited_txt = f"被引 {p.cited_by}" if p.cited_known else "被引数未知"
                meta = f"{p.year or '—'} · {p.venue or p.source.label} · {cited_txt}"
                lines.append(f"- **{p.title}** —— {t.reason}")
                lines.append(f"  - {meta}")
                if t.read_for:
                    lines.append(f"  - 读它为了：{t.read_for}")
                if t.prereq:
                    lines.append(f"  - 可能的前置概念：{'、'.join(t.prereq)}")
                if p.url:
                    lines.append(f"  - {p.url}")
            lines.append("")
    else:
        for p in retrieval.papers:
            lines.append(f"- **{p.title}**（{p.year or '—'}，被引 {p.cited_by}）{p.url}")
        lines.append("")

    if graph is not None:
        deg = graph.degree()
        lines.append("## 关系图")
        lines.append("")
        lines.append(f"- 节点 {len(graph.nodes)}，边 {len(graph.edges)}，孤立点 {sum(1 for d in deg.values() if d == 0)}")
        by_rel: dict[str, int] = defaultdict(int)
        for e in graph.edges:
            by_rel[e.relation.label] += 1
        if by_rel:
            lines.append(f"- 关系构成：{'、'.join(f'{k} {v}' for k, v in by_rel.items())}")
        for w in graph.warnings:
            lines.append(f"- 注意：{w}")
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("> 每条文献都来自上面列出的检索式与来源；来源失败的已在覆盖度里写明。"
                 "没有任何一条是模型凭印象补充的。")
    return "\n".join(lines)
