#!/usr/bin/env python
"""命令行入口：不接大模型也能直接用的环境配置工具。

用法：
    python cli.py check                 体检，输出可读报告
    python cli.py check --json          输出结构化结果（给 agent 或脚本用）
    python cli.py plan                  只看要跑哪些命令（不执行）
    python cli.py fix [--risk config]   执行修复动作，system 级需 --yes
    python cli.py ask "torch 装不上"     问知识库 + 大模型（需配 API Key）
"""

from __future__ import annotations

import argparse
import json
import sys

from research_agent.envsetup import (
    build_guidance,
    build_plan,
    detect,
    execute,
    load_knowledge,
    render_actions,
    render_markdown,
)


def cmd_check(args: argparse.Namespace) -> int:
    report = detect(check_network=not args.no_network, check_torch=not args.no_torch)
    if args.json:
        print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    else:
        print(render_markdown(report, load_knowledge()))
    return 0 if not report.blockers() else 2


def cmd_plan(args: argparse.Namespace) -> int:
    report = detect(check_network=False, check_torch=True)
    actions = build_plan(report, load_knowledge(), max_risk=args.risk)
    print(render_actions([a.to_dict() for a in actions]))
    return 0


def cmd_fix(args: argparse.Namespace) -> int:
    report = detect(check_network=False, check_torch=True)
    actions = build_plan(report, load_knowledge(), max_risk=args.risk)
    if not actions:
        print("没有需要执行的命令。")
        return 0

    if args.dry_run:
        print(render_actions([a.to_dict() for a in actions]))
        return 0

    for i, action in enumerate(actions, 1):
        if action.risk == "system" and not args.yes:
            print(f"[跳过] {action.command}（system 级，加 --yes 才会执行）")
            continue
        print(f"[{i}/{len(actions)}] {action.command}")
        result = execute(action, approve_system=args.yes)
        if result.get("executed"):
            print("   ->", "成功" if result.get("ok") else f"失败（code={result.get('code')}）")
            if not result.get("ok") and result.get("output"):
                print("  ", result["output"][-500:])
        else:
            print("   -> 未执行：", result.get("reason"))
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    from research_agent import agent_core, llm

    if agent_core.sdk_available() and llm.available():
        try:
            print(agent_core.run(args.question))
            return 0
        except Exception as exc:  # noqa: BLE001
            print(f"[agent 调用失败，降级为知识库检索] {type(exc).__name__}: {exc}\n")

    from research_agent.agent import ask

    print(ask(args.question))
    return 0


def _ensure_console_encoding() -> None:
    """中文 Windows 控制台是 GBK，遇到无法编码的字符不要直接崩。"""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")  # type: ignore[union-attr]
        except (AttributeError, ValueError):
            pass


# ---------------------------------------------------------------------------
# 论文复现
# ---------------------------------------------------------------------------

def cmd_paper_scan(args: argparse.Namespace) -> int:
    from research_agent.repro import scan_repo

    spec = scan_repo(args.repo)
    if args.json:
        print(json.dumps(spec.to_dict(), ensure_ascii=False, indent=2))
    else:
        print(f"扫描目录：{spec.repo_path}")
        print(f"识别文件：{', '.join(spec.scanned_files) or '无'}")
        print(f"提取需求：{len(spec.accepted)} 条（丢弃 {len(spec.rejected)} 条）")
        for r in spec.accepted:
            ev = r.evidence.describe() if r.evidence else "无来源"
            print(f"  - {r.id:<16} {r.requirement_text():<16} {ev}")
        for w in spec.warnings:
            print(f"  ! {w}")
    return 0


def cmd_paper_diff(args: argparse.Namespace) -> int:
    from research_agent.repro import collect_profile, diff, render_matrix, scan_repo

    spec = scan_repo(args.repo)
    profile = collect_profile(args.repo)
    report = diff(spec, profile)
    if args.json:
        print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    else:
        print(render_matrix(report))
    return 0


def cmd_paper_report(args: argparse.Namespace) -> int:
    from pathlib import Path

    from research_agent.repro import collect_profile, diff, render_report, scan_repo

    spec = scan_repo(args.repo)
    profile = collect_profile(args.repo)
    report = diff(spec, profile)
    text = render_report(report)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"报告已写入 {args.out}")
    else:
        print(text)
    return 0


# ---------------------------------------------------------------------------
# 文献检索与关联（lit）
# ---------------------------------------------------------------------------

def _coverage_line(retrieval) -> str:
    parts = []
    for s in retrieval.coverage.statuses:
        detail = f"{s.count} 条" if s.ok else (s.error or "无结果")
        parts.append(f"{s.source.label} {'✓' if s.ok else '✗'} {detail}")
    return "　".join(parts) + f"（{retrieval.coverage.note()}）"


def cmd_lit_search(args: argparse.Namespace) -> int:
    from research_agent.litreview import search

    retrieval = search(args.question, limit=args.limit, year_from=args.year_from or None,
                       offline=args.offline)
    if args.json:
        print(json.dumps(retrieval.to_dict(), ensure_ascii=False, indent=2))
        return 0 if retrieval.papers else 1

    print(f"检索式：{'；'.join(retrieval.queries)}")
    print(f"覆盖度：{_coverage_line(retrieval)}")
    if retrieval.dropped:
        print(f"过滤：丢弃 {retrieval.dropped} 条与本问题无关的结果")
    print()
    for i, p in enumerate(retrieval.papers, 1):
        tag = "【综述】" if p.is_survey else ""
        cited = f"被引 {p.cited_by}" if p.cited_known else "被引数未知"
        print(f"{i:>2}. {p.title} {tag}")
        print(f"    {p.year or '—'} · {p.venue or p.source.label} · {cited} · {p.url}")
    if not retrieval.papers:
        print("没有结果。检查网络 / API key，或把问题写具体一点。")
    for w in retrieval.warnings:
        print(f"! {w}")
    return 0 if retrieval.papers else 1


def cmd_lit_triage(args: argparse.Namespace) -> int:
    from research_agent.litreview import search, triage

    retrieval = search(args.question, limit=args.limit, year_from=args.year_from or None,
                       offline=args.offline)
    if not retrieval.papers:
        print(f"没有可用文献。{_coverage_line(retrieval)}")
        return 1
    triaged = triage(retrieval.papers, retrieval.question)
    if args.json:
        print(json.dumps([t.to_dict() for t in triaged], ensure_ascii=False, indent=2))
        return 0

    print(f"覆盖度：{_coverage_line(retrieval)}\n")
    current = None
    for t in triaged:
        if t.verdict is not current:
            current = t.verdict
            print(f"== {current.label} ==")
        print(f"  · {t.paper.title}")
        print(f"    {t.reason}")
        if t.read_for:
            print(f"    读它为了：{t.read_for}")
    return 0


def cmd_lit_expand(args: argparse.Namespace) -> int:
    from research_agent.litreview import expand
    from research_agent.litreview.spec import PaperRef, Source

    seed = PaperRef(source=Source.OPENALEX if args.doi else Source.CROSSREF,
                    source_id=args.doi or args.title[:80], title=args.title, doi=args.doi)
    graph = expand(seed, offline=args.offline)
    if args.json:
        print(json.dumps(graph.to_dict(), ensure_ascii=False, indent=2))
        return 0 if graph.nodes else 1

    if not graph.nodes:
        print("没能建立关系图。")
        for w in graph.warnings:
            print(f"! {w}")
        return 1
    deg = graph.degree()
    if args.out:
        from pathlib import Path

        from research_agent.litreview.render import REL_COLOR, html_page, legend, papers_svg
        from research_agent.litreview.spec import Relation

        edge_tuples = [(e.src, e.dst, e.relation) for e in graph.edges]
        svg = papers_svg(list(graph.nodes.items()), edges=edge_tuples)
        body = (
            f'<div class="facts">种子：{graph.nodes[graph.seed].short(90)}　'
            f'节点 {len(graph.nodes)}　边 {len(graph.edges)}　'
            f'提示：{"；".join(graph.warnings) or "无异常"}</div>'
            + svg
            + legend(
                [
                    ("引用（种子引用的文献）", REL_COLOR[Relation.CITES]),
                    ("被引（引用种子的文献）", REL_COLOR[Relation.CITED_BY]),
                    ("相似（站方推荐，不等于因果）", REL_COLOR[Relation.SIMILAR]),
                    ("文献耦合（共享 ≥2 篇参考文献）", REL_COLOR[Relation.COUPLING]),
                ],
                tail="横轴 = 发表年份；圆越大被引越多",
            )
        )
        Path(args.out).write_text(
            html_page("研途 · 引用关系图", graph.seed, [("引用关系", body)]), encoding="utf-8"
        )
        print(f"已写出 {args.out}（{len(graph.nodes)} 节点 / {len(graph.edges)} 边）")
        return 0

    print(f"种子：{graph.seed}")
    print(f"节点 {len(graph.nodes)}，边 {len(graph.edges)}，"
          f"孤立点 {sum(1 for d in deg.values() if d == 0)}")
    for e in graph.edges[:40]:
        src = graph.nodes[e.src].short(38)
        dst = graph.nodes[e.dst].short(38)
        print(f"  {e.relation.label:<6} {src}  →  {dst}")
    if len(graph.edges) > 40:
        print(f"  …（共 {len(graph.edges)} 条边，用 --json 看全部）")
    for w in graph.warnings:
        print(f"! {w}")
    return 0


def cmd_lit_map(args: argparse.Namespace) -> int:
    from research_agent.tools.lit_tools import lit_map

    res = lit_map(args.question, limit=args.limit, year_from=args.year_from or None,
                  out=args.out, offline=args.offline)
    if args.json:
        print(json.dumps(res, ensure_ascii=False, indent=2))
    elif res.get("ok"):
        print(f"已写出 {res['path']}")
    else:
        print(f"失败：{res.get('error')}")
    return 0 if res.get("ok") else 1


def cmd_lit_concept(args: argparse.Namespace) -> int:
    from research_agent.tools.lit_tools import lit_concept_map

    res = lit_concept_map(concepts_path=args.concepts, out=args.out)
    if not res.get("ok"):
        print(f"失败：{res.get('error')}")
        print(f"提示：{res.get('hint', '')}")
        return 1
    stats = res["stats"]
    print(f"概念 {stats['concepts']} 个 / 前置边 {stats['edges']} 条 / "
          f"根节点 {stats['roots']} / 跨维度 {stats['cross_dimension']}")
    for comp in stats["cycles"]:
        print(f"! 环形依赖：{' ↔ '.join(comp)}（互为前置，按图学会卡死）")
    print(f"已写出 {res['path']}")
    return 0


def cmd_lit_report(args: argparse.Namespace) -> int:
    from pathlib import Path

    from research_agent.litreview import search, triage
    from research_agent.litreview.render import markdown_report

    retrieval = search(args.question, limit=args.limit, year_from=args.year_from or None,
                       offline=args.offline)
    triaged = triage(retrieval.papers, retrieval.question) if retrieval.papers else []
    text = markdown_report(retrieval, triaged)
    Path(args.out).write_text(text, encoding="utf-8")
    print(f"报告已写入 {args.out}（{len(retrieval.papers)} 篇）")
    return 0


def main(argv: list[str] | None = None) -> int:
    _ensure_console_encoding()
    parser = argparse.ArgumentParser(
        prog="yantu", description="计算机科研入门 agent · 环境配置"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_check = sub.add_parser("check", help="体检当前环境")
    p_check.add_argument("--json", action="store_true", help="输出 JSON")
    p_check.add_argument("--no-network", action="store_true", help="跳过网络连通性探测")
    p_check.add_argument("--no-torch", action="store_true", help="跳过 torch 探测（更快）")
    p_check.set_defaults(func=cmd_check)

    p_plan = sub.add_parser("plan", help="列出待执行的修复命令")
    p_plan.add_argument("--risk", default="system", choices=["safe", "config", "system"])
    p_plan.set_defaults(func=cmd_plan)

    p_fix = sub.add_parser("fix", help="执行修复动作")
    p_fix.add_argument("--risk", default="system", choices=["safe", "config", "system"])
    p_fix.add_argument("--dry-run", action="store_true", help="只打印不执行")
    p_fix.add_argument("--yes", action="store_true", help="允许执行 system 级命令")
    p_fix.set_defaults(func=cmd_fix)

    p_ask = sub.add_parser("ask", help="就环境问题提问")
    p_ask.add_argument("question")
    p_ask.set_defaults(func=cmd_ask)

    p_paper = sub.add_parser("paper", help="论文复现：环境需求 vs 本机")
    paper_sub = p_paper.add_subparsers(dest="paper_command", required=True)

    p_scan = paper_sub.add_parser("scan", help="从仓库提取环境需求")
    p_scan.add_argument("repo", help="论文配套代码的本地路径")
    p_scan.add_argument("--json", action="store_true")
    p_scan.set_defaults(func=cmd_paper_scan)

    p_diff = paper_sub.add_parser("diff", help="与本机环境比对，输出差距矩阵")
    p_diff.add_argument("repo")
    p_diff.add_argument("--json", action="store_true")
    p_diff.set_defaults(func=cmd_paper_diff)

    p_report = paper_sub.add_parser("report", help="输出完整的可行性报告")
    p_report.add_argument("repo")
    p_report.add_argument("--out", help="写入指定文件")
    p_report.set_defaults(func=cmd_paper_report)

    # ---- 文献检索与关联 ----------------------------------------------
    p_lit = sub.add_parser("lit", help="文献检索与关联：搜、排、画")
    lit_sub = p_lit.add_subparsers(dest="lit_command", required=True)

    p_ls = lit_sub.add_parser("search", help="检索文献，输出候选与覆盖度")
    p_ls.add_argument("question", help="研究问题或关键词")
    p_ls.add_argument("--limit", type=int, default=15)
    p_ls.add_argument("--year-from", type=int, default=0, help="只要这一年及以后的")
    p_ls.add_argument("--offline", action="store_true", help="只用本地缓存")
    p_ls.add_argument("--json", action="store_true")
    p_ls.set_defaults(func=cmd_lit_search)

    p_lt = lit_sub.add_parser("triage", help="给出四档阅读建议")
    p_lt.add_argument("question")
    p_lt.add_argument("--limit", type=int, default=15)
    p_lt.add_argument("--year-from", type=int, default=0)
    p_lt.add_argument("--offline", action="store_true")
    p_lt.add_argument("--json", action="store_true")
    p_lt.set_defaults(func=cmd_lit_triage)

    p_le = lit_sub.add_parser("expand", help="以一篇论文为种子，拉引用关系图")
    p_le.add_argument("title", help="种子论文标题")
    p_le.add_argument("--doi", default="", help="有 DOI 就优先用")
    p_le.add_argument("--offline", action="store_true")
    p_le.add_argument("--out", default="", help="同时写出可交互 HTML 关系图")
    p_le.add_argument("--json", action="store_true")
    p_le.set_defaults(func=cmd_lit_expand)

    p_lm = lit_sub.add_parser("map", help="生成可离线打开的文献关系图 HTML")
    p_lm.add_argument("question")
    p_lm.add_argument("--limit", type=int, default=15)
    p_lm.add_argument("--year-from", type=int, default=0)
    p_lm.add_argument("--out", default="lit-map.html")
    p_lm.add_argument("--offline", action="store_true", help="只用本地缓存")
    p_lm.add_argument("--json", action="store_true")
    p_lm.set_defaults(func=cmd_lit_map)

    p_lc = lit_sub.add_parser("concept", help="画概念前置图（需要 learning-guide 的概念库）")
    p_lc.add_argument("--concepts", default="", help="concepts.json 路径")
    p_lc.add_argument("--out", default="concept-map.html")
    p_lc.set_defaults(func=cmd_lit_concept)

    p_lr = lit_sub.add_parser("report", help="生成 Markdown 文献速览报告")
    p_lr.add_argument("question")
    p_lr.add_argument("--limit", type=int, default=15)
    p_lr.add_argument("--year-from", type=int, default=0)
    p_lr.add_argument("--offline", action="store_true")
    p_lr.add_argument("--out", default="lit-report.md")
    p_lr.set_defaults(func=cmd_lit_report)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
