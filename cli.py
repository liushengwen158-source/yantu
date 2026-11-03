#!/usr/bin/env python
"""命令行入口：不接大模型也能直接用的科研入门工具。

用法：
    python cli.py check                 体检，输出可读报告
    python cli.py check --json          输出结构化结果（给 agent 或脚本用）
    python cli.py plan                  只看要跑哪些命令（不执行）
    python cli.py fix [--risk config]   执行修复动作，system 级需 --yes
    python cli.py ask "torch 装不上"     问知识库 + 大模型（需配 API Key）
    python cli.py guide profile "..."   从对话推断六维能力画像
    python cli.py guide roadmap "..."   生成带验收标准的学习路线
    python cli.py guide questions "..." 列出接下来该问的问题
    python cli.py guide domains         列出方向及其要求
    python cli.py guide explain 反向传播 解释一个概念
    python cli.py guide domains         列出可选研究方向
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
# 新手引导
# ---------------------------------------------------------------------------

def cmd_guide_profile(args: argparse.Namespace) -> int:
    from pathlib import Path

    from research_agent.guide import build_profile, render_profile, scan_materials

    if args.file:
        path = Path(args.file)
        if not path.exists():
            print(f"找不到文件：{args.file}")
            return 2
        text = path.read_text(encoding="utf-8")
    else:
        text = args.text or ""

    scan = None
    if args.scan:
        try:
            scan = scan_materials(args.scan)
        except (FileNotFoundError, NotADirectoryError) as exc:
            print(f"扫描失败：{exc}")
            return 2

    if not text.strip() and scan is None:
        print("给一段对话内容，或用 --file 指定文件；也可以加 --scan 授权扫描一个目录。例如：")
        print('  python cli.py guide profile "我写过爬虫，torch 装了但 cuda 一直是 false"')
        print("  python cli.py guide profile --scan ~/Desktop/papers")
        return 2

    domain = _resolve_domain_arg(getattr(args, "domain", ""))
    if domain is False:
        return 2

    profile = build_profile(text, scan=scan, domain=domain)
    if args.json:
        payload = profile.to_dict()
        if scan is not None:
            payload["materials"] = scan.to_dict()
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(render_profile(profile))
    return 0


def _guide_inputs(args: argparse.Namespace):
    """读取对话文本与授权目录，两者都可以没有。"""
    from pathlib import Path

    from research_agent.guide import scan_materials

    if args.file:
        path = Path(args.file)
        if not path.exists():
            raise FileNotFoundError(f"找不到文件：{args.file}")
        text = path.read_text(encoding="utf-8")
    else:
        text = args.text or ""

    scan = scan_materials(args.scan) if args.scan else None
    return text, scan


def cmd_guide_roadmap(args: argparse.Namespace) -> int:
    from pathlib import Path

    from research_agent.guide import build_profile, build_roadmap, render_roadmap

    try:
        text, scan = _guide_inputs(args)
    except (FileNotFoundError, NotADirectoryError) as exc:
        print(f"读取失败：{exc}")
        return 2

    if not text.strip() and scan is None:
        print("生成路线需要一点信息：描述一下你现在的情况，或者用 --scan 授权扫描材料目录。")
        print('  python cli.py guide roadmap "我大一，Python 只会 print，没跑过训练"')
        return 2

    domain = _resolve_domain_arg(getattr(args, "domain", ""))
    if domain is False:
        return 2

    profile = build_profile(text, scan=scan, domain=domain)
    roadmap = build_roadmap(
        profile, hours_limit=args.hours * 8 if args.hours else None
    )

    if args.json:
        print(json.dumps(roadmap.to_dict(), ensure_ascii=False, indent=2))
        return 0

    output = render_roadmap(roadmap)
    if args.out:
        Path(args.out).write_text(output, encoding="utf-8")
        print(f"路线已写入 {args.out}")
    else:
        print(output)
    return 0



def _resolve_domain_arg(value: str):
    """解析 --domain 参数。返回 Domain / None（未指定）/ False（无法识别）。"""
    from research_agent.guide import list_domains, resolve_domain

    if not (value or "").strip() or value in ("general", "unknown"):
        return None
    try:
        return resolve_domain(value)
    except ValueError as exc:
        print(f"{exc}")
        print("可选方向：")
        for item in list_domains():
            print(f"  {item['id']:<12} {item['name']}")
        return False


def cmd_guide_questions(args: argparse.Namespace) -> int:
    from research_agent.guide import (
        build_profile,
        probe_plan,
        render_questions,
        scan_materials,
    )

    try:
        text, scan = _guide_inputs(args)
    except (FileNotFoundError, NotADirectoryError) as exc:
        print(f"读取失败：{exc}")
        return 2

    domain = _resolve_domain_arg(getattr(args, "domain", ""))
    if domain is False:
        return 2

    profile = build_profile(text, scan=scan, domain=domain)
    plan = probe_plan(profile, limit=args.limit)
    if args.json:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0
    print(render_questions(plan))
    return 0


def cmd_guide_domains(args: argparse.Namespace) -> int:
    from research_agent.guide import list_domains

    items = [spec.to_dict() for spec in list_domains()]
    if args.json:
        print(json.dumps(items, ensure_ascii=False, indent=2))
        return 0
    for item in items:
        print(f"{item['id']:<12} {item['name']}")
        if item.get("why"):
            print(f"{'':<12} {item['why']}")
        dims = "、".join(f"{k}{v}" for k, v in (item.get("dims") or {}).items())
        print(f"{'':<12} 维度要求：{dims}")
        print(f"{'':<12} 论文：{len(item.get('papers') or [])} 篇")
    return 0

def cmd_guide_explain(args: argparse.Namespace) -> int:
    from research_agent.guide import explain, render_card, render_miss

    result = explain(args.term, dimension=args.dimension)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["found"] else 1

    if result["found"]:
        print(render_card(result["card"]))
        if result["candidates"]:
            print("")
            print("也可能你想问：")
            for item in result["candidates"]:
                print(f"  - {item['name']}（{item['id']}）")
        return 0

    print(render_miss(result))
    return 1


def main(argv: list[str] | None = None) -> int:
    _ensure_console_encoding()
    parser = argparse.ArgumentParser(
        prog="yantu", description="计算机科研入门 agent · 环境配置 / 论文复现 / 新手引导"
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

    p_guide = sub.add_parser("guide", help="新手引导：从对话推断基础水平")
    guide_sub = p_guide.add_subparsers(dest="guide_command", required=True)

    p_profile = guide_sub.add_parser("profile", help="从对话文本生成能力画像")
    p_profile.add_argument("text", nargs="?", default="", help="对话内容（或用 --file）")
    p_profile.add_argument("--file", help="从文本文件读取对话内容")
    p_profile.add_argument("--scan", help="授权扫描的材料目录（PDF / notebook / 笔记 / 源码）")
    p_profile.add_argument("--domain", help="方向 id 或名称，如 systems / 系统；留空则从对话识别")
    p_profile.add_argument("--json", action="store_true", help="输出 JSON（给 agent 或脚本用）")
    p_profile.set_defaults(func=cmd_guide_profile)

    p_roadmap = guide_sub.add_parser("roadmap", help="按画像生成学习路线")
    p_roadmap.add_argument("text", nargs="?", default="", help="对话内容（或用 --file）")
    p_roadmap.add_argument("--file", help="从文本文件读取对话内容")
    p_roadmap.add_argument("--scan", help="授权扫描的材料目录")
    p_roadmap.add_argument("--domain", help="方向 id 或名称，如 database / 数据库；留空则从对话识别")
    p_roadmap.add_argument("--hours", type=int, help="每周可投入小时数（按 8 周折算总预算）")
    p_roadmap.add_argument("--out", help="写入指定 Markdown 文件")
    p_roadmap.add_argument("--json", action="store_true")
    p_roadmap.set_defaults(func=cmd_guide_roadmap)

    p_explain = guide_sub.add_parser("explain", help="解释一个概念")
    p_explain.add_argument("term", help="概念名，如 反向传播、venv、过拟合")
    p_explain.add_argument("--dimension", help="限定维度：coding/tools/math/research")
    p_explain.add_argument("--json", action="store_true")
    p_explain.set_defaults(func=cmd_guide_explain)

    p_questions = guide_sub.add_parser("questions", help="按优先级列出接下来该问的问题")
    p_questions.add_argument("text", nargs="?", default="")
    p_questions.add_argument("--file")
    p_questions.add_argument("--scan")
    p_questions.add_argument("--domain")
    p_questions.add_argument("--limit", type=int, default=3)
    p_questions.add_argument("--json", action="store_true")
    p_questions.set_defaults(func=cmd_guide_questions)

    p_domains = guide_sub.add_parser("domains", help="列出研究方向及要求（含论文与项目）")
    p_domains.add_argument("--json", action="store_true")
    p_domains.set_defaults(func=cmd_guide_domains)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
