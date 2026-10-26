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

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
