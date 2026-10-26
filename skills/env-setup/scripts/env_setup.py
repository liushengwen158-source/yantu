#!/usr/bin/env python
"""环境配置能力的技能入口。

供 JiuwenSwarm 技能系统调用，也可单独使用。真正的逻辑在 yantu 项目里
（research_agent 包），这里只做路径定位和命令分发，避免两份代码漂移。

用法：
    python env_setup.py check            # 体检，输出 markdown
    python env_setup.py check --json     # 结构化输出
    python env_setup.py plan             # 修复计划（不执行）
    python env_setup.py explain <id>     # 解释某个检测项
    python env_setup.py fix --command "..." [--yes]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# 项目根目录：优先环境变量，其次默认安装位置
DEFAULT_HOME = Path.home() / "Desktop" / "yantu"
YANTU_HOME = Path(os.environ.get("YANTU_HOME", str(DEFAULT_HOME)))

if not (YANTU_HOME / "research_agent").is_dir():
    sys.stderr.write(
        f"找不到 yantu 项目：{YANTU_HOME}\n"
        f"请设置环境变量 YANTU_HOME 指向项目根目录。\n"
    )
    sys.exit(2)

sys.path.insert(0, str(YANTU_HOME))

from research_agent.envsetup import (  # noqa: E402
    build_guidance,
    build_plan,
    detect,
    guidance_to_text,
    load_knowledge,
    render_actions,
    render_markdown,
)

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(errors="replace")
    except (AttributeError, ValueError):
        pass


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
    if args.json:
        print(json.dumps([a.to_dict() for a in actions], ensure_ascii=False, indent=2))
    else:
        print(render_actions([a.to_dict() for a in actions]))
    return 0


def cmd_explain(args: argparse.Namespace) -> int:
    report = detect(check_network=False, check_torch=False)
    kb = load_knowledge()
    for r in report.results:
        if r.id != args.check_id:
            continue
        entry = kb.get(r.kb_id or "", {})
        payload = {
            "check_id": r.id,
            "name": r.name,
            "status": r.status.value,
            "observed": r.value,
            "message": r.message,
            "title": entry.get("title", ""),
            "why": entry.get("why", ""),
            "steps": entry.get("steps", []),
            "verify": entry.get("verify", ""),
            "doc": entry.get("doc", ""),
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0
    print(json.dumps({
        "error": f"没有名为 {args.check_id} 的检测项",
        "available": [r.id for r in report.results],
    }, ensure_ascii=False, indent=2))
    return 1


def cmd_fix(args: argparse.Namespace) -> int:
    from research_agent.envsetup import execute

    report = detect(check_network=False, check_torch=False)
    actions = build_plan(report, load_knowledge(), max_risk="system")
    matched = next((a for a in actions if a.command.strip() == args.command.strip()), None)
    if matched is None:
        print(json.dumps({
            "ok": False,
            "error": "该命令不在当前修复计划中，拒绝执行",
            "allowed": [a.command for a in actions],
        }, ensure_ascii=False, indent=2))
        return 1

    if args.dry_run:
        print(json.dumps({"ok": True, "dry_run": True, **matched.to_dict()},
                         ensure_ascii=False, indent=2))
        return 0

    result = execute(matched, approve_system=args.yes)
    result["ok"] = bool(result.get("executed") and result.get("ok"))
    result["risk"] = matched.risk
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["ok"] else 1


def cmd_guidance(args: argparse.Namespace) -> int:
    report = detect(check_network=False, check_torch=True)
    guidance = build_guidance(report, load_knowledge())
    if args.json:
        print(json.dumps(guidance, ensure_ascii=False, indent=2))
    else:
        print(guidance_to_text(guidance))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="env_setup", description="科研环境配置助手")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("check", help="体检当前环境")
    p.add_argument("--json", action="store_true")
    p.add_argument("--no-network", action="store_true")
    p.add_argument("--no-torch", action="store_true")
    p.set_defaults(func=cmd_check)

    p = sub.add_parser("plan", help="生成修复计划")
    p.add_argument("--risk", default="config", choices=["safe", "config", "system"])
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_plan)

    p = sub.add_parser("explain", help="解释某个检测项")
    p.add_argument("check_id")
    p.set_defaults(func=cmd_explain)

    p = sub.add_parser("guidance", help="按严重程度列出问题与步骤")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_guidance)

    p = sub.add_parser("fix", help="执行一条修复命令")
    p.add_argument("--command", required=True)
    p.add_argument("--yes", action="store_true", help="允许执行 system 级命令")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_fix)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
