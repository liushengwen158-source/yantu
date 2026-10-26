"""问答层：把「当前环境状态 + 知识库」组织成有依据的回答。

两种模式：
- 配了模型 API Key：把检测结果和相关知识条目作为上下文，让模型组织语言
- 没配 Key：用关键词检索知识库，直接返回条目内容（不编造）

无论哪种模式，事实都来自 detect() 和 env_kb.json，模型只负责表达。
"""

from __future__ import annotations

import re

from . import llm
from .envsetup import build_guidance, detect, guidance_to_text, load_knowledge

SYSTEM_PROMPT = """你是「研途」，一个帮计算机科研新手配置开发环境的助手。

规则：
1. 只依据提供的【环境检测结果】和【知识库条目】回答，不要编造命令或版本号。
2. 用户系统是 {platform}，给出的命令必须能在这个系统上直接运行。
3. 先给结论，再给步骤。步骤要短、可复制。
4. 涉及安装/卸载时，提醒用户这会改动系统。
5. 如果知识库里没有相关信息，直接说「这个问题知识库里没有覆盖」，并给出排查方向。
6. 不要输出与问题无关的通用建议。

【环境检测结果摘要】
{env_summary}

【相关知识库条目】
{knowledge}

【本次检测发现的问题】
{issues}
"""


def _bigrams(text: str) -> set[str]:
    """中文按二元组切分，英文按单词切分，做粗粒度检索足够。"""
    text = text.lower()
    grams: set[str] = set()
    for token in re.findall(r"[a-z0-9_.\-]+", text):
        grams.add(token)
    han = re.findall(r"[\u4e00-\u9fff]+", text)
    for chunk in han:
        for i in range(len(chunk) - 1):
            grams.add(chunk[i:i + 2])
    return grams


def retrieve(question: str, kb: dict, top_k: int = 3) -> list[dict]:
    """从知识库检索最相关的条目。"""
    q = _bigrams(question)
    if not q:
        return []
    scored: list[tuple[float, dict]] = []
    for entry in kb.values():
        body = " ".join([
            entry.get("title", ""),
            entry.get("why", ""),
            " ".join(s.get("text", "") for s in entry.get("steps", [])),
        ])
        target = _bigrams(body)
        if not target:
            continue
        overlap = len(q & target)
        if overlap == 0:
            continue
        # 用问题侧覆盖率排序：命中的问题词越多越相关
        score = overlap / len(q)
        scored.append((score, entry))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [e for _, e in scored[:top_k]]


def _format_knowledge(entries: list[dict]) -> str:
    if not entries:
        return "（无直接匹配的条目）"
    blocks = []
    for e in entries:
        steps = []
        for s in e.get("steps", []):
            text = s.get("text", "")
            cmd = s.get("command", "")
            steps.append(f"  - {text}" + (f"\n    `{cmd}`" if cmd else ""))
        blocks.append(
            f"### {e['title']}（{e['id']}）\n"
            f"原因：{e.get('why', '')}\n"
            f"步骤：\n" + "\n".join(steps) +
            (f"\n验证：`{e['verify']}`" if e.get("verify") else "")
        )
    return "\n\n".join(blocks)


def build_prompt(question: str) -> str:
    kb = load_knowledge()
    report = detect(check_network=False, check_torch=True)
    guidance = build_guidance(report, kb)
    s = report.summary()

    env_summary = (
        f"系统：{report.platform}\n"
        f"解释器：{report.python_executable}（{report.python_version}）\n"
        f"就绪度：{s['score']}/100，阻塞 {s['blockers']}，注意 {s['warnings']}\n"
        + "\n".join(f"- [{r.status.label}] {r.name}: {r.value}"
                    for r in report.results)
    )

    return SYSTEM_PROMPT.format(
        platform=report.platform,
        env_summary=env_summary,
        knowledge=_format_knowledge(retrieve(question, kb)),
        issues=guidance_to_text(guidance) if guidance else "无",
    )


def _rule_answer(question: str) -> str:
    """没有模型时的降级回答：只复述知识库，不臆测。"""
    kb = load_knowledge()
    hits = retrieve(question, kb, top_k=2)
    if not hits:
        return (
            "没有配置模型 API Key（可设置 DEEPSEEK_API_KEY 等环境变量），"
            "且知识库里没有直接匹配这个问题的条目。\n\n"
            "当前可问的方向：Python 版本与虚拟环境、pip 镜像源、Git 配置与网络、"
            "CUDA 与驱动关系、PyTorch 安装、显存不足。\n"
            "也可以直接跑 `python cli.py check` 看完整体检报告。"
        )
    head = "（未配置模型，以下内容直接来自知识库）\n"
    return head + _format_knowledge(hits)


def ask(question: str) -> str:
    """回答一个环境相关的问题。"""
    if not llm.available():
        return _rule_answer(question)

    messages = [
        {"role": "system", "content": build_prompt(question)},
        {"role": "user", "content": question},
    ]
    try:
        return llm.chat(messages)
    except llm.LLMError as exc:
        return f"调用模型失败：{exc}\n\n以下为知识库直接检索结果：\n\n{_rule_answer(question)}"
