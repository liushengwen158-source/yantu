"""openJiuwen 适配层：把环境配置能力组装成一个 ReAct agent。

用到的框架接口（openjiuwen 0.1.19）：
- ``openjiuwen.core.foundation.tool.tool``      把普通函数变成 LocalFunction 工具
- ``openjiuwen.core.single_agent.agents.react_agent.ReActAgent`` / ``ReActAgentConfig``
- ``openjiuwen.core.single_agent.schema.agent_card.AgentCard``
- ``agent.ability_manager.add_ability(card, resource)`` 注册工具

设计取舍：工具函数本体留在 ``research_agent.tools.env_tools``（纯 Python，
可单独测试），这里只做 schema 绑定和输出裁剪，避免两套实现漂移。
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any

from . import llm
from .tools.env_tools import env_check, env_explain, env_fix, env_guidance, env_plan
from .tools.repro_tools import repro_diff, repro_report, repro_scan

SYSTEM_PROMPT = """你是「研途」，面向计算机科研新手的助手，负责两件事：**开发环境配置**与**论文复现可行性判断**。

环境工具：
- env_check：体检本机环境，返回 Python/pip/conda/Git/SSH/GPU/CUDA/PyTorch 的真实状态
- env_guidance：按严重程度列出需要处理的问题和处理步骤
- env_explain：解释某个具体检测项的原因与修法
- env_plan：生成修复命令清单（不执行）
- env_fix：执行一条修复命令，system 级（安装/卸载软件）必须得到用户明确同意

复现工具（需要用户提供论文配套代码在本机的目录）：
- repro_scan：从仓库文件提取环境需求，每条都带「文件名:行号」来源
- repro_diff：把需求与本机比对，给出四档结论：block 物理不可能 / degrade 改配置能跑 / add 装一下即可 / ok 满足
- repro_report：输出完整的 Markdown 可行性报告

工作方式：
1. 用户问环境问题时，先 env_check 或 env_guidance 拿到事实，再回答，不要凭印象猜测。
2. 用户问「这篇论文的代码我能不能跑」时，先 repro_diff，不要凭经验判断。
3. 给命令前确认它适用于用户的操作系统。
4. 要执行安装/卸载类命令时，先把命令和影响告诉用户，等用户明确说要执行再调用 env_fix 并传 approve_system=true。
5. 复现结论要分清程度：卡数不够、显存差 6 倍以上属于物理阻塞，直说要换算力；版本不符属于装一下就好，别夸大。
6. 引用复现证据时带上来源（如 README.md:5）和可信度（L2 代码推断 / L3 经验推测）。
7. 仓库里没写、工具没返回的信息，直说不知道，不要编造版本号、卡数或命令。
8. 回答简洁：先结论，再步骤，命令用代码块。
"""


def _dump(payload: Any, limit: int = 6000) -> str:
    """工具结果转成模型可读的文本，并限制长度避免撑爆上下文。"""
    text = json.dumps(payload, ensure_ascii=False, indent=1)
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n...（已截断，原长 {len(text)} 字符）"


def build_tools():
    """构造全部 LocalFunction 工具（5 个环境 + 3 个复现）。"""
    from openjiuwen.core.foundation.tool import tool

    @tool(
        name="env_check",
        description=(
            "体检本机开发环境，返回 Python 版本与解释器构建、pip 与镜像源、conda、"
            "虚拟环境支持、Git 与身份配置、SSH 密钥、NVIDIA 驱动、CUDA Toolkit、"
            "PyTorch 及 CUDA 可用性等检测结果和就绪度评分。"
        ),
    )
    def env_check_tool(check_network: bool = False, check_torch: bool = True) -> str:
        """体检本机环境。

        Args:
            check_network: 是否探测代码托管连通性，较慢，默认关闭
            check_torch: 是否探测 PyTorch，默认开启
        """
        return _dump(env_check(check_network=check_network, check_torch=check_torch))

    @tool(
        name="env_guidance",
        description="列出当前环境中需要处理的问题，按「阻塞 > 注意 > 建议」排序，附处理步骤和验证命令。",
    )
    def env_guidance_tool() -> str:
        """获取按严重程度排序的问题与处理步骤。"""
        return _dump(env_guidance()) + "\n\n---\n" + env_guidance().get("text", "")

    @tool(
        name="env_explain",
        description="解释一个具体检测项：为什么重要、当前值是什么、怎么修、怎么验证。",
    )
    def env_explain_tool(check_id: str) -> str:
        """解释某个检测项。

        Args:
            check_id: 检测项 id，例如 python.version、pip.index、git.identity、gpu.driver、torch.installed
        """
        return _dump(env_explain(check_id))

    @tool(
        name="env_plan",
        description="生成修复命令清单但不执行。返回每条命令的风险级别（safe/config/system）和适用平台。",
    )
    def env_plan_tool(max_risk: str = "config") -> str:
        """生成修复计划。

        Args:
            max_risk: 允许的最高风险级别，可选 safe、config、system
        """
        return _dump(env_plan(max_risk=max_risk))

    @tool(
        name="env_fix",
        description=(
            "执行一条修复命令。命令必须来自 env_plan 的结果。"
            "安装/卸载类命令属于 system 级，需要用户明确同意后才能执行。"
        ),
    )
    def env_fix_tool(command: str, approve_system: bool = False) -> str:
        """执行修复命令。

        Args:
            command: 要执行的命令，必须出现在 env_plan 的返回结果中
            approve_system: 用户是否已明确同意执行 system 级命令
        """
        return _dump(env_fix(command=command, approve_system=approve_system))

    # ---- 论文复现 ------------------------------------------------------

    @tool(
        name="repro_scan",
        description=(
            "从论文配套代码目录提取环境需求（Python 版本、框架版本、CUDA、GPU 卡数与显存）。"
            "解析 Dockerfile、environment.yml、requirements.txt、pyproject.toml、"
            ".python-version、README。每条需求都带文件名:行号来源，无来源的会被丢弃。"
        ),
    )
    def repro_scan_tool(repo_path: str) -> str:
        """提取论文仓库的环境需求。

        Args:
            repo_path: 论文配套代码在本机的目录路径，不能是 URL
        """
        return _dump(repro_scan(repo_path))

    @tool(
        name="repro_diff",
        description=(
            "把论文的环境需求与本机环境比对，给出四档可行性结论："
            "block（物理不可能，需换硬件）、degrade（改 batch/精度才能跑）、"
            "add（装一下或换版本即可）、ok（满足）。附差距矩阵和处理建议。"
            "用户问「这篇论文的代码我能不能跑」时用这个。"
        ),
    )
    def repro_diff_tool(repo_path: str) -> str:
        """比对论文需求与本机环境。

        Args:
            repo_path: 论文配套代码在本机的目录路径，不能是 URL
        """
        return _dump(repro_diff(repo_path))

    @tool(
        name="repro_report",
        description="生成完整的论文复现可行性报告（Markdown），含差距矩阵、逐项证据与处理建议。",
    )
    def repro_report_tool(repo_path: str) -> str:
        """生成 Markdown 可行性报告。

        Args:
            repo_path: 论文配套代码在本机的目录路径，不能是 URL
        """
        return _dump(repro_report(repo_path), limit=8000)

    return [
        env_check_tool,
        env_guidance_tool,
        env_explain_tool,
        env_plan_tool,
        env_fix_tool,
        repro_scan_tool,
        repro_diff_tool,
        repro_report_tool,
    ]


def build_agent(
    model_name: str | None = None,
    api_key: str | None = None,
    api_base: str | None = None,
    max_iterations: int = 8,
):
    """构造 ReAct agent 并注册全部环境工具。

    模型参数缺省时从环境变量读取（DEEPSEEK_* / QWEN_* / OPENAI_*）。
    """
    from openjiuwen.core.single_agent.agents.react_agent import ReActAgent, ReActAgentConfig
    from openjiuwen.core.single_agent.schema.agent_card import AgentCard

    cfg = llm.load_config()
    model_name = model_name or (cfg.model if cfg else "")
    api_key = api_key or (cfg.api_key if cfg else "")
    api_base = api_base or (cfg.base_url if cfg else "")

    if not model_name or not api_key:
        raise RuntimeError(
            "缺少模型配置。设置 DEEPSEEK_API_KEY（可选 DEEPSEEK_MODEL / DEEPSEEK_BASE_URL）"
            "或在调用时显式传入。"
        )

    card = AgentCard(
        name="yantu_agent",
        description="计算机科研入门助手：开发环境诊断与修复、论文复现可行性判断",
        input_params={
            "type": "object",
            "properties": {"query": {"type": "string", "description": "用户的问题"}},
            "required": ["query"],
        },
    )
    agent = ReActAgent(card=card)

    config = ReActAgentConfig(
        model_name=model_name,
        model_provider="openai",
        api_key=api_key,
        api_base=api_base,
        max_iterations=max_iterations,
    )
    config.configure_prompt_template([{"role": "system", "content": SYSTEM_PROMPT}])
    agent.configure(config)

    for t in build_tools():
        agent.ability_manager.add_ability(t.card, t)

    return agent


async def arun(query: str, **agent_kwargs) -> str:
    """异步跑一轮对话，返回最终回答文本。"""
    agent = build_agent(**agent_kwargs)
    result = await agent.invoke({"query": query})
    if isinstance(result, dict):
        return str(result.get("output") or result)
    return str(result)


def run(query: str, **agent_kwargs) -> str:
    """同步入口。"""
    return asyncio.run(arun(query, **agent_kwargs))


def sdk_available() -> bool:
    try:
        import openjiuwen.core.single_agent.agents.react_agent  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False
