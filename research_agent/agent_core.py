"""openJiuwen 适配层：把三类能力组装成一个 ReAct agent。

能力：环境配置、论文复现可行性、新手入门引导（基础评估 / 学习路线 / 概念解释）。

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
from .tools.guide_tools import (
    guide_assess,
    guide_domains,
    guide_explain,
    guide_questions,
    guide_roadmap,
)

SYSTEM_PROMPT = """你是「研途」，面向计算机科研新手的助手，负责三件事：**开发环境配置**、**论文复现可行性判断**、**新手引导（评估 → 路线 → 概念）**。

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

引导工具（评估 → 路线 → 概念）：
- guide_domains：列出可选研究方向，以及每个方向需要哪些能力、读哪些论文
- guide_assess：从用户原话（可选加授权目录）推断能力画像，每条结论带原话或文件路径；维度固定四个，方向只改门槛
- guide_roadmap：按方向生成分阶段路线，每项任务带验收标准与预估小时；已知达标的阶段会跳过并说明理由
- guide_explain：解释一个概念，概念库里查不到就直说没有，不要编

关于方向的硬规则：
1. 计算机的研究方向差别很大。**不要默认用户做人工智能**，也不要反过来先追问方向：用户自己会说。用户提到方向就按那个方向做，没提就按通用主干做。方向只有六个族：系统与网络（含网络、体系结构、分布式）、安全、数据与数据库、理论与算法、软件与程序语言（含图形学）、人工智能（含 AI4Science）。
2. 能力维度固定四个（编程与调试、命令行与工程工具、数学基础、科研方法），方向不会新增维度，只改每个维度的门槛。方向没确定就按通用要求评估，不要提「你还需要补机器学习」这类话——那可能跟用户完全无关。用户主动问「有哪些方向 / 我该学什么」时再列方向清单，别把清单当提问塞给用户。
3. 判断用户方向只用他自己说的话；他说的方向永远优先于关键词匹配。识别错了就说「我按 X 处理，不对你改」。
4. 不要给用户套通用模板：系统方向补 C 与并发，理论方向补证明，数据方向补查询，AI 方向补数学与框架——这些内容不一样，也不要互相贬低。

评估与路线的硬规则：
5. 评估只依据两样东西：用户原话，或用户**明确授权**的本地目录。不要主动扫盘、不要猜路径、不要问「要不要我扫一下你的电脑」。
6. 没证据就说「不知道」，不要说「应该会吧」；未知不等于不会，也不等于要从零开始。
7. 用户说的和他自己之前说的矛盾时，指出来并问一句以哪个为准，不要自己挑一个。
8. 一次只问一个问题，并且说明为什么问这个；用户说不清楚或不想答时，跳过这一维，按已有信息继续。
9. 用户不认同判定时，按他的说法重排，并说明改了什么。
10. 路线里的阶段如果被跳过，要给出理由（哪个维度已经到什么档位）；没确认的维度不允许跳过。
11. 解释概念用 guide_explain；查不到就直说概念库里没有，不要自己编定义。

通用规则：
12. 给命令前确认它适用于用户的操作系统。
13. 要执行安装/卸载类命令时，先把命令和影响告诉用户，等用户明确说要执行再调用 env_fix 并传 approve_system=true。
14. 复现结论要分清程度：卡数不够、显存差 6 倍以上属于物理阻塞，直说要换算力；版本不符属于装一下就好，别夸大。
15. 引用复现证据时带上来源（如 README.md:5）和可信度（L2 代码推断 / L3 经验推测）。
16. 仓库里没写、工具没返回的信息，直说不知道，不要编造版本号、卡数或命令。
17. 回答简洁：先结论，再步骤，命令用代码块。
"""


def _dump(payload: Any, limit: int = 6000) -> str:
    """工具结果转成模型可读的文本，并限制长度避免撑爆上下文。"""
    text = json.dumps(payload, ensure_ascii=False, indent=1)
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n...（已截断，原长 {len(text)} 字符）"


def build_tools():
    """构造全部 LocalFunction 工具（5 个环境 + 3 个复现 + 4 个引导）。"""
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

    # ---- 新手引导 ------------------------------------------------------

    @tool(
        name="guide_domains",
        description=(
            "列出可选的研究方向（系统与网络、安全、数据与数据库、理论与算法、软件与程序语言、人工智能），"
            "每个方向对四个固定维度的门槛、读哪些论文、基本功大概多少小时。"
            "用户说不清自己方向时用它给选项，不要替用户选。"
        ),
    )
    def guide_domains_tool() -> str:
        """列出研究方向及要求。"""
        return _dump(guide_domains())

    @tool(
        name="guide_assess",
        description=(
            "从用户原话推能力画像，每条结论都带原话或文件路径来源，没证据的维度标未知。"
            "维度固定四个（编程、工具、数学、科研方法），方向只改门槛、不新增维度。"
            "scan_dir 只在用户明确授权时传，用来扫描其本地 PDF/notebook/笔记/源码目录。"
        ),
    )
    def guide_assess_tool(
        text: str,
        scan_dir: str = "",
        domain: str = "",
        as_markdown: bool = True,
    ) -> str:
        """评估基础水平。

        Args:
            text: 用户原话，可跨多轮拼接
            scan_dir: 用户授权扫描的本地目录，未授权时留空
            domain: 用户已说明的方向（id 或中文名，如 systems/系统、data/数据库），留空则从 text 识别
            as_markdown: 是否附带可读表格
        """
        return _dump(
            guide_assess(
                text=text, scan_dir=scan_dir, domain=domain, as_markdown=as_markdown
            )
        )

    @tool(
        name="guide_roadmap",
        description=(
            "按方向生成分阶段学习路线，每项任务都带验收标准和预估小时数。"
            "通用主干（工程基础/读论文方法/记录表达）人人一样，方向基本功、复现哪篇论文、"
            "第一个项目完全按方向给。已达标的阶段会跳过并说明理由，未确认的维度不会被跳过。"
            "domain 留空表示方向还没定，会给一份「先试方向」的版本。"
        ),
    )
    def guide_roadmap_tool(
        text: str,
        scan_dir: str = "",
        domain: str = "",
        weekly_hours: int = 0,
        as_markdown: bool = True,
    ) -> str:
        """生成学习路线。

        Args:
            text: 用户原话，用来识别方向与推断水平
            scan_dir: 用户授权扫描的本地目录
            domain: 方向 id 或中文名，如 data/数据库；留空表示还没定
            weekly_hours: 每周可投入小时数，0 表示不设预算
            as_markdown: 是否附可读路线
        """
        return _dump(
            guide_roadmap(
                text=text,
                scan_dir=scan_dir,
                domain=domain,
                weekly_hours=weekly_hours,
                as_markdown=as_markdown,
            ),
            limit=9000,
        )

    @tool(
        name="guide_questions",
        description=(
            "按优先级列出接下来该问的问题（一次只问一个，这里给出队列并附理由）。"
            "只问编程/工具/数学/科研方法这四维，不问方向；信息够了会返回空。"
        ),
    )
    def guide_questions_tool(
        text: str,
        scan_dir: str = "",
        domain: str = "",
        limit: int = 3,
    ) -> str:
        """列出待问的问题。

        Args:
            text: 用户原话
            scan_dir: 用户授权目录
            domain: 方向 id 或中文名
            limit: 最多给几个问题
        """
        return _dump(
            guide_questions(text=text, scan_dir=scan_dir, domain=domain, limit=limit)
        )

    @tool(
        name="guide_explain",
        description=(
            "解释一个概念：一句话定义、类比、常见误解、前置概念、下一步、自测问题。"
            "概念库查不到时会返回候选，不要自己编定义。"
        ),
    )
    def guide_explain_tool(term: str, dimension: str = "") -> str:
        """解释概念。

        Args:
            term: 概念名，如「三次握手」「栈溢出」「反向传播」「venv」
            dimension: 可选，限定维度 coding/tools/math/research
        """
        return _dump(guide_explain(term=term, dimension=dimension))

    return [
        env_check_tool,
        env_guidance_tool,
        env_explain_tool,
        env_plan_tool,
        env_fix_tool,
        repro_scan_tool,
        repro_diff_tool,
        repro_report_tool,
        guide_domains_tool,
        guide_assess_tool,
        guide_roadmap_tool,
        guide_questions_tool,
        guide_explain_tool,
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
        description="计算机科研入门助手：环境诊断与修复、论文复现可行性、新手基础评估与学习路线",
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
