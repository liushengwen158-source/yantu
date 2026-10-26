# 研途 · 计算机科研入门 Agent

面向计算机专业本科生 / 低年级研究生的科研入门 agent，基于 [openJiuwen](https://openjiuwen.com) agent-core 构建。

当前阶段完成第一个能力模块：**环境配置**（检测 → 指引 → 可执行安装）。

## 它解决什么问题

科研新手最早卡住的地方不是读不懂论文，而是环境跑不起来：`pip install` 报错、装完 import 不到、`torch.cuda.is_available()` 是 False、GitHub clone 不下来。这些问题的共同点是——**报错信息不告诉你真实原因**。

这个模块做三件事：

1. **检测**：把本机 Python / pip / conda / git / SSH / NVIDIA 驱动 / CUDA / PyTorch 的真实状态摸清楚
2. **诊断**：按「阻塞 / 注意 / 建议」分级，结合知识库给出**这个平台上能直接跑**的处理步骤
3. **执行**：分级授权，`config` 级改配置、`system` 级装软件，`system` 必须显式确认才动手

## 快速开始

### 环境准备

本项目的检测逻辑只依赖标准库，用系统 Python 即可跑；跑 agent（需要模型调用）
时用项目虚拟环境：

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install openjiuwen -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 命令行

```bash
# 体检（人类可读报告）
python cli.py check

# 结构化输出（给 agent / 脚本用）
python cli.py check --json

# 只看要跑哪些命令，不执行
python cli.py plan

# 执行修复：默认 dry-run，system 级要加 --yes
python cli.py fix --risk config
python cli.py fix --yes

# 就具体问题提问：配了 API Key 走 ReAct agent，没配则走知识库检索
.venv/Scripts/python.exe cli.py ask "torch 装上了但 cuda 不可用"
```

### 接入模型

```bash
set DEEPSEEK_API_KEY=sk-xxx          # Windows PowerShell: $env:DEEPSEEK_API_KEY="sk-xxx"
set DEEPSEEK_MODEL=deepseek-chat     # 可选
set DEEPSEEK_BASE_URL=https://api.deepseek.com/v1   # 可选
```

不配 Key 也能用：`check` / `plan` / `fix` 是纯规则实现，`ask` 会退化为知识库检索。

## openJiuwen 集成

### 源码与安装

```bash
# 需要代理访问 GitHub
git clone --depth 1 https://github.com/openJiuwen-ai/agent-core.git openjiuwen

# 让框架从本地源码加载（改源码后立即生效，无需重装）
.venv/Scripts/python.exe -m pip install -e ./openjiuwen --no-deps
```

验证加载路径：

```bash
.venv/Scripts/python.exe -c "import openjiuwen.core.single_agent.agents.react_agent as m; print(m.__file__)"
# 应输出 ...\yantu\openjiuwen\openjiuwen\core\single_agent\agents\react_agent.py
```

### 适配层

`research_agent/agent_core.py` 把环境能力注册成一个 ReAct agent：

- 用 `@tool` 把 5 个函数变成 `LocalFunction`（env_check / env_guidance / env_explain / env_plan / env_fix）
- `ReActAgent(card=AgentCard(...))` + `ReActAgentConfig` 配置模型与系统提示词
- `agent.ability_manager.add_ability(t.card, t)` 注册工具

已验证：agent 构建成功，5 个工具全部注册进 ability_manager 与 resource_mgr，
且模块从本地 clone 源码加载。

工具函数本体在 `research_agent/tools/env_tools.py`，是纯 Python，可脱离框架单独测试。

## 作为技能使用

`skills/env-setup/` 是一个自包含技能包，已安装到 JiuwenSwarm：

```
C:\Users\<你的用户名>\.jiuwenswarm\agent\workspace\skills\env-setup
```

安装后，在对话里提到「配置环境」「装不上包」「cuda 不可用」等场景时可直接调用，
命令形如：

```bash
python <skill_dir>/scripts/env_setup.py check
python <skill_dir>/scripts/env_setup.py guidance
python <skill_dir>/scripts/env_setup.py explain torch.installed
```

技能脚本通过 `YANTU_HOME` 环境变量定位项目（默认 `~/Desktop/yantu`），
能力本体只有一份，不会与项目代码漂移。


## 目录结构

```
yantu/
├── openjiuwen/                    # openJiuwen agent-core 源码（git clone，editable 安装）
├── research_agent/
│   ├── knowledge/                 # 环境配置知识库
│   │   ├── env_kb.json            # 结构化条目：检测项 → 原因 → 步骤 → 验证
│   │   ├── python_env.md          # Python 与包管理详述
│   │   ├── git.md                 # Git 与代码托管详述
│   │   └── gpu_dl.md              # GPU 与深度学习详述
│   ├── envsetup/
│   │   ├── models.py              # 数据结构（Status / CheckResult / EnvReport）
│   │   ├── probes.py              # 只读探测原语
│   │   ├── detector.py            # 判定逻辑 + 知识库加载
│   │   ├── advisor.py             # 指引生成 + 报告渲染
│   │   └── installer.py           # 分级执行
│   ├── tools/env_tools.py         # 纯 Python 工具函数（可单独测试）
│   ├── agent_core.py              # openJiuwen 适配层（ReActAgent + 5 个工具）
│   ├── agent.py                   # 问答层（有 Key 走模型，无 Key 走检索）
│   └── llm.py                     # OpenAI 兼容客户端
├── skills/env-setup/              # 可被 JiuwenSwarm 直接调用的技能包
│   ├── SKILL.md
│   └── scripts/env_setup.py
└── cli.py                         # 命令行入口
```

## 设计说明

### 为什么知识和判定分离

`probes.py` 只回答「是什么」，`detector.py` 只回答「好不好」，`env_kb.json` 只回答「怎么办」。

这样调整建议文案不用改代码，新增检测项不用碰知识库，两者可以独立演进。知识库也是 agent 回答用户问题的**唯一事实来源**，避免模型自由发挥编出跑不通的命令。

### 三级风险模型

| 级别 | 含义 | 例子 | 默认行为 |
| --- | --- | --- | --- |
| `safe` | 只读、幂等 | `nvidia-smi`、`git ls-remote` | 可直接执行 |
| `config` | 改用户级配置，可回退 | `pip config set`、`git config --global` | 需用户确认 |
| `system` | 安装/卸载软件、改系统 | `winget install`、`pip uninstall` | 必须 `--yes` |

带 `&&`、`|`、`;` 的命令一律按 `system` 处理，防止复合命令绕过风险判定。

### 平台过滤

知识库每条步骤带 `platform` 字段（`win` / `linux` / `any`），生成建议时按当前系统过滤。Windows 用户不会看到 `sudo apt install`。

## 已验证的真实案例

在本机（Windows 11 + RTX 4050 6GB）跑出的结果：

- `py` 的**默认**解释器是 `python3.13t.exe`（free-threaded，GIL 关闭），而 `python` 指向标准版——直接 `pip install numpy` 会退回源码编译失败，检测器能提前发现
- GitHub 直连 TLS 握手失败，检测器会把「代码托管连通性」标成注意并建议国内镜像
- 显存 6GB 会给「入门够用、大模型会 OOM」的提示，而不是简单标 OK

## 后续路线

- [ ] 环境快照对比（两次检测的 diff，看修复是否生效）
- [ ] 文献阅读模块：arXiv 检索、论文速读、笔记沉淀
- [ ] 实验管理：跑通 baseline、记录超参、复现他人结果
- [ ] openJiuwen 多 agent 协作：环境 agent 与文献 agent 分工
