# 研途 · 计算机科研入门 Agent

面向计算机专业本科生 / 低年级研究生的科研入门 agent，基于 [openJiuwen](https://openjiuwen.com) agent-core 构建。

当前完成两个能力模块：

1. **环境配置**（可运行的代码 + 技能）：检测 → 指引 → 可执行安装
2. **论文复现可行性**（可运行的代码）：从我有什么 → 论文要什么 → 差在哪

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

# 论文复现：从配套代码里提取环境需求
python cli.py paper scan /path/to/repo

# 与本机环境比对，输出差距矩阵
python cli.py paper diff /path/to/repo

# 输出完整可行性报告（Markdown）
python cli.py paper report /path/to/repo --out report.md
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


## 论文复现可行性

读一篇论文想跑通它的代码，第一步不是 `pip install -r`，而是搞清楚**它要什么、你有什么、差在哪、差的部分能不能补**。这个模块做这件事。

### 两个对称对象

```
PaperEnvSpec      论文仓库要什么        ← scan_repo()
LocalEnvProfile   本机有什么            ← collect_profile()
        ↓ diff()
FeasibilityReport 差在哪、能不能补
```

### 需求从哪来（只采信确定性证据）

| 来源文件 | 提取内容 | 证据级别 |
| --- | --- | --- |
| `Dockerfile` | `FROM nvidia/cuda:X.Y` → CUDA 版本、Python 版本、`pip install` 里的框架版本 | L2 |
| `environment.yml` | `python=x.y`、`torch=x.y`、`cudatoolkit` | L2 |
| `requirements.txt` | `torch==2.4.0` 等框架版本 | L2 |
| `pyproject.toml` | `requires-python` | L2 |
| `.python-version` / `runtime.txt` | Python 版本 | L2 |
| `README` | `8×A100` → 卡数；型号 → 显存（型号折算属推测） | L2 / L3 |

每条需求都带 `文件名:行号`。**没有来源的需求直接丢弃**，不进报告——这是防编造的地基。
论文正文里的环境描述尚未接入（需要 PDF 解析），报告会明确写出这个边界。

### 四档判定

| 判定 | 含义 | 典型触发 |
| --- | --- | --- |
| `BLOCK` | 物理上不可能 | 要 8 卡只有 1 卡；要 40GB 显存只有 6GB（差 6 倍以上）；需求 CUDA 高于驱动支持 |
| `DEGRADE` | 改配置能跑 | 显存差 2～6 倍，减小 batch / 开 amp / LoRA 可弥补 |
| `ADD` | 装一下就好 | 框架未装、版本不符、Python 版本不符 |
| `OK` | 满足 | — |
| `UNKNOWN` | 信息不足 | 未探测的框架，不假装满足 |
| `CONFLICT` | 需求自相矛盾 | `torch==2.4` 但 Dockerfile 指定 CUDA 11.7（torch 2.4 只有 cu118/121/124） |

最终结论取最严重的一项。

### 建议命令的一致性

安装建议会**对齐仓库声明的 CUDA**，而不是本机最高版本。
例如仓库声明 CUDA 11.8 + torch 2.4.0，建议就是 `cu118`；不会出现「CUDA 说 11.8、torch 让装 cu124」的矛盾。

### 已验证

```
$ python cli.py paper diff tests/fixtures/repo_a

需求项          论文要求  本机              判定       证据
--------------  --------  ----------------  ---------  ----------------------
gpu.count       >= 8 卡   1 卡              BLOCK      README.md:5 (L2)
gpu.vram        >= 40 GB  6 GB              BLOCK      README.md:5 (L3)
cuda            == 11.8   12.8              ADD        Dockerfile:1 (L2)
torch.version   == 2.4.0  2.11.0+cu128      ADD        Dockerfile:3 (L2)
python.version  == 3.9.7  3.13.4            ADD        .python-version:1 (L2)
cuda.required   需要      1 卡 / CUDA 12.7  OK         Dockerfile:1 (L2)

结论：阻塞 2 · 冲突 0 · 需降级 0 · 可补齐 3 · 待确认 0 · 满足 1
可行性：阻塞（BLOCK）—— 存在物理阻塞（gpu.count、gpu.vram），需更换算力
```

回归测试（12 项，含版本约束、需求提取、冲突检测、判定规则）：

```bash
python -m unittest discover -s tests -v
```

### 边界

- 只解析仓库文件，**不读论文 PDF**，所以「论文说需要 4 张 V100」这类信息接不进来
- GPU 显存从型号折算（A100 → 40GB）时标 L3，A100 实际有 40/80 两版，可能偏差
- `torch` 之外只对 `tensorflow` / `jax` 等做了版本提取，判定侧目前只探测了 torch 的安装情况
- 不做依赖求解（`pip` 冲突、conda 环境解析交给专门工具）

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
│   ├── repro/                     # 论文复现可行性
│   │   ├── spec.py                # 数据契约（需求 / 画像 / 报告 / 证据）
│   │   ├── version.py             # 版本约束判断（零依赖）
│   │   ├── scanner.py             # 仓库 → 环境需求（带文件名行号）
│   │   ├── profile.py             # 本机 → 画像（含磁盘/内存/容器）
│   │   ├── diff.py                # 四档判定 + 修复建议
│   │   └── render.py              # 差距矩阵 / Markdown 报告
│   ├── tools/env_tools.py         # 纯 Python 工具函数（可单独测试）
│   ├── agent_core.py              # openJiuwen 适配层（ReActAgent + 5 个工具）
│   ├── agent.py                   # 问答层（有 Key 走模型，无 Key 走检索）
│   └── llm.py                     # OpenAI 兼容客户端
├── skills/env-setup/              # 可被 JiuwenSwarm 直接调用的技能包
│   ├── SKILL.md
│   └── scripts/env_setup.py
├── tests/
│   ├── fixtures/                  # 回归用的仓库样本
│   └── test_repro.py              # 12 项单元测试
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

### 为什么复现需求必须带行号

模型很容易"记得"某篇论文是 8 卡训练，但记得不等于对。所以复现模块里每一条需求都必须能指回 `文件名:行号`，否则直接丢弃，宁可在报告里说"信息不足"。

同理，判定只说两种话：「本机 6GB，需要 40GB，差 6.7 倍」和「无法判断」。前者可核对，后者诚实。

## 已验证的真实案例

在本机（Windows 11 + RTX 4050 6GB）跑出的结果：

- `py` 的**默认**解释器是 `python3.13t.exe`（free-threaded，GIL 关闭），而 `python` 指向标准版——直接 `pip install numpy` 会退回源码编译失败，检测器能提前发现
- GitHub 直连 TLS 握手失败，检测器会把「代码托管连通性」标成注意并建议国内镜像
- 显存 6GB 会给「入门够用、大模型会 OOM」的提示，而不是简单标 OK

## 后续路线

- [x] 论文复现可行性：从配套代码提取环境需求，与本机比对给出四档结论
- [ ] 论文 PDF 解析：把正文里的「we train on 8×A100」也纳入需求（当前只认代码）
- [ ] 复现需求接进 agent 工具层（现在只有 CLI）
- [ ] 环境快照对比（两次检测的 diff，看修复是否生效）
- [ ] 文献阅读模块：arXiv 检索、论文速读、笔记沉淀
- [ ] 实验管理：跑通 baseline、记录超参、复现他人结果
- [ ] openJiuwen 多 agent 协作：环境 agent 与文献 agent 分工
