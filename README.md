# 研途 · 计算机科研入门 Agent

面向计算机专业本科生 / 低年级研究生的科研入门 agent，基于 [openJiuwen](https://openjiuwen.com) agent-core 构建。

当前完成三个能力模块：

1. **环境配置**（可运行的代码 + 技能）：检测 → 指引 → 可执行安装
2. **论文复现可行性**（可运行的代码）：从我有什么 → 论文要什么 → 差在哪
3. **新手引导**（可运行的代码 + 知识库）：基础评估 → 学习路线 → 概念解释

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

`research_agent/agent_core.py` 把三类能力注册成一个 ReAct agent（`yantu_agent`），共 12 个工具：

| 分组 | 工具 | 作用 |
| --- | --- | --- |
| 环境 | `env_check` | 体检本机环境，返回真实状态与就绪度 |
| 环境 | `env_guidance` | 按「阻塞 > 注意 > 建议」列出问题与步骤 |
| 环境 | `env_explain` | 解释单个检测项的原因与修法 |
| 环境 | `env_plan` | 生成修复命令清单（不执行） |
| 环境 | `env_fix` | 执行一条修复命令，`system` 级需明确同意 |
| 复现 | `repro_scan` | 从论文仓库提取环境需求（带文件名:行号） |
| 复现 | `repro_diff` | 与本机比对，给四档结论与差距矩阵 |
| 复现 | `repro_report` | 生成完整的 Markdown 可行性报告 |
| 引导 | `guide_assess` | 从原话（可加授权目录）出六维能力画像 |
| 引导 | `guide_roadmap` | 生成分阶段路线，每项任务带验收标准 |
| 引导 | `guide_explain` | 解释概念：定义 / 类比 / 误解 / 前置 / 下一步 |
| 引导 | `guide_tracks` | 列出可选研究方向 |

实现方式：

- 用 `@tool` 把纯函数变成 `LocalFunction`，schema 由函数签名与 docstring 推导
- `ReActAgent(card=AgentCard(...))` + `ReActAgentConfig` 配置模型与系统提示词
- `agent.ability_manager.add_ability(t.card, t)` 注册工具

系统提示词里写死了几条行为约束，避免模型即兴发挥：

- 问「这篇论文的代码我能不能跑」→ 先调 `repro_diff`，不许凭经验判断
- 引用复现结论必须带来源（`README.md:5`）和可信度（L2/L3）；仓库里没写的，直说没写
- 判断用户水平只能依据他自己说的话或明确授权的目录，不许猜路径、扫盘
- 没有证据的维度说「还不确定」，不按 0 分当作「不会」；概念库查不到就直说没有

已验证：agent 构建成功，12 个工具全部注册进 `ability_manager` 与 `resource_mgr`：

```
names = ['env_check', 'env_guidance', 'env_explain', 'env_plan', 'env_fix',
         'repro_scan', 'repro_diff', 'repro_report',
         'guide_assess', 'guide_roadmap', 'guide_explain', 'guide_tracks']
```

工具函数本体分别在 `tools/env_tools.py`、`tools/repro_tools.py` 与 `tools/guide_tools.py`，都是纯 Python，可脱离框架单独测试。

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

### 在 agent 里用

复现能力已经接进 agent，不需要记命令行：

> **你**：我 clone 了一篇论文的代码在 `D:\papers\xxx`，我这台机器能跑吗？
>
> **agent**：调 `repro_diff` → 读差距矩阵 → 回答「能/不能，卡在哪，要花多少钱或改什么」

工具层的三条约定：

- 返回 `dict` 且**不抛异常**，出错时把原因和 `hint` 写进返回值，模型能读懂并转述
- 传入 URL 会被拒绝并提示先 `git clone`（扫描需要本地文件）
- 长文本有截断上限，`repro_report` 返回的 Markdown 限 8000 字符，超出会标 `truncated`

### 边界

- 只解析仓库文件，**不读论文 PDF**，所以「论文说需要 4 张 V100」这类信息接不进来
- GPU 显存从型号折算（A100 → 40GB）时标 L3，A100 实际有 40/80 两版，可能偏差
- `torch` 之外只对 `tensorflow` / `jax` 等做了版本提取，判定侧目前只探测了 torch 的安装情况
- 不做依赖求解（`pip` 冲突、conda 环境解析交给专门工具）

## 新手引导

新手最先卡住的往往不是「不知道学什么」，而是「不知道自己现在在哪」。所以这块的顺序是：先定位，再给路线，概念卡按需补。

```
基础评估 → 你现在在哪（六维：Python / 命令行 / 数学 / ML / DL / 科研流程）
学习路线 → 从这到能干活的路（每项任务带验收标准）
概念解释 → 路上每个卡点的翻译（定义 / 类比 / 误解 / 前置 / 下一步）
```

### 评估只认两种输入

| 输入 | 来源 | 标成 | 局限 |
| --- | --- | --- | --- |
| 自述 | 用户自己说的话 | 自述 | 可能高估 |
| 文件 | 用户显式授权的目录：PDF / notebook / 笔记 | 文件 | 只证明「材料在」，不证明「读过」 |

没有问卷、没有判题，也**不主动扫盘**。自述和文件对不上时（比如自称 DL 熟练，目录里只有报错日志），把它作为「对撞」摆出来，不替用户下结论：

```
维度               档位    置信  来源        证据                        为什么这么判
深度学习与框架     从零    中    自述+文件    「也没碰过深度学习」 等 3 条  没碰过深度学习，路线前段要先…
科研流程           能干活  低    文件 papers/attention.pdf   「1 篇论文旁边有同名笔记…」  读过并留下笔记，但只作旁证
```

### 可解释性写成代码，不是写成文档

1. **每条结论都能指回证据**：原话逐字保留，文件证据带路径；不给孤立总分
2. **来源分列**：自述 / 文件 / 推断分开显示，不混算成一个分数
3. **没采集 ≠ 不会**：未知标 `UNKNOWN`，不按 0 分计入
4. **矛盾就不定档**：两句话指向的水平差太远时，标待确认并直接问
5. **匹配限定在短句内**：`我没用过命令行，也没碰过torch` 不会把命令行那句算成深度学习的证据

### 路线：跳过要讲理由

```
阶段 1 工具与工程     终端 / 虚拟环境 / Git / SSH（可直接复用 env-setup）
阶段 2 数学与 ML 底子  线代·概率·梯度 + 一门 ML 课 + 手写反向传播
阶段 3 复现一篇小论文  跑通 → 对上数字 → 改一处 → 记录
阶段 4 读论文的方法    综述画脉络 + 三遍法 + 经典 3 篇 + 近三年 5 篇
阶段 5 方向分岔        CV / NLP-LLM / 系统 / 数据推荐 / 理论 / AI4Science
阶段 6 改进与实验管理  消融 / 多种子 / 实验追踪
阶段 7 写作与开源      报告 / 仓库 / 讲一次
```

- 已具备的阶段跳过，并写明依据：「你的命令行已是『能干活』档（要求 ≥有基础）」
- 维度未确认**不跳过**，不默认你会
- 每项任务必须有验收标准，没有验收的条目在生成时就被丢掉
- 每个方向 3 篇阅读清单，按「入门 / 必读 / 进阶」标注，并注明「不必复现全部」

```bash
# 从对话出画像（可选授权扫描目录）
python cli.py guide profile "我大一，Python 只会 print，没用过命令行" --scan ~/Desktop/papers

# 生成路线，指定方向，落成 Markdown
python cli.py guide roadmap "我大一，Python 只会 print" --track cv --out plan.md

# 解释概念
python cli.py guide explain "反向传播"
python cli.py guide tracks
```

### 概念库

35 张概念卡，路线里的「相关概念」直接指回卡片。检索顺序：id → 名称 → 别名 → 句内别名 → 模糊候选；**查不到就直说没有**，不用模型生成定义：

```
概念库里没有「量子纠缠」这张卡。
也没有足够接近的条目。可以换个说法，或者告诉我上下文，我按你的场景解释。
```

### 边界

- 不解析 PDF 正文，论文是否真读过只看「有没有同名笔记」这种旁证（正文抽取在 `document-processing` 分支）
- 评估靠对话里的关键词与固定规则，说法超出规则覆盖范围时会标未知，而不是硬猜
- 路线内容是通用入门路径，不替代导师的方向判断；预计时长是估计值，不是承诺

### 已验证

43 项回归，重点盯可解释性：否定句不被读成正面证据、矛盾不硬猜、跨短句不串维度、
文件证据不定档、跳过阶段必须带理由、每项任务都有验收、概念交叉引用无悬空。

## 目录结构

```
yantu/
├── openjiuwen/                    # openJiuwen agent-core 源码（git clone，editable 安装）
├── research_agent/
│   ├── knowledge/                 # 知识库
│   │   ├── env_kb.json            # 环境条目：检测项 → 原因 → 步骤 → 验证
│   │   ├── python_env.md          # Python 与包管理详述
│   │   ├── git.md                 # Git 与代码托管详述
│   │   ├── gpu_dl.md              # GPU 与深度学习详述
│   │   ├── guide_signals.json     # 对话证据规则（40 条，正 28 / 负 12）
│   │   ├── roadmap.json           # 学习路线的阶段模板与六个方向
│   │   └── concepts.json          # 概念卡（35 张，含前置/下一步关系）
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
│   ├── guide/                     # 新手引导
│   │   ├── spec.py                # 契约：六维 / 四档 / 证据来源 / 画像
│   │   ├── signals.py             # 对话 → 证据（否定优先、短句内匹配）
│   │   ├── profile.py             # 证据 → 画像（矛盾不猜、文件不定档）
│   │   ├── materials.py           # 授权目录 → 文件证据（PDF/notebook/笔记）
│   │   ├── roadmap.py             # 画像 → 阶段路线（跳过给理由、任务带验收）
│   │   ├── concepts.py            # 概念检索（别名 / 句内 / 模糊候选）
│   │   ├── probe.py               # 缺信息时一次只问一句
│   │   └── render.py              # 可核对表格 / 路线 Markdown
│   ├── tools/env_tools.py         # 纯 Python 工具函数（可单独测试）
│   ├── tools/repro_tools.py       # 复现工具函数（agent 用）
│   ├── tools/guide_tools.py       # 引导工具函数（agent 用）
│   ├── agent_core.py              # openJiuwen 适配层（ReActAgent + 12 个工具）
│   ├── agent.py                   # 问答层（有 Key 走模型，无 Key 走检索）
│   └── llm.py                     # OpenAI 兼容客户端
├── skills/env-setup/              # 可被 JiuwenSwarm 直接调用的技能包
│   ├── SKILL.md
│   └── scripts/env_setup.py
├── tests/
│   ├── fixtures/                  # 回归样本（论文仓库 / 学习材料目录）
│   ├── test_repro.py              # 17 项：复现判定
│   └── test_guide.py              # 43 项：画像 / 材料 / 路线 / 概念 / 工具
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
- [x] 复现能力接进 agent 工具层（repro_scan / repro_diff / repro_report）
- [x] 基础评估：对话式六维画像，每条结论指回原话或文件路径
- [x] 学习路线：阶段化路线 + 验收标准 + 六个方向分岔
- [x] 概念解释：35 张概念卡，别名与句内检索，查不到直说没有
- [ ] 论文 PDF 解析：把正文里的「we train on 8×A100」也纳入需求（当前只认代码）
- [ ] 评估接入 PDF 正文：读过哪些论文不再只靠文件名（依赖 document-processing 分支）
- [ ] 路线进度追踪：勾选任务、记录实际耗时、按完成度重排
- [ ] 复现能力做成 JiuwenSwarm 技能（现在只有 CLI + agent 工具）
- [ ] 环境快照对比（两次检测的 diff，看修复是否生效）
- [ ] 文献阅读模块：arXiv 检索、论文速读、笔记沉淀
- [ ] 实验管理：跑通 baseline、记录超参、复现他人结果
- [ ] openJiuwen 多 agent 协作：环境 agent、文献 agent、引导 agent 分工
