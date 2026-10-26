---
name: env-setup
version: 1.0.0
author: 研途
license: MIT
category: 开发环境 / 科研入门
platforms: [jiuwenswarm, claude-code, cursor]
description: 计算机科研入门的环境配置助手——检测本机 Python/pip/conda/venv、Git 与 SSH、NVIDIA 驱动/CUDA/PyTorch 的真实状态，按「阻塞＞注意＞建议」给出分平台处理步骤，并可在授权后执行安装命令。适用于：科研环境从零搭建、pip 装不上包、torch.cuda.is_available() 为 False、GitHub clone 失败、虚拟环境混乱、CUDA 与驱动版本不匹配、显存不足等场景。触发词：配置环境、环境配置、搭环境、科研环境、装环境、pip 报错、装不上、import 不到、torch cuda 不可用、CUDA 版本、显卡驱动、conda、虚拟环境、SSH 配好了吗、体检环境、检查环境。
agent_created: true
---

# 科研环境配置助手

面向计算机科研新手的环境诊断与修复。核心原则：**先拿到事实，再给建议**，命令必须能在当前系统上直接跑。

## 何时使用

用户在以下场景时应主动调用本技能：

- 刚开始做科研，要从零搭开发环境
- `pip install` 报错、装了包但 `import` 不到
- `torch.cuda.is_available()` 返回 False
- 不确定 CUDA 版本、显卡驱动、PyTorch 版本之间怎么匹配
- GitHub clone 不下来、SSH 拉代码要输密码
- 虚拟环境/conda 用不明白，怕污染系统 Python
- 训练时报 OOM，不知道 6GB/8GB 显存能干什么

## 命令

所有命令通过项目里的脚本执行。`python` 用系统解释器即可，检测逻辑只依赖标准库：

```bash
# 1. 体检：输出完整报告（先跑这个）
python <skill_dir>/scripts/env_setup.py check

# 2. 精简版：只列要处理的问题和步骤
python <skill_dir>/scripts/env_setup.py guidance

# 3. 结构化输出，便于自己组织回答
python <skill_dir>/scripts/env_setup.py check --json

# 4. 解释单项：为什么重要、怎么修、怎么验证
python <skill_dir>/scripts/env_setup.py explain python.version
python <skill_dir>/scripts/env_setup.py explain torch.installed

# 5. 修复计划：只列命令，不执行
python <skill_dir>/scripts/env_setup.py plan --risk config

# 6. 执行某条命令（默认 dry-run）
python <skill_dir>/scripts/env_setup.py fix --command "<来自 plan>" --dry-run
python <skill_dir>/scripts/env_setup.py fix --command "<来自 plan>" --yes
```

可选参数：`check --no-network` 跳过连通性探测（更快），`check --no-torch` 跳过 PyTorch 探测。

## 检测项一览

| 领域 | 检测项 |
| --- | --- |
| Python | 版本区间、解释器构建（标准/free-threaded）、多解释器并存、pip 可用性与归属、pip 镜像源、venv 支持、conda |
| Git | 安装与版本、user.name/user.email、SSH 公钥、代码托管连通性 |
| GPU | NVIDIA GPU 与驱动、驱动支持的 CUDA 版本、CUDA Toolkit(nvcc)、PyTorch 是否安装、是否 CPU 版、CUDA 版本是否超出驱动、显存大小 |

## 三级风险模型

| 级别 | 含义 | 例子 | 执行条件 |
| --- | --- | --- | --- |
| `safe` | 只读、幂等 | `nvidia-smi`、`py -0p`、`python -c "..."` | 可直接执行 |
| `config` | 改用户级配置，可回退 | `pip config set`、`git config --global` | 需告知用户后再执行 |
| `system` | 安装/卸载软件 | `winget install`、`pip uninstall` | **必须用户明确同意**，加 `--yes` |

**重要**：向用户执行 `system` 级命令前，必须先把「要装什么、会影响什么」说清楚，得到明确同意后再加 `--yes`。不要自行决定安装软件。

## 回答用户时的规矩

1. 先跑 `check` 或 `guidance` 拿事实，**不要凭印象回答版本号或命令**。
2. 命令要适配当前系统：Windows 不出现 `sudo apt`，Linux 不出现 `winget`。
3. 先给结论，再给步骤；命令用代码块，一步一条。
4. 检测结果里没有的信息，直说不确定。
5. 已知的常见陷阱要主动提示，例如：
   - Windows 上 `py` 的默认解释器可能是 free-threaded（`python3.13t.exe`），装科学计算库会失败
   - 有 NVIDIA 卡但装了 CPU 版 PyTorch，`cuda.is_available()` 恒为 False
   - PyTorch 的 CUDA 轮子不在 PyPI 镜像里，必须走 `download.pytorch.org`

## 知识库

处理建议来自结构化知识库，不是模型自由发挥：

- `<project>/research_agent/knowledge/env_kb.json` —— 20 条条目，含原因、分平台步骤、验证命令
- `<project>/research_agent/knowledge/python_env.md` —— Python 与包管理详述
- `<project>/research_agent/knowledge/git.md` —— Git 与代码托管详述
- `<project>/research_agent/knowledge/gpu_dl.md` —— GPU、CUDA 与 PyTorch 详述

## 项目位置

默认 `~/Desktop/yantu`。若项目被移动，设置环境变量：

```bash
set YANTU_HOME=D:\path\to\yantu
```

## 与 openJiuwen agent 的关系

本技能是**规则驱动**的，不需要模型 API Key 也能完整工作。

项目里还有一层 openJiuwen 封装（`research_agent/agent_core.py`），把同样的能力注册成 5 个 ReAct 工具（`env_check` / `env_guidance` / `env_explain` / `env_plan` / `env_fix`），配好 `DEEPSEEK_API_KEY` 后可让 agent 自主决定调用顺序。那是可选增强，不是本技能的前提。
