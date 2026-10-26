# Python 与包管理

科研入门第一件正经事：让 `python` 和 `pip` 指向同一个你控制得住的环境。

## 安装与版本

主流科研库对 Python 版本有硬约束。当前（2026 年）建议：

| 版本 | 评价 |
| --- | --- |
| 3.9 及以下 | 不要用。PyTorch 新版已停止支持，很多库装不上 |
| 3.10 | 可用，但逐渐被新轮子跳过 |
| 3.11 / 3.12 | **推荐**，轮子覆盖最全，本知识库默认 3.12 |
| 3.13 | 可用，少数库的轮子还在补齐 |
| 3.14+ | 太新，科学计算栈轮子覆盖不全，踩坑多 |
| 3.13t（free-threaded） | 实验性，C 扩展支持差，**不要**当默认解释器 |

openJiuwen agent-core 的硬性要求是 Python 3.11–3.13，超出范围会安装失败。

Windows 上推荐用 winget：

```bash
winget install -e --id Python.Python.3.12
```

安装向导里**一定勾选 "Add python.exe to PATH"**，否则命令行找不到 `python`。

Linux（Ubuntu/Debian）：

```bash
sudo apt update
sudo apt install -y python3 python3-pip python3-venv
```

## 多解释器共存

Windows 上装了多个 Python 后，`python` 可能指向 A，`pip` 指向 B。症状是 `pip install xxx` 成功但 `import xxx` 报 ModuleNotFoundError。

排查三连：

```bash
py -0p                      # 列出所有解释器及路径
python -m pip --version     # pip 绑定的是哪个 python
where python                # PATH 里 python 的解析顺序（Linux 用 which -a python）
```

习惯用 `python -m pip` 而不是裸 `pip`，能规避绝大多数串味问题。

判断当前是不是 free-threaded 构建：

```bash
python -c "import sys; print(sys.abiflags)"   # 含 't' 即为 3.13t
```

## pip 异常

pip 丢失时用标准库自带的 ensurepip 引导：

```bash
python -m ensurepip --upgrade
```

再不行下载官方脚本：

```bash
curl -O https://bootstrap.pypa.io/get-pip.py
python get-pip.py
```

pip 版本太旧会导致依赖解析失败、装不上新轮子：

```bash
python -m pip install --upgrade pip
```

## 虚拟环境

**每个项目一个环境**是科研的好习惯：论文代码往往依赖特定版本的库，全局安装必然互相覆盖。

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Linux / macOS
python -m pip install -U pip
```

退出用 `deactivate`。导出可复现的依赖清单：

```bash
python -m pip freeze > requirements.txt
python -m pip install -r requirements.txt
```

Linux 上 `python -m venv` 报 `ensurepip is not available`，说明缺 `python3-venv` 包：

```bash
sudo apt install -y python3-venv
```

## conda

conda 不是必需品，但有两个场景值得用：

- 需要多个 CUDA 版本共存（不同项目要用不同 torch）
- 复现别人的 `environment.yml`

轻量选择是 Miniforge（默认 conda-forge 源，无 Anaconda 商用授权问题）：

```bash
winget install -e --id CondaForge.Miniforge3      # Windows
```

```bash
# Linux
curl -L -O https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh
```

常用命令：

```bash
conda create -n research python=3.12 -y
conda activate research
conda env export > environment.yml
```

不要混用 conda 和 pip 装同一个包，先 conda 装不了的再 pip 装，且都装完后再 pip 装会在卸载时留下残留。

## 镜像源

国内直连 PyPI 下载 torch 这类大包经常只有几十 KB/s。配置镜像：

```bash
python -m pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
python -m pip config set global.timeout 120
```

备选镜像：

- 阿里云 `https://mirrors.aliyun.com/pypi/simple/`
- 中科大 `https://pypi.mirrors.ustc.edu.cn/simple/`

查看当前生效配置：

```bash
python -m pip config list
```

注意：PyTorch 的 CUDA 版轮子不在 PyPI 镜像里，必须走官方源：

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cu124
```

## 常见报错对照

| 报错 | 原因 | 处理 |
| --- | --- | --- |
| `No module named xxx` 但装过了 | pip 与 python 不是同一环境 | 用 `python -m pip install` |
| `Microsoft Visual C++ 14.0 is required` | 装的包只有源码，需要编译器 | 优先找有 wheel 的版本/换 Python 版本；否则装 VS Build Tools |
| `ERROR: Could not build wheels` | 同上，或该版本无对应轮子 | 降 Python 版本到 3.12 |
| `Read timed out` | 网络 | 配镜像 + `--timeout 120`，或 `--default-timeout=1000` |
| `Permission denied` | 装到了系统目录 | 用虚拟环境，或 `--user` |
