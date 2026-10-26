# GPU 与深度学习

这一块是新手踩坑最多的地方，核心只有一个关系：**驱动版本 ≥ torch 编译时的 CUDA 版本**。

## 显卡与驱动

先确认有没有 NVIDIA 卡：

```bash
nvidia-smi
```

输出里看三项：

- `Driver Version`：驱动版本
- `CUDA Version`：**这个驱动最高支持的 CUDA 版本**（不是已安装的 CUDA）
- `Memory Total`：显存

没有 `nvidia-smi` 命令说明两种情况之一：没 NVIDIA 卡，或驱动没装。

AMD 卡走 ROCm，不在本知识库范围；Intel 核显和 Apple M 系列用 MPS 后端，能跑小实验。

## CUDA 与驱动的版本关系

CUDA 12.x 系列要求驱动 ≥ 525（Linux）/ 527（Windows）。装更高版本的驱动总是安全的，向下兼容。

当前 PyTorch 常用的 CUDA 构建与最低驱动：

| torch 的 CUDA 版本 | 最低驱动（Linux） | 说明 |
| --- | --- | --- |
| cu118 | 450.80.02 | 老卡兜底选择 |
| cu121 | 530.30.02 | 兼容面广 |
| cu124 | 550.54.14 | 目前推荐默认 |
| cu126 | 560.28.03 | 较新 |
| cu128 | 570.26 | 最新，需较新驱动 |

**关键认知**：用 pip 装的 torch 自带 CUDA 运行时库，**不需要**单独安装 CUDA Toolkit。只有需要编译自定义算子（flash-attn、apex、deepspeed 的某些组件）时才要 `nvcc`。

所以看到「未安装 CUDA Toolkit」不必慌，先验证 torch：

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

三个值分别是 torch 版本、编译用 CUDA 版本、CUDA 是否可用。

## PyTorch 安装

到 pytorch.org 的 Get Started 页面选自己的系统、包管理器、CUDA 版本，它会生成对应命令。手工版：

```bash
# CUDA 12.4 版（推荐）
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# CPU 版
python -m pip install torch torchvision
```

注意：**PyTorch 官方 CUDA 轮子不在 PyPI 镜像里**，清华源只有 CPU 版和少量版本。国内下载慢时：

- 给 pip 配代理后走官方源
- 或用国内第三方镜像（版本可能滞后，注意核对）
- 或用 `--timeout 1000` 硬等

验证：

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available()); print(torch.zeros(3).cuda())"
```

## CPU 版陷阱

最常见的坑：机器有 RTX 显卡，却因为直接 `pip install torch` 装了 CPU 版，`torch.cuda.is_available()` 永远是 False。

判断：

```bash
python -c "import torch; print(torch.version.cuda)"   # 输出 None 即 CPU 版
```

处理：

```bash
python -m pip uninstall -y torch torchvision torchaudio
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

## 显存与常见 OOM

`torch.cuda.OutOfMemoryError` 的处理顺序：

1. **减小 batch size**（最有效，通常对半分）
2. **梯度累积**模拟大 batch：`loss = loss / accum; loss.backward()` 累积 N 次再 `step()`
3. **混合精度**：`with torch.autocast('cuda'):` 省一半显存
4. **梯度检查点**：`model.gradient_checkpointing_enable()`，用算力换显存
5. **清理无用张量**：`del x; torch.cuda.empty_cache()`

入门级显存需求参考：

| 任务 | 大致显存 |
| --- | --- |
| MNIST / CIFAR 小 CNN | 1–2 GB |
| ResNet-50 训练（bs=64） | 6–8 GB |
| BERT-base 微调（bs=16） | 8–12 GB |
| 7B 模型 LoRA 微调 | 16–24 GB |
| 7B 全参微调 | 80 GB 起 |

6GB 显存的卡（如 RTX 4050 Laptop）适合前两行，跑通了再上云。

## 多卡与集群

单卡放不下时先考虑租卡（AutoDL、恒源云等按小时计费，比买卡划算得多）。多卡训练的入口是：

```bash
torchrun --nproc_per_node=2 train.py
```

入门阶段不必急，先把单卡跑顺。

## 环境自检脚本

把下面存成 `check_cuda.py` 随时验证：

```python
import torch

print("torch:", torch.__version__)
print("编译 CUDA:", torch.version.cuda)
print("可用:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("设备:", torch.cuda.get_device_name(0))
    print("显存 GB:", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1))
    x = torch.randn(1000, 1000, device="cuda")
    print("矩阵乘测试:", float((x @ x).sum()))
```

`torch.version.cuda` 有值但 `is_available()` 为 False，通常是驱动太旧或装了 CPU 版。
