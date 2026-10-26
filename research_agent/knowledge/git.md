# Git 与代码托管

科研里 Git 不只是「存代码」：复现实验、管理论文版本、协作改代码、从 GitHub 拿别人的实现，都绕不开它。

## 安装

Windows：

```bash
winget install -e --id Git.Git
```

Linux：

```bash
sudo apt install -y git
```

Windows 还需处理换行符。默认 `core.autocrlf=true` 会把 LF 转成 CRLF，和 Linux 服务器、CI 产生满屏 diff：

```bash
git config --global core.autocrlf false
```

## 首次配置

```bash
git config --global user.name "你的名字"
git config --global user.email "you@example.com"
```

邮箱建议与 GitHub 账号一致，否则提交无法归属到你的主页。

其他常用配置：

```bash
git config --global init.defaultBranch main
git config --global core.quotepath false      # 中文文件名不转义
git config --global pull.rebase false
```

检查配置：

```bash
git config --global --list
```

## SSH 密钥

用 SSH 拉代码免去每次输入密码，也方便连服务器和 GPU 机器。

```bash
ssh-keygen -t ed25519 -C "you@example.com"
```

一路回车（密码可留空）。然后：

```bash
cat ~/.ssh/id_ed25519.pub
```

把输出整段复制到 GitHub → Settings → SSH and GPG keys → New SSH key。

测试：

```bash
ssh -T git@github.com
```

看到 `Hi <username>! You've successfully authenticated` 即成功。第一次会问是否信任主机，输 `yes`。

如果已有 RSA 密钥且能正常用，不必换。

## 网络问题

国内访问 GitHub 常见三种失败：

**1. TLS 握手失败**

```
fatal: unable to access 'https://github.com/...': schannel: failed to receive handshake
```

处理优先级：

- 换国内镜像：gitcode、atomgit、gitee 上常有同名仓库
- 给 git 配代理（端口按自己的代理软件改，Clash 默认 7890）

```bash
git config --global http.proxy http://127.0.0.1:7890
git config --global --unset http.proxy      # 取消
```

**2. 大仓库 clone 到一半断**

```bash
git clone --depth 1 <url>                   # 只要最新一次提交
```

需要历史时再 `git fetch --unshallow`。

**3. pip 从 git 源安装失败**

同样受网络影响。可以先 clone 到本地，再 `pip install -e ./目录`。

## 常用操作

```bash
git status                        # 看改动
git add -A                        # 暂存全部
git commit -m "feat: xxx"         # 提交
git log --oneline -10             # 看最近提交
git diff                          # 看未暂存改动
git switch -c experiment-a        # 建分支做实验
git switch main                   # 切回主分支
```

科研场景的实用习惯：

- 每个实验一个分支，别在主分支上直接改
- 论文/实验数据用 `.gitignore` 排除（大文件用 Git LFS 或 DVC）
- 提交信息写清「改了什么 + 为什么」，三个月后的自己会感谢现在的你

`.gitignore` 最小可用版：

```gitignore
.venv/
__pycache__/
*.pyc
.ipynb_checkpoints/
data/
outputs/
*.pt
*.pth
```

## 大文件与数据

Git 不适合存数据集和模型权重。选择：

- **Git LFS**：适合几十 MB 到几 GB 的文件
- **DVC**：适合数据集版本管理，与 Git 联动好
- 对象存储 / 网盘 + 脚本下载：最简单

## 提交前自查

```bash
git status                        # 有没有误加数据文件
git diff --cached                 # 实际要提交的内容
```

确认没有把 API Key、密码、服务器地址提交上去。一旦推到公开仓库，密钥就等于泄露，需要立刻作废重发。
