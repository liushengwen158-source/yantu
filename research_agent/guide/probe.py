"""缺信息时该问什么。

两条约定：

- **一次只问一句**。画像不是问卷，是对话，连续追问会把人问跑。
- **只在需要时问**。用户没问「我该学什么」之前，不主动盘问水平。

问题刻意写成聊天口气，不带「请选择 A/B/C」的考试感。
"""

from __future__ import annotations

from .spec import Dimension, SkillProfile

# 路线最先要用到的维度排前面：先环境，再框架，最后数学与论文
PROBE_ORDER = [
    Dimension.SHELL,
    Dimension.PYTHON,
    Dimension.DL,
    Dimension.ML,
    Dimension.RESEARCH,
    Dimension.MATH,
]

PROBES: dict[Dimension, str] = {
    Dimension.PYTHON: "你平时用 Python 做过什么？写过最长的一个大概多少行？",
    Dimension.SHELL: "你装东西、跑代码一般是在 Windows 上点，还是走命令行？",
    Dimension.MATH: "线代和概率你是怎么学的——课上学的还是自己补的？推公式吃力吗？",
    Dimension.ML: "机器学习你接触过吗？比如用 sklearn 跑过分类，或者上过课？",
    Dimension.DL: "你有没有装过 torch，正经跑完过一次训练？",
    Dimension.RESEARCH: "论文你读过吗？是老师给的，还是自己找的？",
}

REASONS: dict[Dimension, str] = {
    Dimension.PYTHON: "后面所有实验都要写代码，先确认编程底子",
    Dimension.SHELL: "跑实验绕不开命令行，这一维空着路线没法起头",
    Dimension.MATH: "读论文卡符号多半是数学没过关，要先确认",
    Dimension.ML: "决定你是从「什么是训练」开始，还是直接进复现",
    Dimension.DL: "决定复现阶段挑 CNN 小模型还是直接上 Transformer",
    Dimension.RESEARCH: "决定路线里要不要加「怎么读第一篇论文」这一段",
}


def next_question(profile: SkillProfile) -> dict[str, str] | None:
    """返回下一个该问的问题；信息够了就返回 None，不硬凑话题。"""
    # 证据打架优先问清楚，否则画像里的档位随时会翻
    for state in profile.conflicts:
        quotes = " / ".join(state.conflicts)
        return {
            "dimension": state.dimension.value,
            "dimension_label": state.dimension.label,
            "question": f"我这儿对不上：{quotes}。具体是哪种情况？",
            "reason": "两句话指向的水平差太远，先澄清再定档",
        }

    for dim in PROBE_ORDER:
        state = profile.dimensions.get(dim)
        if state is not None and state.is_known:
            continue
        return {
            "dimension": dim.value,
            "dimension_label": dim.label,
            "question": PROBES[dim],
            "reason": REASONS[dim],
        }

    # 都定了档，只补「只有含糊表述」的那几维；明确说过的就不再追问
    thin = [s for s in profile.known if s.confidence.value == "low"]
    if thin:
        dim = thin[0].dimension
        return {
            "dimension": dim.value,
            "dimension_label": dim.label,
            "question": PROBES[dim],
            "reason": "这一维只有含糊表述，再确认一句才敢用来排路线",
        }

    return None
