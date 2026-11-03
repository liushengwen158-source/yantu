"""学习路线：把六维画像换成一条能验收的路。

三条规则：

- **跳过要讲清理由**：某阶段的能力已经具备就跳过，并写明「你的命令行已是
  能干活档，要求是有基础」，用户能反驳。
- **没确认就不跳过**：维度未知时宁可多学一遍，也不默认你会。
- **每项任务都要有验收标准**：没有可验收产出的条目不进路线，
  否则「学完了」只能靠感觉。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .spec import Dimension, Level, SkillProfile

KB_PATH = Path(__file__).resolve().parent.parent / "knowledge" / "roadmap.json"

DEFAULT_WEEKLY_HOURS = 10


@dataclass
class Task:
    id: str
    title: str
    detail: str
    acceptance: list[str] = field(default_factory=list)
    hours: int = 0
    concepts: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "detail": self.detail,
            "acceptance": self.acceptance,
            "hours": self.hours,
            "concepts": self.concepts,
        }


@dataclass
class Stage:
    id: str
    name: str
    goal: str
    weeks: tuple[int, int]
    tasks: list[Task] = field(default_factory=list)
    is_track: bool = False
    index: int = 0   # 在知识库里的序号，跳过阶段后仍然对得上

    @property
    def hours(self) -> int:
        return sum(t.hours for t in self.tasks)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "index": self.index,
            "name": self.name,
            "goal": self.goal,
            "weeks": list(self.weeks),
            "hours": self.hours,
            "is_track": self.is_track,
            "tasks": [t.to_dict() for t in self.tasks],
        }


@dataclass
class Track:
    id: str
    name: str
    why: str
    skills: list[str] = field(default_factory=list)
    first_project: str = ""
    papers: list[dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "why": self.why,
            "skills": self.skills,
            "first_project": self.first_project,
            "papers": self.papers,
        }


@dataclass
class Roadmap:
    stages: list[Stage] = field(default_factory=list)
    skipped: list[dict[str, str]] = field(default_factory=list)
    track: Track | None = None
    total_hours: int = 0
    weekly_hours: int = DEFAULT_WEEKLY_HOURS
    notes: list[str] = field(default_factory=list)
    profile_summary: dict[str, Any] = field(default_factory=dict)

    @property
    def start(self) -> Stage | None:
        return self.stages[0] if self.stages else None

    @property
    def weeks(self) -> tuple[int, int]:
        lo = sum(s.weeks[0] for s in self.stages)
        hi = sum(s.weeks[1] for s in self.stages)
        return lo, hi

    def to_dict(self) -> dict[str, Any]:
        return {
            "start": self.start.id if self.start else "",
            "start_name": self.start.name if self.start else "",
            "stages": [s.to_dict() for s in self.stages],
            "skipped": self.skipped,
            "track": self.track.to_dict() if self.track else None,
            "total_hours": self.total_hours,
            "weekly_hours": self.weekly_hours,
            "weeks": list(self.weeks),
            "notes": self.notes,
            "profile": self.profile_summary,
        }


def load_roadmap(path: Path | None = None) -> dict[str, Any]:
    p = path or KB_PATH
    return json.loads(p.read_text(encoding="utf-8"))


def list_tracks() -> list[dict[str, str]]:
    """所有可选的细分方向，供 agent 或 CLI 展示。"""
    data = load_roadmap()
    return [
        {"id": t["id"], "name": t["name"], "why": t.get("why", "")}
        for t in data.get("tracks", [])
    ]


def _track_from(data: dict[str, Any]) -> Track:
    return Track(
        id=data["id"],
        name=data["name"],
        why=data.get("why", ""),
        skills=list(data.get("skills", [])),
        first_project=data.get("first_project", ""),
        papers=list(data.get("papers", [])),
    )


def _track_tasks(track: Track) -> list[Task]:
    """方向确定后，阶段 4 拆成「读透 → 复现 → 做一个自己的」。"""
    must = [p for p in track.papers if p.get("level") == "必读"] or track.papers[:1]
    titles = "、".join(p.get("title", "") for p in must)
    return [
        Task(
            id="track.read",
            title="读透方向的必读工作",
            detail=f"必读：{titles}。先读摘要与图表，再抠方法与实验。",
            acceptance=[
                "每篇写三句话：解决什么问题、怎么做的、局限在哪",
                "能说清这个方向现在主要卡在哪里",
            ],
            hours=8,
            concepts=["three-pass", "related-work"],
        ),
        Task(
            id="track.repro",
            title="复现其中一篇",
            detail="挑代码最干净、算力要求最低的一篇，跑通并对上数字。",
            acceptance=[
                "指标与论文差距 ≤3%，或写清差距来源",
                "记录硬件、种子与关键超参",
            ],
            hours=20,
            concepts=["reproducibility", "baseline"],
        ),
        Task(
            id="track.project",
            title="做一个自己的小项目",
            detail=track.first_project or "在方向内选一个具体问题，做出可交付的结果。",
            acceptance=[
                "有可运行的代码与一份结果说明",
                "至少做一次对比实验，能解释差异",
            ],
            hours=15,
            concepts=["ablation", "experiment-log", "report"],
        ),
    ]


def _generic_track_tasks() -> list[Task]:
    """还没定方向时，先花小成本试三个，而不是空想。"""
    return [
        Task(
            id="track.try",
            title="花两天试三个方向",
            detail="每个方向跑一个官方 demo 或教程，记录「卡在哪、爽在哪」。",
            acceptance=[
                "三个方向各留下一份 200 字体验记录",
                "写下自己最愿意再花一周的是哪个，为什么",
            ],
            hours=6,
            concepts=[],
        ),
        Task(
            id="track.decide",
            title="把方向收窄成一个能动手的问题",
            detail="领域太大不能做，问题要窄到「一周能验证一次」。",
            acceptance=[
                "用一句话写出问题，并说明为什么它值得做",
                "给出第一个可验证的实验设计",
            ],
            hours=4,
            concepts=[],
        ),
    ]


def _skip_reason(stage: dict[str, Any], profile: SkillProfile) -> str | None:
    """返回跳过理由；不该跳过时返回 None。"""
    skip_if: dict[str, int] = stage.get("skip_if") or {}
    if not skip_if:
        return None

    met: list[str] = []
    for key, need in skip_if.items():
        dim = Dimension(key)
        state = profile.dimensions[dim]
        if not state.is_known or int(state.level) < int(need):
            return None
        met.append(f"{dim.label} 已是「{state.level.label}」档（要求 ≥{Level(int(need)).label}）")
    return "；".join(met)


def build_roadmap(
    profile: SkillProfile,
    track: str = "general",
    weekly_hours: int | None = None,
) -> Roadmap:
    """根据画像生成路线。

    ``track`` 取方向 id（cv/nlp/system/data/theory/ai4science），
    留空或 ``general`` 表示还没定方向。
    """
    data = load_roadmap()
    hours_per_week = weekly_hours or int(data.get("weekly_hours", DEFAULT_WEEKLY_HOURS))

    roadmap = Roadmap(weekly_hours=hours_per_week)
    roadmap.profile_summary = {
        "coverage": profile.coverage,
        "overall": profile.overall.label,
        "known": {d.dimension.label: d.level.label for d in profile.known},
        "unknown": [d.dimension.label for d in profile.unknown],
        "pending": [d.dimension.label for d in profile.pending],
    }

    chosen: Track | None = None
    if track and track != "general":
        raw = next((t for t in data.get("tracks", []) if t["id"] == track), None)
        if raw is None:
            ids = "、".join(t["id"] for t in data.get("tracks", []))
            raise ValueError(f"没有这个方向：{track}（可选 {ids}）")
        chosen = _track_from(raw)
    roadmap.track = chosen

    for position, stage_data in enumerate(data.get("stages", []), 1):
        stage_id = stage_data["id"]
        reason = _skip_reason(stage_data, profile)
        if reason:
            roadmap.skipped.append(
                {
                    "stage_id": stage_id,
                    "index": position,
                    "name": stage_data["name"],
                    "reason": reason,
                }
            )
            continue

        stage = Stage(
            id=stage_id,
            name=stage_data["name"],
            goal=stage_data.get("goal", ""),
            weeks=tuple(stage_data.get("weeks", [1, 1])),  # type: ignore[arg-type]
            is_track=bool(stage_data.get("track")),
            index=position,
        )

        if stage.is_track:
            if chosen is not None:
                stage.tasks = _track_tasks(chosen)
                stage.name = f"定方向：{chosen.name}"
            else:
                stage.tasks = _generic_track_tasks()
                stage.name = "定方向：先试三个方向，再收窄成一个问题"
        else:
            stage.tasks = [
                Task(
                    id=t["id"],
                    title=t["title"],
                    detail=t.get("detail", ""),
                    acceptance=list(t.get("acceptance", [])),
                    hours=int(t.get("hours", 0)),
                    concepts=list(t.get("concepts", [])),
                )
                for t in stage_data.get("tasks", [])
                if t.get("acceptance")   # 没有验收标准的条目不进路线
            ]
        roadmap.stages.append(stage)

    roadmap.total_hours = sum(s.hours for s in roadmap.stages)

    if profile.unknown:
        names = "、".join(d.dimension.label for d in profile.unknown)
        roadmap.notes.append(f"{names} 还没有证据，按「需要学」排进了路线；如果你已经会，可以直接跳过对应阶段。")
    if profile.pending:
        names = "、".join(d.dimension.label for d in profile.pending)
        roadmap.notes.append(f"{names} 的证据互相矛盾，路线按低档排，确认后可重排。")
    if roadmap.skipped:
        roadmap.notes.append("跳过的阶段依据的是你已确认的档位，不认同可以直接说，我按你的说法重排。")
    if not roadmap.stages:
        roadmap.notes.append("所有阶段都已满足，直接找一个具体问题开始做。")
    return roadmap


def _concept_names(ids: list[str]) -> str:
    """路线里写概念名称而不是 id；库里没有的 id 原样保留，不假装认识。"""
    from .concepts import get as get_concept

    names: list[str] = []
    for cid in ids:
        card = get_concept(cid)
        names.append(card["name"] if card else cid)
    return "、".join(names)


def render_roadmap(roadmap: Roadmap) -> str:
    """人类可读的 Markdown 路线。"""
    parts: list[str] = []
    start = roadmap.start
    title = f"# 学习路线 · 从「{start.name}」开始" if start else "# 学习路线 · 阶段已全部满足"
    parts.append(title)
    parts.append("")

    profile = roadmap.profile_summary
    parts.append(
        f"画像：覆盖 {profile.get('coverage', '')} 维，整体{profile.get('overall', '')}"
    )
    if profile.get("unknown"):
        parts.append(f"未确认：{'、'.join(profile['unknown'])}")

    if roadmap.skipped:
        parts.append("")
        parts.append("## 已跳过")
        for item in roadmap.skipped:
            parts.append(f"- 阶段 {item['index']} {item['name']}：{item['reason']}")

    for stage in roadmap.stages:
        parts.append("")
        parts.append(f"## 阶段 {stage.index} · {stage.name}（{stage.weeks[0]}–{stage.weeks[1]} 周）")
        if stage.goal:
            parts.append(f"目标：{stage.goal}")
        for task in stage.tasks:
            parts.append("")
            parts.append(f"- [ ] **{task.title}**（{task.hours} 小时）")
            if task.detail:
                parts.append(f"  - 做什么：{task.detail}")
            for item in task.acceptance:
                parts.append(f"  - 验收：{item}")
            if task.concepts:
                parts.append(f"  - 相关概念：{_concept_names(task.concepts)}")

    if roadmap.track:
        track = roadmap.track
        parts.append("")
        parts.append(f"## 方向：{track.name}")
        parts.append(f"为什么选它：{track.why}")
        if track.papers:
            parts.append("")
            parts.append("阅读清单（按顺序）：")
            for i, paper in enumerate(track.papers, 1):
                level = paper.get("level", "")
                parts.append(f"{i}. 《{paper.get('title', '')}》{'（' + level + '）' if level else ''}")
                if paper.get("why"):
                    parts.append(f"   - 为什么读：{paper['why']}")
                if paper.get("note"):
                    parts.append(f"   - 提醒：{paper['note']}")
        if track.skills:
            parts.append("")
            parts.append(f"要补的工具：{'、'.join(track.skills)}")

    parts.append("")
    lo, hi = roadmap.weeks
    parts.append(
        f"预计投入约 {roadmap.total_hours} 小时；按每周 {roadmap.weekly_hours} 小时算，"
        f"大约 {lo}–{hi} 周。"
    )
    if roadmap.notes:
        parts.append("")
        for note in roadmap.notes:
            parts.append(f"> {note}")
    return "\n".join(parts)


def render_brief(roadmap: Roadmap, limit: int = 400) -> str:
    """给 agent 的一句话摘要。"""
    start = roadmap.start
    if start is None:
        return "所有阶段都已满足，可以直接开始做具体问题。"
    head = f"从「{start.name}」开始，共 {len(roadmap.stages)} 个阶段、约 {roadmap.total_hours} 小时"
    if roadmap.skipped:
        head += f"，跳过 {len(roadmap.skipped)} 个已满足阶段"
    if roadmap.track:
        head += f"，方向：{roadmap.track.name}"
    return head[:limit]
