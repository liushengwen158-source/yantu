"""按方向编排学习路线。

设计要点：

- **主干通用，内容按方向注入**。工程基础、读论文方法、记录与表达这三段
  哪个方向都一样；方向基本功、复现哪篇论文、第一个项目完全由方向决定。
- **跳过要给理由**。只有「已知且达到要求」才跳过，未知一律不跳；
  跳过的阶段仍然列出来，附上各维度的档位对比。
- **验收标准是任务的入场券**。没有验收标准的任务不进路线，
  否则「读几篇论文」这种任务做了也不知道算不算做完。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from .concepts import get as get_concept
from .domains import (
    DomainSpec,
    get_domain,
    list_domains,
    load_domains,
    required_levels,
)
from .spec import Dimension, Domain, Level, SkillProfile

KNOWLEDGE_DIR = Path(__file__).resolve().parent.parent / "knowledge"
ROADMAP_PATH = KNOWLEDGE_DIR / "roadmap.json"


@dataclass
class RoadmapTask:
    id: str
    title: str
    detail: str = ""
    acceptance: list[str] = field(default_factory=list)
    hours: int = 0
    concepts: list[str] = field(default_factory=list)
    dims: list[str] = field(default_factory=list)
    paper: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "detail": self.detail,
            "acceptance": self.acceptance,
            "hours": self.hours,
            "concepts": self.concepts,
            "dims": self.dims,
            "paper": self.paper,
        }


@dataclass
class RoadmapStage:
    id: str
    title: str
    why: str = ""
    tasks: list[RoadmapTask] = field(default_factory=list)
    skipped: bool = False
    skip_reason: str = ""
    deferred: bool = False          # 超出时间预算，排到后面

    @property
    def hours(self) -> int:
        return sum(t.hours for t in self.tasks)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "why": self.why,
            "hours": self.hours,
            "skipped": self.skipped,
            "skip_reason": self.skip_reason,
            "deferred": self.deferred,
            "tasks": [t.to_dict() for t in self.tasks],
        }


@dataclass
class Roadmap:
    domain: Domain
    stages: list[RoadmapStage] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    reading_hint: str = ""
    profile: SkillProfile | None = None

    @property
    def active_stages(self) -> list[RoadmapStage]:
        return [s for s in self.stages if not s.skipped and not s.deferred]

    @property
    def deferred_stages(self) -> list[RoadmapStage]:
        return [s for s in self.stages if s.deferred and not s.skipped]

    @property
    def skipped_stages(self) -> list[RoadmapStage]:
        return [s for s in self.stages if s.skipped]

    @property
    def total_hours(self) -> int:
        return sum(s.hours for s in self.active_stages)

    @property
    def task_count(self) -> int:
        return sum(len(s.tasks) for s in self.active_stages)

    def to_dict(self) -> dict[str, Any]:
        return {
            "domain": self.domain.value,
            "domain_label": self.domain.label,
            "total_hours": self.total_hours,
            "task_count": self.task_count,
            "reading_hint": self.reading_hint,
            "notes": self.notes,
            "stages": [s.to_dict() for s in self.stages],
        }


@lru_cache(maxsize=1)
def load_roadmap() -> dict[str, Any]:
    if not ROADMAP_PATH.exists():
        return {}
    return json.loads(ROADMAP_PATH.read_text(encoding="utf-8"))


def _known_concepts(ids: list[str]) -> list[str]:
    """过滤掉知识库里不存在的概念 id，避免路线里出现点不开的链接。"""
    return [cid for cid in ids if get_concept(cid)]


def _task_from_dict(raw: dict[str, Any]) -> RoadmapTask | None:
    title = raw.get("title")
    acceptance = list(raw.get("acceptance") or [])
    if not title or not acceptance:
        # 没有验收标准的任务定义不清，宁可不进路线
        return None
    return RoadmapTask(
        id=raw.get("id", ""),
        title=title,
        detail=raw.get("detail", ""),
        acceptance=acceptance,
        hours=int(raw.get("hours") or 0),
        concepts=_known_concepts(list(raw.get("concepts") or [])),
        dims=list(raw.get("dims") or []),
    )


def _domain_tasks(spec: DomainSpec | None, fallback: list[dict[str, Any]]) -> list[RoadmapTask]:
    source = spec.stage_tasks if spec and spec.stage_tasks else fallback
    out: list[RoadmapTask] = []
    for raw in source:
        task = _task_from_dict(raw)
        if task:
            out.append(task)
    return out


def _paper_tasks(spec: DomainSpec | None) -> list[RoadmapTask]:
    """复现阶段：读清单 + 复现其中一篇。论文清单来自方向库。"""
    if spec is None or not spec.papers:
        return []
    titles = "、".join(f"《{p.get('title', '')}》" for p in spec.papers[:3])
    read_task = RoadmapTask(
        id="repro.read",
        title=f"精读这个方向的 {len(spec.papers[:3])} 篇论文",
        detail=f"{titles}。读法用三遍读法，每篇都要能说出问题、方法、证据。",
        acceptance=[
            "每篇写一段「一句话结论」，不许抄摘要",
            "画出方法与实验的关系，说清它和谁比",
            "标出你想复现的那一篇，以及需要什么资源",
            "记录读不懂的具体位置，而不是笼统说难",
        ],
        hours=12,
        concepts=_known_concepts(["three-pass", "limitation"]),
        dims=["research"],
    )
    target = spec.papers[0] if spec.papers else {}
    repro_task = RoadmapTask(
        id="repro.run",
        title=f"复现《{target.get('title', '')}》的核心实验",
        detail=target.get("note") or "按论文的设定跑一次，先求跑通，再求对上数字。",
        acceptance=[
            "环境与版本记录完整，别人能照着重跑",
            "结果与论文同量级，或写清差距来自数据、算力还是实现",
            "至少跑两次，说明结果稳不稳定",
            "整理成一份可读的复现日志",
        ],
        hours=18,
        concepts=_known_concepts(["reproducibility", "experiment-log"]),
        dims=["research"],
        paper=target,
    )
    return [read_task, repro_task]


def _project_task(spec: DomainSpec | None) -> RoadmapTask | None:
    if spec is None or not spec.first_project:
        return None
    raw = dict(spec.first_project)
    raw.setdefault("dims", ["coding", "research"])
    return _task_from_dict(raw)


def _generic_project(domain: Domain) -> RoadmapTask:
    return RoadmapTask(
        id="project.generic",
        title="做一个能验证的小问题",
        detail=(
            f"方向还没定（当前：{domain.label}），先不要碰大题目："
            "找一个能在两周内做完、能拿数据或证明说话的小问题。"
        ),
        acceptance=[
            "用一句话写清你要回答的问题",
            "给出结果，并说明它和现有做法相比如何",
            "写下三个局限",
        ],
        hours=20,
        concepts=_known_concepts(["baseline", "report"]),
        dims=["coding", "research"],
    )


def _should_skip(
    skip_if: Any, profile: SkillProfile | None
) -> tuple[bool, str]:
    """跳过必须建立在「已知且达标」上。未知一律不跳。"""
    if not skip_if or profile is None:
        return False, ""
    if skip_if == "domain.required":
        required = required_levels(profile.domain)
        if not required:
            return False, ""
        details: list[str] = []
        for dim, need in required.items():
            state = profile.dimensions.get(dim)
            have = int(state.level) if state and state.is_known else Level.UNKNOWN
            details.append(f"{dim.label} {Level(have).label}/{Level(need).label}")
            if have < need:
                return False, ""
        return True, "该方向要求的基本功都已达标：" + "，".join(details)
    if not isinstance(skip_if, dict):
        return False, ""
    details = []
    for dim_name, need in skip_if.items():
        try:
            dimension = Dimension(dim_name)
            level = Level(int(need))
        except ValueError:
            continue
        state = profile.dimensions.get(dimension)
        if state is None:
            return False, ""
        have = int(state.level) if state.is_known else int(Level.UNKNOWN)
        details.append(f"{dimension.label} {Level(have).label}/{level.label}")
        if not state.is_known or have < int(level):
            return False, ""
    if not details:
        return False, ""
    return True, "已达标的维度：" + "，".join(details)


def build_roadmap(
    profile: SkillProfile,
    *,
    hours_limit: int | None = None,
) -> Roadmap:
    """按画像编排路线。方向来自画像，不会另猜。"""
    data = load_roadmap()
    domain = profile.domain if profile else Domain.UNKNOWN
    spec = get_domain(domain)
    roadmap = Roadmap(domain=domain, profile=profile)

    if spec is not None:
        roadmap.reading_hint = spec.reading_hint
        roadmap.notes.append(
            f"路线按「{spec.name}」编排：方向基本功与复现论文都取这个方向的内容。"
        )
    else:
        unknown = data.get("unknown_domain", {})
        roadmap.notes.append(
            "方向还没定，路线先走「试方向」的版本；确认方向后我再排一份针对性的。"
        )
        if unknown.get("why"):
            roadmap.notes.append(unknown["why"])

    fallback_tasks = list((data.get("unknown_domain") or {}).get("tasks") or [])

    for raw_stage in data.get("stages", []):
        stage = RoadmapStage(
            id=raw_stage.get("id", ""),
            title=raw_stage.get("title", ""),
            why=raw_stage.get("why", ""),
        )
        source = raw_stage.get("source", "static")
        if source == "static":
            stage.tasks = [
                t for t in (_task_from_dict(raw) for raw in raw_stage.get("tasks") or []) if t
            ]
        elif source == "domain.stage_tasks":
            stage.tasks = _domain_tasks(spec, fallback_tasks)
        elif source == "domain.papers":
            stage.tasks = _paper_tasks(spec)
            if not stage.tasks:
                # 方向未定时不排复现，改成一节「读懂一篇论文」的通用任务
                stage.tasks = [
                    RoadmapTask(
                        id="repro.generic",
                        title="完整读完一篇论文并讲清它",
                        detail="方向未定时先不求复现，把一篇论文读到能讲给别人听。",
                        acceptance=[
                            "能不看论文说出它解决什么问题",
                            "说清它用什么办法、证据是什么",
                            "指出一个你认为站不住的地方",
                        ],
                        hours=10,
                        concepts=_known_concepts(["three-pass"]),
                        dims=["research"],
                    )
                ]
        elif source == "domain.first_project":
            task = _project_task(spec)
            stage.tasks = [task] if task else [_generic_project(domain)]

        skipped, reason = _should_skip(raw_stage.get("skip_if"), profile)
        if skipped and stage.tasks:
            stage.skipped = True
            stage.skip_reason = reason
        roadmap.stages.append(stage)

    if hours_limit and hours_limit > 0:
        used = 0
        for stage in roadmap.stages:
            if stage.skipped:
                continue
            if used >= hours_limit:
                stage.deferred = True
                continue
            used += stage.hours
            if used > hours_limit:
                roadmap.notes.append(
                    f"你给的预算约 {hours_limit} 小时，"
                    f"「{stage.title}」这一阶段会超出，可拆到下一段做。"
                )

    roadmap.notes.append(
        "顺序可以调：如果某个阶段的前置已经具备，直接跳过；"
        "判定不对就说你的实际情况，我按你说的重排。"
    )
    return roadmap


def list_tracks() -> list[dict[str, Any]]:
    """列出所有方向。名字沿用旧接口，内容是方向库。"""
    return [
        {
            "id": spec.domain.value,
            "name": spec.name,
            "why": spec.why,
            "skills": spec.skills,
            "hours": sum(int(t.get("hours") or 0) for t in spec.stage_tasks),
        }
        for spec in list_domains()
    ]


def render_roadmap(roadmap: Roadmap) -> str:
    """完整路线，Markdown。"""
    out: list[str] = []
    out.append(f"# 学习路线：{roadmap.domain.label}")
    if roadmap.profile is not None and roadmap.profile.domain_evidence:
        out.append(f"> 方向依据你原话：「{roadmap.profile.domain_evidence}」")
    out.append(
        f"共 {len(roadmap.active_stages)} 个阶段、{roadmap.task_count} 个任务、"
        f"约 {roadmap.total_hours} 小时。"
    )
    if roadmap.reading_hint:
        out.append(f"这个方向的读法：{roadmap.reading_hint}")

    if roadmap.notes:
        out.append("")
        for note in roadmap.notes:
            out.append(f"- {note}")

    index = 0
    for stage in roadmap.active_stages:
        index += 1
        out.append("")
        out.append(f"## {index}. {stage.title}（约 {stage.hours} 小时）")
        if stage.why:
            out.append(stage.why)
        for task in stage.tasks:
            out.append("")
            out.append(f"- [ ] **{task.title}**（{task.hours} 小时）")
            if task.detail:
                out.append(f"  - 做什么：{task.detail}")
            if task.acceptance:
                out.append("  - 验收：")
                for item in task.acceptance:
                    out.append(f"    - {item}")
            if task.concepts:
                names = "、".join(
                    (get_concept(cid) or {}).get("name", cid) for cid in task.concepts
                )
                out.append(f"  - 卡住时问：{names}")

    if roadmap.deferred_stages:
        out.append("")
        out.append("## 超出预算，可延后")
        for stage in roadmap.deferred_stages:
            out.append(f"- {stage.title}（约 {stage.hours} 小时）")

    if roadmap.skipped_stages:
        out.append("")
        out.append("## 已跳过（附理由）")
        for stage in roadmap.skipped_stages:
            out.append(f"- {stage.title}：{stage.skip_reason}")

    out.append("")
    out.append("## 说明")
    out.append("- 每个任务都有验收标准；做不出来说明前置没到位，不算失败。")
    out.append("- 跳过只发生在「已知且达标」的情况下，未知的维度不会被跳过。")
    out.append("- 方向不同，同一阶段的内容就不同；换方向时说一声，我重排。")
    return "\n".join(out)


def render_brief(roadmap: Roadmap) -> str:
    """一段话摘要，给工具返回和 CLI 用。"""
    if not roadmap.active_stages:
        return f"方向：{roadmap.domain.label}；没有可排的阶段。"
    titles = "、".join(s.title for s in roadmap.active_stages)
    text = (
        f"方向：{roadmap.domain.label}；"
        f"{len(roadmap.active_stages)} 个阶段（{titles}），"
        f"{roadmap.task_count} 个任务，约 {roadmap.total_hours} 小时。"
    )
    if roadmap.skipped_stages:
        text += f" 跳过 {len(roadmap.skipped_stages)} 个阶段（已知达标）。"
    return text


def stage_by_id(roadmap: Roadmap, stage_id: str) -> RoadmapStage | None:
    return next((s for s in roadmap.stages if s.id == stage_id), None)


def all_domains() -> list[DomainSpec]:
    return list_domains()
