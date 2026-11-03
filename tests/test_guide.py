"""对话式基础评估的回归测试。

重点不是「分数对不对」，而是四条可解释性约束有没有被破坏：
1. 每条结论都能指回原话
2. 否定句不能被读成正面证据
3. 矛盾证据不硬猜
4. 没有证据就是未知，不是 0 分
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from research_agent.guide import (
    Confidence,
    Dimension,
    Level,
    Source,
    build_profile,
    build_roadmap,
    explain,
    get_concept,
    list_tracks,
    merge_profiles,
    next_question,
    render_card,
    render_roadmap,
    scan_materials,
    validate_concepts,
)
from research_agent.tools.guide_tools import (
    guide_assess,
    guide_explain,
    guide_roadmap,
    guide_tracks,
)

FIXTURES = Path(__file__).parent / "fixtures" / "guide_materials"


class TestSignals(unittest.TestCase):
    def test_beginner_dialogue(self) -> None:
        """典型新手自述：三个维度该被读出来，其余留未知。"""
        text = "我大一，Python 只会 print，没用过命令行，也没碰过深度学习。"
        p = build_profile(text)

        self.assertEqual(p.dimensions[Dimension.PYTHON].level, Level.BEGINNER)
        self.assertEqual(p.dimensions[Dimension.SHELL].level, Level.BEGINNER)
        self.assertEqual(p.dimensions[Dimension.DL].level, Level.BEGINNER)
        self.assertEqual(p.dimensions[Dimension.MATH].level, Level.UNKNOWN)
        self.assertEqual(p.overall, Level.BEGINNER)

    def test_negation_is_not_positive_evidence(self) -> None:
        """「没写过类」不能被「写过类」规则吃成 3 档。"""
        p = build_profile("我没写过类，也没写过包。")
        state = p.dimensions[Dimension.PYTHON]
        self.assertEqual(state.level, Level.BEGINNER)
        self.assertTrue(all(s.level != Level.WORKING for s in state.signals))

    def test_foundation_dialogue(self) -> None:
        p = build_profile("我写过爬虫，也用过 numpy。torch 装了两次，cuda 一直是 false。")
        self.assertEqual(p.dimensions[Dimension.PYTHON].level, Level.FOUNDATION)
        self.assertEqual(p.dimensions[Dimension.DL].level, Level.FOUNDATION)

    def test_conflict_is_not_guessed(self) -> None:
        """自述互相矛盾时，该维度不猜档位，并把两句话都摆出来。"""
        text = "我实现过 transformer。不过我其实没碰过深度学习。"
        p = build_profile(text)
        state = p.dimensions[Dimension.DL]

        self.assertEqual(state.level, Level.UNKNOWN)
        self.assertTrue(state.conflicts)
        self.assertEqual(state.confidence, Confidence.LOW)
        self.assertIn("没碰过深度学习", " ".join(state.conflicts))

    def test_weak_evidence_is_low_confidence(self) -> None:
        p = build_profile("我用过一点 pandas。")
        state = p.dimensions[Dimension.PYTHON]
        self.assertEqual(state.level, Level.FOUNDATION)
        self.assertEqual(state.confidence, Confidence.LOW)

    def test_quote_never_crosses_into_another_dimension(self) -> None:
        """「没用过命令行」不能被算成深度学习的证据。

        整句正则会让「没用过.{0,8}torch」从「没用过命令行」起头，
        引文于是变成命令行那句——档位也许对，但证据说的是别的维度。
        """
        text = "我没用过命令行，也没碰过torch。"
        p = build_profile(text)
        state = p.dimensions[Dimension.DL]

        self.assertTrue(state.signals)
        for s in state.signals:
            self.assertNotIn("命令行", s.quote, "深度学习证据引用到了命令行那句")
            self.assertIn(s.quote, text)
        self.assertEqual(state.level, Level.BEGINNER)
        self.assertEqual(p.dimensions[Dimension.SHELL].level, Level.BEGINNER)

    def test_each_clause_matched_on_its_own(self) -> None:
        """逗号两侧是两件事：一句给一个维度，不能互相借用关键词。"""
        p = build_profile("我写过爬虫，没用过命令行。")
        self.assertEqual(p.dimensions[Dimension.PYTHON].level, Level.FOUNDATION)
        self.assertEqual(p.dimensions[Dimension.SHELL].level, Level.BEGINNER)

    def test_empty_text_is_unknown_not_zero(self) -> None:
        p = build_profile("")
        self.assertEqual(p.overall, Level.UNKNOWN)
        self.assertEqual(p.known, [])
        self.assertEqual(p.coverage, "0/6")
        self.assertTrue(p.notes)
        self.assertFalse(any(d.score == 0 and d.is_known for d in p.dimensions.values()))


class TestExplainability(unittest.TestCase):
    def test_every_level_has_a_quote(self) -> None:
        p = build_profile("我写过类，用过 Linux，学过线性代数，上过机器学习，跑过训练，读过论文。")
        for state in p.known:
            self.assertTrue(state.signals, f"{state.dimension} 没有证据却定了档")
            for s in state.signals:
                self.assertTrue(s.quote, "证据缺少原话")
                self.assertTrue(s.why, "证据缺少判定理由")
                self.assertIn(s.quote, "我写过类，用过 Linux，学过线性代数，上过机器学习，跑过训练，读过论文。")

    def test_quote_is_verbatim_sentence(self) -> None:
        text = "我上周用 Linux 服务器跑了三天实验。"
        p = build_profile(text)
        quotes = [s.quote for state in p.known for s in state.signals]
        self.assertIn("我上周用 Linux 服务器跑了三天实验", quotes)

    def test_profile_is_json_serializable(self) -> None:
        p = build_profile("我写过爬虫。")
        payload = json.dumps(p.to_dict(), ensure_ascii=False)
        self.assertIn("python", payload)


class TestProbe(unittest.TestCase):
    def test_one_question_at_a_time(self) -> None:
        q = next_question(build_profile("我写过爬虫。"))
        self.assertIsNotNone(q)
        self.assertIsInstance(q["question"], str)
        self.assertNotIn("A.", q["question"])
        self.assertNotIn("问卷", q["question"])

    def test_conflict_question_first(self) -> None:
        q = next_question(build_profile("我实现过 transformer。不过我其实没碰过深度学习。"))
        self.assertIn("对不上", q["question"])

    def test_no_question_when_profile_is_solid(self) -> None:
        text = (
            "我写过类，也用过 pdb 和 pytest。我常年在 Linux 服务器上用 ssh，配过 PATH。"
            "学过线性代数，能推梯度。上过机器学习课，用 sklearn 跑过分类，调过超参。"
            "改过网络结构，实现过 attention，跑过训练。读过 5 篇论文，复现过别人的实验。"
        )
        p = build_profile(text)
        self.assertGreaterEqual(len(p.known), 5)
        self.assertIsNone(next_question(p))


class TestMerge(unittest.TestCase):
    def test_multi_turn_accumulates(self) -> None:
        first = build_profile("我没学过机器学习。")
        second = build_profile("最近用 sklearn 跑了个分类，还调过超参。")
        self.assertEqual(first.dimensions[Dimension.ML].level, Level.BEGINNER)

        merged = merge_profiles(first, second)
        self.assertTrue(merged.dimensions[Dimension.ML].is_known)
        self.assertEqual(merged.dimensions[Dimension.ML].level, Level.WORKING)


class TestMaterials(unittest.TestCase):
    """授权扫描的边界：只能证明材料在，不能证明掌握。"""

    def test_scan_counts_and_kinds(self) -> None:
        scan = scan_materials(FIXTURES)
        self.assertEqual(len(scan.by_kind("paper")), 2)
        self.assertEqual(len(scan.by_kind("notebook")), 2)
        self.assertEqual(sum(1 for p in scan.by_kind("paper") if p.has_note), 1)

    def test_file_evidence_is_weak_and_attributed(self) -> None:
        scan = scan_materials(FIXTURES)
        self.assertTrue(scan.signals)
        for s in scan.signals:
            self.assertEqual(s.source, Source.FILE)
            self.assertEqual(s.strength, "weak")
            self.assertTrue(s.source_ref, "文件证据必须带路径，否则没法核对")

    def test_training_loop_is_seen_but_two_notebooks_are_not_enough(self) -> None:
        scan = scan_materials(FIXTURES)
        ids = {s.rule_id for s in scan.signals}
        self.assertIn("material.notebook_train", ids)
        self.assertIn("material.papers_noted", ids)
        self.assertNotIn("material.notebooks", ids, "只有 2 个 notebook，不该触发")

    def test_missing_dir_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            scan_materials(FIXTURES / "no-such-dir")

    def test_file_only_profile_is_low_confidence(self) -> None:
        scan = scan_materials(FIXTURES)
        p = build_profile("", scan=scan)
        dl = p.dimensions[Dimension.DL]
        self.assertTrue(dl.is_known)
        self.assertEqual(dl.confidence, Confidence.LOW)
        self.assertIn("只有文件证据", dl.note)
        self.assertTrue(p.scanned)
        self.assertTrue(p.file_evidence)

    def test_file_contradicting_self_report_is_surfaced(self) -> None:
        """自述「没碰过」但目录里有训练 notebook：自述定档，冲突摆出来。"""
        scan = scan_materials(FIXTURES)
        p = build_profile("我没碰过深度学习。", scan=scan)
        dl = p.dimensions[Dimension.DL]
        self.assertEqual(dl.level, Level.BEGINNER)
        self.assertTrue(any("文件证据" in c for c in dl.caveats))

    def test_scan_is_json_serializable(self) -> None:
        scan = scan_materials(FIXTURES)
        json.dumps(scan.to_dict(), ensure_ascii=False)


ADVANCED = (
    "我写过类，也写过包。用过 Linux 服务器，配过 PATH，写过 bash 脚本。"
    "学过线性代数，能推梯度。上过机器学习课，用 sklearn 跑过分类，调过超参。"
    "跑过训练，改过网络结构。读过 5 篇论文，复现过别人的实验，做过实验记录。"
)


class TestRoadmap(unittest.TestCase):
    def test_beginner_starts_at_first_stage(self) -> None:
        r = build_roadmap(build_profile("我大一，Python 只会 print，没用过命令行。"))
        self.assertEqual(r.start.id, "stage0")
        self.assertEqual(r.skipped, [])
        self.assertGreater(r.total_hours, 0)

    def test_unknown_dimensions_are_not_skipped(self) -> None:
        """什么都没说时不能默认你会，一阶段都不跳。"""
        r = build_roadmap(build_profile(""))
        self.assertEqual(r.skipped, [])
        self.assertEqual(r.start.id, "stage0")

    def test_skip_comes_with_reason(self) -> None:
        r = build_roadmap(build_profile(ADVANCED), track="cv")
        self.assertGreater(len(r.skipped), 0)
        for item in r.skipped:
            self.assertTrue(item["reason"], "跳过阶段必须给出可反驳的理由")
            self.assertIn("已是", item["reason"])
        self.assertEqual(r.start.id, "stage4")

    def test_every_task_has_acceptance(self) -> None:
        for track in ["general", *[t["id"] for t in list_tracks()]]:
            r = build_roadmap(build_profile("我只会一点 python。"), track=track)
            for stage in r.stages:
                for task in stage.tasks:
                    self.assertTrue(task.acceptance, f"{stage.id}/{task.id} 没有验收标准")

    def test_track_expands_the_branch_stage(self) -> None:
        r = build_roadmap(build_profile("我只会一点 python。"), track="cv")
        stage = next(s for s in r.stages if s.is_track)
        self.assertEqual([t.id for t in stage.tasks], ["track.read", "track.repro", "track.project"])
        self.assertEqual(r.track.id, "cv")
        self.assertTrue(all(p.get("title") for p in r.track.papers))

    def test_unknown_track_raises(self) -> None:
        with self.assertRaises(ValueError):
            build_roadmap(build_profile("我只会一点 python。"), track="no-such-track")

    def test_render_shows_acceptance_and_reason(self) -> None:
        r = build_roadmap(build_profile(ADVANCED), track="cv")
        text = render_roadmap(r)
        self.assertIn("已跳过", text)
        self.assertIn("验收：", text)
        self.assertIn("预计投入", text)


class TestConcepts(unittest.TestCase):
    def test_exact_and_alias(self) -> None:
        self.assertEqual(explain("反向传播")["card"]["id"], "backprop")
        self.assertEqual(explain("venv")["card"]["id"], "venv")
        self.assertEqual(explain("BP")["match"], "alias")

    def test_alias_inside_a_sentence(self) -> None:
        r = explain("torch.cuda.is_available() 是 false")
        self.assertTrue(r["found"])
        self.assertEqual(r["card"]["id"], "cuda")

    def test_miss_does_not_fabricate(self) -> None:
        r = explain("量子纠缠")
        self.assertFalse(r["found"])
        self.assertIsNone(r["card"])
        self.assertIsInstance(r["candidates"], list)

    def test_all_cross_references_are_valid(self) -> None:
        self.assertEqual(validate_concepts(), [])

    def test_card_has_all_sections(self) -> None:
        card = get_concept("backprop")
        text = render_card(card)
        for part in ("一句话：", "类比：", "常见误解：", "前置：", "下一步：", "自检："):
            self.assertIn(part, text)

    def test_every_concept_is_reachable_by_name(self) -> None:
        for track in list_tracks():
            self.assertTrue(track["id"])
        for term in ("过拟合", "交叉验证", "消融实验", "三遍读法", "Docker"):
            self.assertTrue(explain(term)["found"], f"{term} 查不到")


class TestGuideTools(unittest.TestCase):
    """工具层约定：返回 dict、不抛异常、出错带 hint。"""

    def test_assess_without_input_returns_hint(self) -> None:
        r = guide_assess()
        self.assertFalse(r["ok"])
        self.assertIn("hint", r)

    def test_assess_carries_evidence(self) -> None:
        r = guide_assess("我写过爬虫，没用过命令行。")
        self.assertTrue(r["ok"])
        py = next(d for d in r["dimensions"] if d["dimension"] == "python")
        self.assertEqual(py["level"], 2)
        self.assertTrue(py["signals"][0]["quote"])
        self.assertIn("markdown", r)

    def test_scan_url_is_rejected(self) -> None:
        r = guide_assess("我写过爬虫", scan_dir="https://github.com/x/y")
        self.assertFalse(r["ok"])
        self.assertIn("本地", r["hint"])

    def test_scan_missing_dir_is_readable_error(self) -> None:
        r = guide_assess("我写过爬虫", scan_dir=str(FIXTURES / "no-such-dir"))
        self.assertFalse(r["ok"])
        self.assertIn("hint", r)

    def test_roadmap_tool_keeps_acceptance(self) -> None:
        r = guide_roadmap("我大一，Python 只会 print。", track="cv")
        self.assertTrue(r["ok"])
        self.assertEqual(r["track"]["id"], "cv")
        for stage in r["stages"]:
            for task in stage["tasks"]:
                self.assertTrue(task["acceptance"])

    def test_invalid_track_lists_options(self) -> None:
        r = guide_roadmap("我只会一点 python。", track="no-such")
        self.assertFalse(r["ok"])
        self.assertTrue(r["tracks"])

    def test_explain_miss_is_honest(self) -> None:
        r = guide_explain("量子纠缠")
        self.assertFalse(r["ok"])
        self.assertIn("没有", r["error"])
        self.assertIn("text", r)

    def test_tracks_tool(self) -> None:
        r = guide_tracks()
        self.assertEqual(r["count"], len(list_tracks()))


if __name__ == "__main__":
    unittest.main()
