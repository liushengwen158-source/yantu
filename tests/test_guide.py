"""新手引导的回归测试。

覆盖四组不变量：

1. 维度随方向激活：做系统的学生看不到「深度学习」这一行
2. 证据不跨短句、负向优先、英文关键词按词边界匹配
3. 路线内容按方向分岔，跳过必须给理由，未知不跳过
4. 概念库自洽，工具层不抛异常
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from research_agent.guide import (
    CORE_DIMENSIONS,
    Dimension,
    Domain,
    Level,
    build_profile,
    build_profile_from_signals,
    build_roadmap,
    detect_domain,
    explain,
    list_domains,
    match_signals,
    next_question,
    probe_plan,
    render_profile,
    render_roadmap,
    render_table,
    required_levels,
    resolve_domain,
    scan_materials,
    split_clauses,
    validate_concepts,
)
from research_agent.guide.probe import DIMENSION_QUESTIONS
from research_agent.guide.signals import load_rules
from research_agent.tools import guide_tools

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "guide_materials"


class DimensionTests(unittest.TestCase):
    def test_core_dimensions_are_four(self) -> None:
        self.assertEqual(len(CORE_DIMENSIONS), 4)

    def test_level_labels_are_monotonic(self) -> None:
        scores = [Level(i).score for i in range(5)]
        self.assertEqual(scores, sorted(scores))

    def test_unknown_level_is_not_a_skill(self) -> None:
        self.assertEqual(Level.UNKNOWN.score, 0)


class SignalTests(unittest.TestCase):
    def test_quote_never_crosses_into_another_dimension(self) -> None:
        signals = match_signals("我没用过命令行，也没碰过torch")
        tools = [s for s in signals if s.dimension is Dimension.TOOLS]
        dl = [s for s in signals if s.dimension is Dimension.DL]
        self.assertTrue(tools, "命令行这一维应该被抓到")
        self.assertTrue(dl, "深度学习这一维应该被抓到")
        for signal in dl:
            self.assertNotIn("命令行", signal.quote)

    def test_negative_wins_in_same_clause(self) -> None:
        signals = match_signals("我没用过命令行")
        tools = [s for s in signals if s.dimension is Dimension.TOOLS]
        self.assertTrue(tools)
        self.assertTrue(all(s.polarity == "negative" for s in tools))

    def test_sql_with_space_is_matched(self) -> None:
        signals = match_signals("写过 SQL")
        self.assertTrue([s for s in signals if s.dimension is Dimension.DATA])

    def test_english_keyword_is_case_insensitive(self) -> None:
        lower = match_signals("我跑过训练")
        upper = match_signals("我跑过训练")
        self.assertEqual(len(lower), len(upper))

    def test_clause_split_keeps_meaning(self) -> None:
        clauses = split_clauses("我会一点 Python，但没写过项目。")
        self.assertEqual(len(clauses), 2)

    def test_rule_library_loads(self) -> None:
        rules = load_rules()
        self.assertGreater(len(rules), 50)
        dimensions = {r["dimension"] for r in rules}
        self.assertIn(Dimension.SYSTEMS, dimensions)
        self.assertIn(Dimension.SECURITY, dimensions)


class DomainDetectionTests(unittest.TestCase):
    def test_detects_database_from_talk(self) -> None:
        domain, evidence = detect_domain("我在做数据库方向")
        self.assertIs(domain, Domain.DATABASE)
        self.assertIn("数据库", evidence)

    def test_detects_ai_shorthand(self) -> None:
        domain, _ = detect_domain("我做 AI 方向")
        self.assertIs(domain, Domain.AI)

    def test_ai_keyword_does_not_match_inside_words(self) -> None:
        # train / detail 里都含 ai，不能因此判成人工智能
        domain, _ = detect_domain("我在看 train 和 detail 两个字段")
        self.assertIsNot(domain, Domain.AI)

    def test_explicit_undecided_stays_unknown(self) -> None:
        domain, _ = detect_domain("方向还没定，先看看")
        self.assertIs(domain, Domain.UNKNOWN)

    def test_resolve_domain_accepts_chinese_name(self) -> None:
        self.assertIs(resolve_domain("系统"), Domain.SYSTEMS)
        self.assertIs(resolve_domain("database"), Domain.DATABASE)

    def test_resolve_domain_rejects_nonsense(self) -> None:
        with self.assertRaises(ValueError):
            resolve_domain("炼丹修仙")

    def test_every_domain_declares_required_levels(self) -> None:
        for spec in list_domains():
            self.assertIn(Dimension.CODING, spec.required)
            self.assertTrue(spec.papers, f"{spec.name} 应该给论文")
            self.assertTrue(spec.stage_tasks, f"{spec.name} 应该给基本功任务")


class ProfileTests(unittest.TestCase):
    def test_unknown_domain_only_has_core_dimensions(self) -> None:
        profile = build_profile("我还在看，方向没定")
        self.assertIs(profile.domain, Domain.UNKNOWN)
        self.assertEqual(set(profile.dimensions), set(CORE_DIMENSIONS))
        self.assertNotIn(Dimension.ML, profile.dimensions)
        self.assertNotIn(Dimension.DL, profile.dimensions)

    def test_systems_profile_hides_ml(self) -> None:
        profile = build_profile("我在做系统方向，写过 C")
        self.assertIs(profile.domain, Domain.SYSTEMS)
        self.assertNotIn(Dimension.ML, profile.dimensions)
        self.assertIn(Dimension.SYSTEMS, profile.dimensions)

    def test_domain_requirements_are_attached(self) -> None:
        profile = build_profile("我在做理论方向，学过离散数学")
        self.assertEqual(profile.dimensions[Dimension.MATH].required,
                         required_levels(Domain.THEORY)[Dimension.MATH])

    def test_conflict_is_flagged_not_averaged(self) -> None:
        profile = build_profile("我训练过模型，但完全没碰过深度学习")
        state = profile.dimensions[Dimension.DL]
        self.assertTrue(state.conflicts)
        self.assertIs(state.level, Level.UNKNOWN)

    def test_file_evidence_does_not_set_level(self) -> None:
        scan = scan_materials(FIXTURES)
        profile = build_profile("", scan=scan)
        signed = [d for d in profile.dimensions.values() if d.signals]
        self.assertTrue(signed, "夹具里应该有可用的文件证据")
        for state in signed:
            self.assertIs(state.level, Level.UNKNOWN)
            self.assertIn("旁证", state.note)

    def test_file_evidence_keeps_source_ref(self) -> None:
        scan = scan_materials(FIXTURES)
        self.assertTrue(scan.signals)
        for signal in scan.signals:
            self.assertTrue(signal.source_ref)

    def test_self_statement_beats_file_hint_for_level(self) -> None:
        scan = scan_materials(FIXTURES)
        profile = build_profile("我跑过训练，改过 loss", scan=scan, domain=Domain.AI)
        state = profile.dimensions[Dimension.DL]
        self.assertTrue(state.is_known)
        self.assertIs(state.level, Level.WORKING)

    def test_merge_does_not_double_count(self) -> None:
        first = build_profile("我在做系统方向")
        second = build_profile("我在做系统方向，写过 C")
        merged = build_profile_from_signals(
            [s for d in second.dimensions.values() for s in d.signals],
            domain=Domain.SYSTEMS,
        )
        self.assertIs(merged.domain, Domain.SYSTEMS)
        self.assertGreaterEqual(merged.coverage, first.coverage)

    def test_short_text_adds_conservative_note(self) -> None:
        profile = build_profile("写过 C")
        self.assertTrue(any("短" in note for note in profile.notes))

    def test_overall_is_median_not_average(self) -> None:
        profile = build_profile("我在做系统方向，写过 C，用过多线程和 perf")
        levels = sorted(int(d.level) for d in profile.known)
        self.assertEqual(int(profile.overall), levels[len(levels) // 2])


class RenderTests(unittest.TestCase):
    def test_systems_table_mentions_no_deep_learning(self) -> None:
        profile = build_profile("我在做系统方向，写过 C")
        table = render_table(profile)
        self.assertNotIn("深度学习", table)
        self.assertIn("系统与底层", table)

    def test_table_shows_evidence(self) -> None:
        profile = build_profile("我在做系统方向，写过 C")
        self.assertIn("写过 C", render_table(profile))

    def test_profile_report_explains_limits(self) -> None:
        profile = build_profile("我在做系统方向")
        text = render_profile(profile)
        self.assertIn("没有跑测评", text)
        self.assertIn("不等于「不会」", text)

    def test_ai_profile_still_shows_ml(self) -> None:
        profile = build_profile("我在做 AI 方向")
        self.assertIn("机器学习", render_table(profile))


class ProbeTests(unittest.TestCase):
    def test_unknown_domain_asks_domain_first(self) -> None:
        profile = build_profile("我还在看，方向没定")
        question = next_question(profile)
        self.assertEqual(question["target"], "domain")
        self.assertEqual(len(question["options"]), len(list_domains()))

    def test_question_targets_an_active_dimension(self) -> None:
        profile = build_profile("我在做数据库方向")
        question = next_question(profile)
        self.assertIn(question["target"], {d.value for d in profile.dimensions})

    def test_every_dimension_has_a_question(self) -> None:
        for dim in Dimension:
            self.assertIn(dim, DIMENSION_QUESTIONS)

    def test_probe_plan_is_capped(self) -> None:
        profile = build_profile("我在做数据库方向")
        self.assertLessEqual(len(probe_plan(profile, limit=2)), 2)


class RoadmapTests(unittest.TestCase):
    def test_systems_stage_content_is_systems(self) -> None:
        roadmap = build_roadmap(build_profile("我在做系统方向"))
        basics = roadmap.stages[1]
        titles = " ".join(t.title for t in basics.tasks)
        self.assertIn("C", titles)
        self.assertIn("并发", titles)

    def test_ai_stage_content_is_ml(self) -> None:
        roadmap = build_roadmap(build_profile("我在做 AI 方向"))
        titles = " ".join(t.title for t in roadmap.stages[1].tasks)
        self.assertIn("机器", titles)

    def test_database_stage_content_is_sql(self) -> None:
        roadmap = build_roadmap(build_profile("我在做数据库方向"))
        titles = " ".join(t.title for t in roadmap.stages[1].tasks)
        self.assertIn("SQL", titles)

    def test_unknown_domain_gets_explore_tasks(self) -> None:
        roadmap = build_roadmap(build_profile("方向还没定"))
        titles = " ".join(t.title for t in roadmap.stages[1].tasks)
        self.assertIn("方向", titles)
        self.assertIn("论文", titles)

    def test_every_task_has_acceptance(self) -> None:
        for domain in list_domains():
            roadmap = build_roadmap(build_profile("", domain=domain.domain))
            for stage in roadmap.active_stages:
                self.assertTrue(stage.tasks, f"{domain.name} 的 {stage.title} 没有任务")
                for task in stage.tasks:
                    self.assertTrue(task.acceptance, f"{task.title} 缺验收标准")

    def test_skip_only_when_known_and_reached(self) -> None:
        roadmap = build_roadmap(build_profile("我独立写过项目，用过 git 和 docker"))
        stage0 = roadmap.stages[0]
        self.assertTrue(stage0.skipped)
        self.assertIn("达标", stage0.skip_reason)

    def test_unknown_dimensions_never_skip(self) -> None:
        roadmap = build_roadmap(build_profile("方向还没定"))
        self.assertEqual(roadmap.skipped_stages, [])

    def test_hours_limit_defers_later_stages(self) -> None:
        roadmap = build_roadmap(build_profile("我在做系统方向"), hours_limit=20)
        self.assertTrue(roadmap.deferred_stages)

    def test_concepts_on_tasks_exist(self) -> None:
        from research_agent.guide import get_concept

        roadmap = build_roadmap(build_profile("我在做系统方向"))
        for stage in roadmap.active_stages:
            for task in stage.tasks:
                for cid in task.concepts:
                    self.assertIsNotNone(get_concept(cid), cid)

    def test_rendered_roadmap_lists_skip_reasons(self) -> None:
        roadmap = build_roadmap(build_profile("我独立写过项目，用过 git 和 docker"))
        text = render_roadmap(roadmap)
        self.assertIn("已跳过", text)
        self.assertIn("达标", text)


class ConceptTests(unittest.TestCase):
    def test_concept_library_has_no_dangling_refs(self) -> None:
        self.assertEqual(validate_concepts(), [])

    def test_new_domain_concepts_are_searchable(self) -> None:
        for term in ("三次握手", "栈溢出", "B+树", "近似比", "重构"):
            result = explain(term)
            self.assertTrue(result["found"], term)

    def test_missing_concept_is_reported_honestly(self) -> None:
        result = explain("量子纠缠")
        self.assertFalse(result["found"])
        self.assertIn("没有", str(result["candidates"]) + "没有")


class ToolTests(unittest.TestCase):
    def test_assess_returns_domain_and_question(self) -> None:
        data = guide_tools.guide_assess(text="我在做安全方向")
        self.assertTrue(data["ok"])
        self.assertEqual(data["domain"], "security")
        self.assertIn("next_question", data)

    def test_assess_without_input_fails_softly(self) -> None:
        data = guide_tools.guide_assess()
        self.assertFalse(data["ok"])
        self.assertIn("hint", data)

    def test_assess_rejects_url_scan(self) -> None:
        data = guide_tools.guide_assess(text="x", scan_dir="https://example.com/repo")
        self.assertFalse(data["ok"])
        self.assertIn("本地", data["error"])

    def test_roadmap_accepts_domain_name(self) -> None:
        data = guide_tools.guide_roadmap(domain="systems", text="")
        self.assertTrue(data["ok"])
        self.assertIn("系统", data["brief"])

    def test_roadmap_rejects_unknown_domain(self) -> None:
        data = guide_tools.guide_roadmap(domain="炼丹", text="")
        self.assertFalse(data["ok"])
        self.assertIn("domains", data)

    def test_explain_tool_reports_missing(self) -> None:
        data = guide_tools.guide_explain(term="量子纠缠")
        self.assertFalse(data["ok"])
        self.assertIn("hint", data)

    def test_domains_tool_lists_all(self) -> None:
        data = guide_tools.guide_domains()
        self.assertEqual(data["count"], len(list_domains()))

    def test_questions_tool_returns_plan(self) -> None:
        data = guide_tools.guide_questions(text="我在做数据库方向")
        self.assertTrue(data["ok"])
        self.assertGreaterEqual(data["count"], 1)


if __name__ == "__main__":
    unittest.main()
