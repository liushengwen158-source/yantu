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

from research_agent.guide import (
    Confidence,
    Dimension,
    Level,
    build_profile,
    merge_profiles,
    next_question,
)


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


if __name__ == "__main__":
    unittest.main()
