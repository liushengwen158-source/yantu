"""论文复现模块的回归测试。

只依赖标准库 unittest：
    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from research_agent.repro import collect_profile, diff, render_matrix, scan_repo
from research_agent.repro.version import Constraint, parse_version

FIXTURES = Path(__file__).parent / "fixtures"


class TestVersion(unittest.TestCase):
    def test_tilde(self):
        self.assertTrue(Constraint("~=2.4").satisfied_by("2.4.1"))
        self.assertFalse(Constraint("~=2.4").satisfied_by("3.0"))

    def test_range(self):
        self.assertTrue(Constraint(">=1.13,<2.0").satisfied_by("1.13.1"))
        self.assertFalse(Constraint(">=1.13,<2.0").satisfied_by("2.0.0"))

    def test_exact(self):
        self.assertTrue(Constraint("==2.4.0").satisfied_by("2.4"))
        self.assertFalse(Constraint("==2.4.0").satisfied_by("2.4.1"))

    def test_unknown(self):
        # 无法判断时返回 None，不能当成满足
        self.assertIsNone(Constraint(">=abc").satisfied_by("1.0"))

    def test_parse(self):
        self.assertEqual(parse_version("2.4.1+cu121"), (2, 4, 1))
        self.assertIsNone(parse_version("not-a-version"))


class TestScanner(unittest.TestCase):
    def test_repo_a_extracts_core_requirements(self):
        spec = scan_repo(FIXTURES / "repo_a")
        ids = {r.id for r in spec.accepted}

        self.assertIn("python.version", ids)
        self.assertIn("torch.version", ids)
        self.assertIn("cuda", ids)
        self.assertIn("gpu.count", ids)

        by_id = {r.id: r for r in spec.accepted}
        self.assertEqual(by_id["python.version"].value, "3.9.7")
        self.assertEqual(by_id["cuda"].value, "11.8")
        self.assertEqual(by_id["gpu.count"].value, "8")

    def test_every_accepted_requirement_has_source(self):
        for repo in ("repo_a", "repo_b"):
            spec = scan_repo(FIXTURES / repo)
            for r in spec.accepted:
                self.assertTrue(r.evidence is not None, f"{r.id} 缺少证据")
                self.assertTrue(r.evidence.source_file, f"{r.id} 缺少来源文件")

    def test_conflict_detected(self):
        spec = scan_repo(FIXTURES / "repo_b")
        joined = " ".join(spec.warnings)
        self.assertIn("torch", joined)
        self.assertIn("11.7", joined)

    def test_missing_dir_raises(self):
        with self.assertRaises(FileNotFoundError):
            scan_repo(FIXTURES / "not_exists")


class TestDiff(unittest.TestCase):
    def test_vram_block_when_gap_too_large(self):
        spec = scan_repo(FIXTURES / "repo_a")
        profile = collect_profile(FIXTURES / "repo_a")
        # 强制一个差距明显的场景，避免依赖真实机器
        profile.gpu_vram_gb = 6.0
        profile.gpu_count = 1
        report = diff(spec, profile)

        by_id = {i.requirement.id: i for i in report.items}
        self.assertEqual(by_id["gpu.vram"].verdict.value, "block")
        self.assertEqual(by_id["gpu.count"].verdict.value, "block")

    def test_torch_install_uses_declared_cuda(self):
        """torch 的安装建议应与仓库声明的 CUDA 一致，而非本机最高版本。"""
        spec = scan_repo(FIXTURES / "repo_a")
        profile = collect_profile(FIXTURES / "repo_a")
        profile.torch_version = None
        profile.driver_cuda = "12.7"
        report = diff(spec, profile)

        by_id = {i.requirement.id: i for i in report.items}
        torch_item = by_id["torch.version"]
        self.assertTrue(torch_item.remediation)
        self.assertIn("cu118", torch_item.remediation[0])

    def test_matrix_renders(self):
        spec = scan_repo(FIXTURES / "repo_a")
        profile = collect_profile(FIXTURES / "repo_a")
        text = render_matrix(diff(spec, profile))
        self.assertIn("需求项", text)
        self.assertIn("可行性", text)


class TestReproTools(unittest.TestCase):
    """agent 工具层：必须返回 dict 且不抛异常。"""

    def test_scan_returns_dict(self):
        from research_agent.tools.repro_tools import repro_scan

        out = repro_scan(str(FIXTURES / "repo_a"))
        self.assertTrue(out["ok"])
        self.assertGreaterEqual(out["count"], 4)
        for item in out["requirements"]:
            self.assertTrue(item["evidence"], "工具返回的需求必须带来源")
            self.assertIn(item["level"], ("L1", "L2", "L3"))

    def test_diff_returns_verdicts(self):
        from research_agent.tools.repro_tools import repro_diff

        out = repro_diff(str(FIXTURES / "repo_a"))
        self.assertTrue(out["ok"])
        self.assertIn(out["overall"], ("block", "conflict", "degrade", "add", "unknown", "ok"))
        self.assertIn("text", out)
        ids = {i["id"] for i in out["items"]}
        self.assertIn("gpu.count", ids)

    def test_report_renders(self):
        from research_agent.tools.repro_tools import repro_report

        out = repro_report(str(FIXTURES / "repo_a"))
        self.assertTrue(out["ok"])
        self.assertIn("论文复现可行性报告", out["text"])

    def test_errors_do_not_raise(self):
        from research_agent.tools import repro_tools

        missing = repro_tools.repro_diff("这是不存在的路径")
        self.assertFalse(missing["ok"])
        self.assertIn("hint", missing)

        url = repro_tools.repro_scan("https://github.com/example/repo")
        self.assertFalse(url["ok"])

        empty = repro_tools.repro_report("")
        self.assertFalse(empty["ok"])

    def test_tool_registry_complete(self):
        from research_agent.tools.repro_tools import REPRO_TOOL_FUNCTIONS

        self.assertEqual(
            set(REPRO_TOOL_FUNCTIONS),
            {"repro_scan", "repro_diff", "repro_report"},
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
