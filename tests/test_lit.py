"""文献模块的离线回归测试。**全程不联网**：所有网络响应都来自 fixtures。

覆盖六件事，每一件都对应设计里的一条硬约束：

1. 身份对齐：DOI/arXiv 优先，模糊匹配只标不并（合并错了不可逆）。
2. 适配器解析：三个源的返回都能变成 PaperRef，摘要倒排索引能还原。
3. 检索编排：源失败要降级并记账，偏题结果要被过滤并报数。
4. 阅读建议：四档判定与理由，方向不能反（综述该排前面，跑题该被丢）。
5. 渲染：SVG 结构正确、不含 NaN、环能被检测出来。
6. 缓存：离线模式下没缓存要报"该源不可用"，而不是抛到用户脸上。
"""

from __future__ import annotations

import json
import re
import tempfile
import unittest
from pathlib import Path

import importlib

from research_agent.litreview import identity, render
from research_agent.litreview.triage import SIGNAL_KEYS, triage as run_triage

# 注意：包的 __init__ 里 re-export 了同名函数，``from ... import search`` 拿到的是函数，
# 这里要的是模块本身（测试要 patch 模块里的 build_sources），所以走 import_module。
search_mod = importlib.import_module("research_agent.litreview.search")
from research_agent.litreview.cache import Cache
from research_agent.litreview.sources.arxiv import parse_feed
from research_agent.litreview.sources.base import Http, SourceError
from research_agent.litreview.sources.crossref import to_paper as crossref_to_paper
from research_agent.litreview.sources.openalex import to_paper as openalex_to_paper
from research_agent.litreview.spec import PaperRef, Source

FIXTURES = Path(__file__).parent / "fixtures" / "lit"


def load(name: str):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def paper(**kw) -> PaperRef:
    base = dict(source=Source.OPENALEX, source_id="W1", title="t", year=2024, cited_by=0)
    base.update(kw)
    return PaperRef(**base)


class TestIdentity(unittest.TestCase):
    def test_normalize_doi(self):
        for raw in ("https://doi.org/10.1000/ABC", "doi:10.1000/abc", "10.1000/abc."):
            self.assertEqual(identity.normalize_doi(raw), "10.1000/abc")

    def test_normalize_arxiv_drops_version(self):
        for raw in ("arXiv:2401.01234v2", "abs/2401.01234", "2401.01234v9"):
            self.assertEqual(identity.normalize_arxiv(raw), "2401.01234")

    def test_same_doi_merges_and_keeps_max_citations(self):
        a = paper(source=Source.OPENALEX, source_id="W1", doi="10.1/x", cited_by=10)
        b = paper(source=Source.CROSSREF, source_id="10.1/x", doi="https://doi.org/10.1/X", cited_by=99)
        reps, ids, _ = identity.resolve([a, b])
        self.assertEqual(len(reps), 1)
        self.assertEqual(reps[0].cited_by, 99)
        self.assertEqual(ids["doi:10.1/x"].match_method, "doi")
        self.assertEqual(len(ids["doi:10.1/x"].merged), 2)

    def test_title_match_is_flagged_not_merged(self):
        a = paper(source=Source.OPENALEX, source_id="W1", title="Retrieval Augmented Generation")
        b = paper(source=Source.ARXIV, source_id="2401.00001", arxiv_id="2401.00001",
                  title="retrieval-augmented  generation!")
        reps, ids, warns = identity.resolve([a, b])
        self.assertEqual(len(reps), 2, "没有共同 id 时不许合并")
        all_dups = [k for i in ids.values() for k in i.possible_duplicates]
        self.assertTrue(all_dups, "同标题应被标记为疑似重复")
        self.assertTrue(any("疑似" in w or "待人工确认" in w for w in warns))

    def test_real_arxiv_id_wins_over_title(self):
        a = paper(source=Source.ARXIV, source_id="2401.01234v2", arxiv_id="2401.01234v2",
                  title="Some Title")
        key = identity.key_of(a)
        self.assertEqual(key, "arxiv:2401.01234")


class TestAdapters(unittest.TestCase):
    def test_arxiv_feed_parses(self):
        xml = (FIXTURES / "arxiv_feed.xml").read_text(encoding="utf-8")
        items = parse_feed(xml, "2026-11-05T12:00:00+0800")
        self.assertEqual(len(items), 2)
        first = items[0]
        self.assertEqual(first.arxiv_id, "2401.01234")
        self.assertEqual(first.year, 2024)
        self.assertEqual(first.authors, ["Ada Lovelace", "Alan Turing"])
        self.assertTrue(first.is_survey, "标题含 Survey 应被标成综述")
        self.assertEqual(first.fields, ["cs.RO"])

    def test_openalex_abstract_inverted_index_restored(self):
        data = load("openalex_search.json")
        p = openalex_to_paper(data["results"][0], "2026-11-05T12:00:00+0800")
        self.assertEqual(p.abstract, "retrieval augmented generation for knowledge")
        self.assertEqual(p.openalex_id, "W1000000001")
        self.assertEqual(p.doi, "10.1000/abc.2023.1")
        self.assertEqual(len(p.extra["referenced_works"]), 2)

    def test_crossref_paper_keeps_reference_dois(self):
        data = load("crossref_works.json")
        p = crossref_to_paper(data["message"]["items"][0], "2026-11-05T12:00:00+0800")
        self.assertEqual(p.doi, "10.1000/abc.2023.1")
        self.assertEqual(p.year, 2023)
        self.assertEqual(p.extra["references"], ["10.1000/ref.1", "10.1000/ref.2"])

    def test_survey_detection_excludes_peer_review_records(self):
        """实测踩过：Crossref 的 'Review of: "X"' 是同行评审记录，不是综述，曾被顶到读第一条。"""
        from research_agent.litreview.sources.base import looks_like_survey

        self.assertTrue(looks_like_survey("A Survey of Retrieval Augmented Generation"))
        self.assertTrue(looks_like_survey("Retrieval Augmented Generation: An Overview"))
        self.assertFalse(looks_like_survey('Review of: "GeAR: Generation Augmented Retrieval"'))
        self.assertFalse(looks_like_survey("Peer review report on RAG benchmarks"))
        self.assertFalse(looks_like_survey(""))


class _StubSource:
    def __init__(self, name: Source, papers=None, error: str = ""):
        self.name = name
        self._papers = papers or []
        self._error = error

    def search(self, query, limit=20, year_from=None):
        if self._error:
            raise SourceError(self.name.value, self._error)
        return list(self._papers)


def _patch_sources(stubs):
    original = search_mod.build_sources
    search_mod.build_sources = lambda cache, order=None: stubs
    return original


class TestContainerFilter(unittest.TestCase):
    """OpenAlex 的 referenced_works 里会混进"论文集壳子"，不能当论文画进图里。"""

    def test_container_record_is_detected(self):
        from research_agent.litreview.graph import _is_container

        shell = paper(title="Neural Information Processing Systems", authors=[], abstract="")
        real = paper(title="Long Short-Term Memory", authors=["Hochreiter"], abstract="")
        self.assertTrue(_is_container(shell))
        self.assertFalse(_is_container(real), "有作者的真实论文不许被误删")


class TestSearchOrchestration(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cache = Cache(root=Path(self.tmp.name))

    def tearDown(self):
        self.tmp.cleanup()

    def test_degraded_coverage_is_reported(self):
        on_topic = paper(title="Retrieval Augmented Generation for QA", year=2024)
        stubs = [_StubSource(Source.OPENALEX, [on_topic]), _StubSource(Source.S2, error="被限流（429）")]
        original = _patch_sources(stubs)
        try:
            r = search_mod.search("retrieval augmented generation", cache=self.cache)
        finally:
            search_mod.build_sources = original
        self.assertTrue(r.coverage.degraded)
        self.assertIn("s2", r.coverage.failed_sources)
        self.assertIn("不可用", r.coverage.note())
        self.assertEqual(len(r.papers), 1)

    def test_off_topic_results_are_dropped_and_counted(self):
        good = paper(source=Source.CROSSREF, source_id="10.1/a",
                     title="Retrieval Augmented Generation for Question Answering", year=2024)
        bad = paper(source=Source.CROSSREF, source_id="10.1/b",
                    title="Projector augmented-wave method", year=1994, cited_by=91716)
        original = _patch_sources([_StubSource(Source.CROSSREF, [good, bad])])
        try:
            r = search_mod.search("retrieval augmented generation question answering",
                                  cache=self.cache)
        finally:
            search_mod.build_sources = original
        self.assertEqual(len(r.papers), 1)
        self.assertEqual(r.papers[0].title[:9], "Retrieval")
        self.assertGreaterEqual(r.dropped, 1)
        self.assertTrue(any("偏题" in w for w in r.warnings))

    def test_zero_terms_still_returns_something(self):
        p = paper(title="Anything at all")
        original = _patch_sources([_StubSource(Source.OPENALEX, [p])])
        try:
            r = search_mod.search("the of and", cache=self.cache)
        finally:
            search_mod.build_sources = original
        self.assertEqual(search_mod.terms_of("the of and"), [])
        self.assertEqual(len(r.papers), 1, "问题里全是停用词时不许把结果全丢掉")


class TestTriage(unittest.TestCase):
    def test_survey_is_read_now(self):
        p = paper(title="A Survey of Retrieval Augmented Generation", year=2024,
                  cited_by=20, is_survey=True)
        t = run_triage([p], "retrieval augmented generation")[0]
        self.assertEqual(t.verdict.value, "read_now")
        self.assertIn("综述", t.reason)

    def test_off_topic_is_dropped(self):
        p = paper(title="Projector augmented-wave method", year=1994, cited_by=91716)
        t = run_triage([p], "retrieval augmented generation for question answering")[0]
        self.assertEqual(t.verdict.value, "drop")

    def test_brand_new_high_relevance_is_skim(self):
        p = paper(title="Retrieval Augmented Generation for Question Answering", year=2026,
                  cited_by=1)
        t = run_triage([p], "retrieval augmented generation question answering")[0]
        self.assertEqual(t.verdict.value, "skim")
        self.assertIn("还没有足够引用验证", t.reason)

    def test_old_and_low_cited_is_drop(self):
        p = paper(title="Retrieval Augmented Generation for Question Answering",
                  year=1990, cited_by=0)
        t = run_triage([p], "retrieval augmented generation question answering")[0]
        self.assertEqual(t.verdict.value, "drop")

    def test_signals_are_exported_for_audit(self):
        p = paper(title="Retrieval Augmented Generation", year=2025, cited_by=3)
        t = run_triage([p], "retrieval augmented generation")[0]
        for k in SIGNAL_KEYS:
            self.assertIn(k, t.signals)


class TestRender(unittest.TestCase):
    def test_papers_svg_is_valid(self):
        papers = [
            ("k1", paper(title="Retrieval Augmented Generation", year=2023, cited_by=50)),
            ("k2", paper(title="Another Paper", year=2024, cited_by=1)),
        ]
        svg = render.papers_svg(papers, question="rag")
        self.assertTrue(svg.startswith("<svg"))
        self.assertEqual(svg.count("<circle"), 2)
        self.assertNotIn("nan", svg.lower())
        self.assertIn("2023", svg)

    def test_papers_svg_handles_missing_years(self):
        svg = render.papers_svg([("k1", paper(title="No Year", year=None))])
        self.assertIn("<svg", svg)
        self.assertIn("没有带年份", svg)

    def test_concept_cycle_detected(self):
        concepts = load("concepts_small.json")["concepts"]
        stats = render.concept_stats(concepts)
        self.assertEqual(stats["concepts"], 5)
        self.assertEqual(len(stats["cycles"]), 1)
        self.assertEqual(sorted(stats["cycles"][0]), sorted(["终端基本操作", "PATH 与环境变量"]))
        self.assertEqual(stats["cross_dimension"], 3)  # c←a、d←c、e←c 三条跨维度

    def test_concepts_svg_marks_cycle_edges_red(self):
        concepts = load("concepts_small.json")["concepts"]
        svg = render.concepts_svg(concepts)
        self.assertIn("#e8684a", svg, "环上的边要用红色虚线标出来")
        self.assertEqual(svg.count("<rect"), 5)
        self.assertNotIn("nan", svg.lower())

    def test_html_page_is_self_contained(self):
        page = render.html_page("t", "s", [("a", "<svg></svg>"), ("b", "x")])
        self.assertIn("<style>", page)
        self.assertIn("<script>", page)
        self.assertNotIn("http://", page.replace("http://www.w3.org", ""), "不许引外部资源")

    def test_markdown_report_puts_trust_info_first(self):
        p = paper(title="Retrieval Augmented Generation", year=2024, cited_by=10)
        r = search_mod.Retrieval(question="rag", queries=["rag"], papers=[p])
        text = render.markdown_report(r, run_triage([p], "rag"))
        self.assertLess(text.index("检索式"), text.index("结果"), "可信度信息要排在结论前面")
        self.assertIn("覆盖度", text)


class TestCache(unittest.TestCase):
    def test_put_then_get(self):
        with tempfile.TemporaryDirectory() as tmp:
            c = Cache(root=Path(tmp))
            c.put("https://example.org/x", {"a": 1})
            self.assertEqual(c.get("https://example.org/x"), {"a": 1})
            self.assertEqual(c.stats()["hits"], 1)

    def test_offline_without_cache_raises_source_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            c = Cache(root=Path(tmp), offline=True)
            http = Http(cache=c)
            with self.assertRaises(SourceError) as ctx:
                http.get_text("https://example.org/nothing", source="openalex")
            self.assertFalse(ctx.exception.retryable, "离线未命中不是可重试错误")

    def test_offline_reuses_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            Cache(root=Path(tmp)).put("https://example.org/x", {"a": 1})
            http = Http(cache=Cache(root=Path(tmp), offline=True))
            res = http.get_text("https://example.org/x", source="openalex")
            self.assertTrue(res.cached)
            self.assertEqual(res.payload, {"a": 1})


if __name__ == "__main__":
    unittest.main()
