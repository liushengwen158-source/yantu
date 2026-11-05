"""建图：从一篇种子论文出发，拉出它的"上游 / 下游 / 近邻"。

四条边、四种含义，刻意分开存（画的时候也分开画）：

| 关系 | 含义 | 数据来源 |
| --- | --- | --- |
| ``cites`` | 种子引用了谁 —— 这是它的**地基** | OpenAlex ``referenced_works`` |
| ``cited_by`` | 谁引用了种子 —— 这是它的**后续** | OpenAlex ``filter=cites:W...`` |
| ``similar`` | 站方给的近邻 —— **相似不等于因果** | OpenAlex ``related_works`` |
| ``coupling`` | 两篇共享参考文献 —— 才是"同一条技术路线" | 由 ``referenced_works`` 二次计算 |

``coupling`` 是这里唯一"算出来"的关系，也是最容易被误用的：共享 2 篇以上参考文献
才连边，且只连**同侧**（上游之间、下游之间），不跨侧连——跨侧连出来的图没法解释。

每条边都带 ``provenance``（哪个源的哪次请求、什么时候）。没有 provenance 的边不画。
"""

from __future__ import annotations

import time
from collections import defaultdict

from .cache import Cache
from .identity import key_of, title_fingerprint
from .spec import Coverage, Edge, PaperGraph, PaperRef, Relation, SourceStatus
from .sources import SourceError, build_sources

MAX_REFS = 25
MAX_CITERS = 20
MAX_RELATED = 12
MIN_SHARED_REFS = 2


def _stamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _find_seed(paper: PaperRef, oa) -> PaperRef:
    """把种子变成一条带 OpenAlex id 的记录。拿不到就抛 SourceError。"""
    if paper.openalex_id:
        return paper
    if paper.doi:
        got = oa.by_doi(paper.doi)
        if got:
            return got
    if paper.title:
        cands = oa.search(f'title.search:"{paper.title[:120]}"', limit=3)
        fp = title_fingerprint(paper.title)
        for c in cands:
            if title_fingerprint(c.title) == fp:
                return c
        if cands and title_fingerprint(cands[0].title)[:60] == fp[:60]:
            return cands[0]
    raise SourceError("openalex", "无法把种子论文对应到 OpenAlex 记录", retryable=False)


def _coupling(nodes: dict[str, PaperRef], side: str) -> list[Edge]:
    """同侧共享参考文献 ≥ N 篇 → 文献耦合并边。"""
    shared: dict[tuple[str, str], int] = defaultdict(int)
    keys = list(nodes)
    ref_sets = {
        k: set((nodes[k].extra.get("referenced_works") or [])) for k in keys
    }
    for i, a in enumerate(keys):
        ra = ref_sets[a]
        if len(ra) < MIN_SHARED_REFS:
            continue
        for b in keys[i + 1:]:
            rb = ref_sets[b]
            if not rb:
                continue
            overlap = len(ra & rb)
            if overlap >= MIN_SHARED_REFS:
                shared[(a, b)] = overlap
    out = []
    for (a, b), n in shared.items():
        out.append(Edge(
            src=a, dst=b, relation=Relation.COUPLING,
            provenance=f"computed:referenced_works∩ ({side}) {_stamp()}",
            context=f"共享 {n} 篇参考文献",
        ))
    return out


def _is_container(p: PaperRef) -> bool:
    """OpenAlex 的 ``referenced_works`` 里混着"容器记录"（会议论文集、书章节壳子）。

    判据用三条同时成立：**没有作者 + 没有摘要 + 没有参考文献**。
    实测（种子 = Deep learning, LeCun 2015，54 个节点）只有 1 条命中：
    "Neural Information Processing Systems"（book-chapter，0 作者 0 引用 0 参考文献）——
    它显然不是一篇论文，而是个论文集壳子。宁可漏删，也不要按标题猜：
    按标题关键词过滤会误杀 "Neural Information Processing Systems" 这种**正当地**
    出现在真论文标题里的词。
    """
    return (
        not p.authors
        and not p.abstract
        and not (p.extra.get("referenced_works") or [])
    )


def _filter_bucket(bucket: dict[str, PaperRef], graph: PaperGraph, side: str) -> dict[str, PaperRef]:
    kept = {k: p for k, p in bucket.items() if not _is_container(p)}
    dropped = len(bucket) - len(kept)
    if dropped:
        graph.warnings.append(f"{side}有 {dropped} 条记录缺作者/摘要/参考文献，按容器记录丢弃")
        for k in set(bucket) - set(kept):
            graph.nodes.pop(k, None)
        graph.edges = [e for e in graph.edges if e.src in graph.nodes and e.dst in graph.nodes]
    return kept


def expand(
    seed: PaperRef,
    *,
    cache: Cache | None = None,
    offline: bool = False,
    include_citers: bool = True,
    include_related: bool = True,
) -> PaperGraph:
    """从种子扩一跳。失败不抛异常，把原因写进 ``warnings`` 与 ``coverage``。"""
    cache = cache or Cache(offline=offline)
    sources = build_sources(cache, order=("openalex",))
    graph = PaperGraph(warnings=[])
    if not sources:
        graph.warnings.append("OpenAlex 适配器不可用，无法建图")
        return graph
    oa = sources[0]
    stamp = _stamp()

    try:
        seed_resolved = _find_seed(seed, oa)
    except SourceError as exc:
        graph.coverage = Coverage([SourceStatus(source=oa.name, ok=False, error=exc.message)])
        graph.warnings.append(f"种子解析失败：{exc.message}")
        return graph

    seed_key = key_of(seed_resolved)
    graph.seed = seed_key
    graph.nodes[seed_key] = seed_resolved
    refs = list(seed_resolved.extra.get("referenced_works") or [])
    related = list(seed_resolved.extra.get("related_works") or [])
    status = SourceStatus(source=oa.name, ok=True, attempts=1)

    # ---- 上游：种子引用的文献 -------------------------------------------
    upstream: dict[str, PaperRef] = {}
    if refs:
        try:
            for p in oa.by_ids(refs[:MAX_REFS], limit=MAX_REFS):
                k = key_of(p)
                upstream[k] = p
                graph.nodes[k] = p
                graph.add_edge(Edge(seed_key, k, Relation.CITES,
                                    provenance=f"openalex:referenced_works {stamp}"))
        except SourceError as exc:
            graph.warnings.append(f"取上游文献失败：{exc.message}")
            status.ok = False
    else:
        graph.warnings.append("种子在 OpenAlex 上没有 referenced_works，上游为空")

    # ---- 下游：引用种子的文献 -------------------------------------------
    downstream: dict[str, PaperRef] = {}
    if include_citers and seed_resolved.openalex_id:
        try:
            for p in oa.citing(seed_resolved.openalex_id, limit=MAX_CITERS):
                k = key_of(p)
                downstream[k] = p
                graph.nodes[k] = p
                graph.add_edge(Edge(k, seed_key, Relation.CITED_BY,
                                    provenance=f"openalex:cites:{seed_resolved.openalex_id} {stamp}"))
            if not downstream:
                graph.warnings.append("还没有论文引用它（新论文常见），下游为空")
        except SourceError as exc:
            graph.warnings.append(f"取下游文献失败：{exc.message}")
            status.ok = False

    # ---- 近邻：站方推荐 -------------------------------------------------
    if include_related and related:
        try:
            for p in oa.by_ids(related[:MAX_RELATED], limit=MAX_RELATED):
                k = key_of(p)
                graph.nodes[k] = p
                graph.add_edge(Edge(seed_key, k, Relation.SIMILAR,
                                    provenance=f"openalex:related_works {stamp}"))
        except SourceError as exc:
            graph.warnings.append(f"取近邻文献失败：{exc.message}")
            status.ok = False

    # ---- 二次计算：同侧文献耦合 -----------------------------------------
    upstream = _filter_bucket(upstream, graph, "上游")
    downstream = _filter_bucket(downstream, graph, "下游")
    for side, bucket in (("upstream", upstream), ("downstream", downstream)):
        if len(bucket) >= 2:
            for e in _coupling(bucket, side):
                graph.add_edge(e)

    status.count = len(graph.nodes)
    graph.coverage = Coverage([status])
    return graph
