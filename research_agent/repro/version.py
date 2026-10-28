"""版本约束判断：零依赖实现，不引入 packaging。

支持 pip 风格的约束串：
    "==2.0"  ">=1.13,<2.0"  "~=2.4"  "!=2.1"  "2.4.1"
也支持纯数值比较（显存 40、卡数 8）。

设计原则：解析不了的约束一律返回 unknown，绝不猜成满足——宁可提示人工确认。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_OPS = ("==", ">=", "<=", "!=", "~=", ">", "<", "=")

# 版本号里的数字段，允许 2.4.1 / 2.4.1rc1 / 2.4.1+cu121
_VER_RE = re.compile(r"^(\d+(?:\.\d+)*)")


def parse_version(text: str) -> tuple[int, ...] | None:
    """把版本串转成可比较的整数元组；失败返回 None。"""
    if text is None:
        return None
    m = _VER_RE.match(str(text).strip().lstrip("vV"))
    if not m:
        return None
    parts = m.group(1).split(".")
    return tuple(int(p) for p in parts)


def _pad(a: tuple[int, ...], b: tuple[int, ...]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    n = max(len(a), len(b))
    return a + (0,) * (n - len(a)), b + (0,) * (n - len(b))


def compare(left: str, op: str, right: str) -> bool | None:
    """单条约束判断。返回 True/False，无法判断时返回 None。"""
    if op in ("exists", "present"):
        return bool(left)
    lv = parse_version(left)
    rv = parse_version(right)
    if lv is None or rv is None:
        # 非版本号时退化成字符串相等判断
        if op in ("==", "="):
            return str(left).strip().lower() == str(right).strip().lower()
        return None
    lv, rv = _pad(lv, rv)
    if op in ("==", "="):
        return lv == rv
    if op == "!=":
        return lv != rv
    if op == ">=":
        return lv >= rv
    if op == "<=":
        return lv <= rv
    if op == ">":
        return lv > rv
    if op == "<":
        return lv < rv
    if op == "~=":
        # ~=2.4 等价于 >=2.4, <3.0；~=2.4.1 等价于 >=2.4.1, <2.5
        if len(rv) >= 2 and rv[-1] != 0 and rv[-2] != 0:
            upper = rv[:-1]
            upper = upper[:-1] + (upper[-1] + 1,)
        else:
            upper = (rv[0] + 1,)
        return lv >= rv and lv < upper
    return None


@dataclass
class Constraint:
    """一条或多条用逗号连接的约束。"""

    raw: str

    def specs(self) -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        for chunk in self.raw.split(","):
            chunk = chunk.strip()
            if not chunk:
                continue
            for op in _OPS:
                if chunk.startswith(op):
                    out.append((op, chunk[len(op):].strip()))
                    break
            else:
                out.append(("==", chunk))
        return out

    def satisfied_by(self, version: str | None) -> bool | None:
        """给定版本是否满足全部约束。任一条件无法判断则返回 None。"""
        if version is None or version == "":
            return False
        results = [compare(version, op, val) for op, val in self.specs()]
        if any(r is False for r in results):
            return False
        if any(r is None for r in results):
            return None
        return True

    def describe(self) -> str:
        return self.raw.strip()
