"""Shared self-check helpers for the paper's generated statistics.

Why this module exists
----------------------
Round 15 shipped a wrong statistic. Two mapping groups were built with a
grouping key that omitted `distractor_replicate`, so a plain
`dict[key] = record` silently kept only the last record per key and dropped
1,894 rows. The resulting group sizes were 1,537 and 2,035 instead of 2,355 and
3,111. Nothing raised: both groups looked plausible, the percentages were
computed correctly *within* the broken partition, and only the arithmetic
identity `1537 + 2035 != 5466` exposed it -- an identity a reviewer had to
notice by hand.

The lesson is mechanical, not statistical: a partition that is supposed to
reconstruct a total must be checked against that total, and a key that is
supposed to be unique must be checked for uniqueness rather than silently
overwriting. Both checks are cheap and both belong next to the code that builds
the partition, so this module provides them as small, named assertions.

Every helper raises `SelfCheckFailure` with the label, the observed value, and
the expected value, so a failure names the quantity that broke rather than
surfacing later as a confusing TeX error or a plausible-looking wrong number.

Usage
-----
    from checks import assert_partition, bucket_unique, assert_total

    groups = bucket_unique(rows, lambda r: r['full_key'], 'mapping groups')
    assert_partition({'A': nA, 'B': nB}, known_total, 'mapping groups')
"""
from __future__ import annotations

from collections import defaultdict

__all__ = [
    'SelfCheckFailure', 'assert_partition', 'assert_total',
    'bucket_unique', 'bucket_multi', 'sum_matches', 'check_registry',
    'verified', 'report',
]


class SelfCheckFailure(AssertionError):
    """Raised when a generated statistic fails an internal consistency check."""


# Names of checks that passed, so a run can print what was actually verified
# rather than only what failed.
_PASSED: list[str] = []


def verified(label: str) -> None:
    """Record that `label` was checked and passed."""
    _PASSED.append(label)


def check_registry() -> list[str]:
    """Labels of every check that has passed in this process."""
    return list(_PASSED)


def report(stream=None) -> None:
    """Print the labels of all checks that passed."""
    import sys
    stream = stream if stream is not None else sys.stdout
    if _PASSED:
        stream.write(f'  自检通过 {len(_PASSED)} 项: ' + '; '.join(_PASSED) + '\n')


def _fmt(x) -> str:
    if isinstance(x, float):
        return f'{x:,.4g}'
    return f'{x:,}' if isinstance(x, int) else str(x)


def assert_total(observed, expected, label: str, *, detail: str = '') -> None:
    """Assert a directly computed total equals the value it must equal.

    Use when a single count has an independent source of truth, e.g. a count
    recomputed here against the count another build script already published.
    """
    if observed != expected:
        raise SelfCheckFailure(
            f'{label}: 观察到 {_fmt(observed)}, 但应为 {_fmt(expected)}'
            + (f' ({detail})' if detail else '')
            + '\n  提示: 若这是与另一脚本的交叉核对, 两者必须来自同一筛选口径;'
              '\n        口径不同(如分组键少一个字段)会让同一条记录被静默覆盖。')
    verified(label)


def assert_partition(parts, total, label: str, *, detail: str = '') -> None:
    """Assert disjoint parts sum to a known total.

    `parts` may be a mapping of name -> count or a sequence of counts. This is
    the guard for the round-15 bug: the mapping groups were supposed to
    partition the 5,466 common-correct units, and checking
    `1537 + 2035 == 5466` would have failed immediately.

    A non-integer or negative part is also a failure, since a partition of a
    count cannot contain one.
    """
    counts = list(parts.values()) if hasattr(parts, 'values') else list(parts)
    names = list(parts.keys()) if hasattr(parts, 'keys') else None
    for i, c in enumerate(counts):
        if not isinstance(c, int) or isinstance(c, bool):
            raise SelfCheckFailure(
                f'{label}: 分项 {names[i] if names else i} 不是整数 ({c!r}); '
                f'分项合计自检只适用于计数。')
        if c < 0:
            raise SelfCheckFailure(
                f'{label}: 分项 {names[i] if names else i} 为负数 ({c})。')
    got = sum(counts)
    if got != total:
        breakdown = ', '.join(
            f'{n}={_fmt(c)}' for n, c in zip(names, counts)) if names else \
            ', '.join(_fmt(c) for c in counts)
        raise SelfCheckFailure(
            f'{label}: 分项合计 {_fmt(got)} != 已知总体 {_fmt(total)}'
            f' (差 {_fmt(got - total)})\n  分项: {breakdown}'
            + (f'\n  {detail}' if detail else '')
            + '\n  提示: 这通常意味着分组键漏了字段, 导致同一键的多条记录'
              '\n        互相覆盖; 请核对分组键是否唯一标识每条记录。')
    verified(label)


def assert_same_total(a, b, label: str) -> None:
    """Two independent countings of the same population must agree."""
    if a != b:
        raise SelfCheckFailure(
            f'{label}: 两种口径不一致 {_fmt(a)} vs {_fmt(b)}')
    verified(label)


def bucket_unique(records, keyf, label: str):
    """Group records by `keyf`, requiring each key to be unique.

    Raises on a collision instead of silently replacing the earlier record.
    Use this whenever the key is meant to identify a single record; a collision
    means the key is under-specified, which is exactly how 1,894 rows vanished
    in round 15.
    """
    out = {}
    first = {}
    for r in records:
        k = keyf(r)
        if k in out:
            raise SelfCheckFailure(
                f'{label}: 键 {k!r} 出现多次, 普通字典会静默覆盖前面的记录。\n'
                f'  第一条: {first[k]!r}\n  冲突条: {r!r}\n'
                f'  提示: 键很可能缺少区分字段(例如 distractor_replicate),'
                f'\n        或者此处本应使用 bucket_multi 收集同键多条记录。')
        out[k] = r
        first[k] = r
    verified(label)
    return out


def bucket_multi(records, keyf, label: str):
    """Group records by `keyf` into lists, keeping every record.

    The intentional many-per-key counterpart to `bucket_unique`. Asserts that
    no record was lost, so a caller that expects exactly the input population
    cannot be silently short-changed.
    """
    out = defaultdict(list)
    for r in records:
        out[keyf(r)].append(r)
    kept = sum(len(v) for v in out.values())
    if kept != len(records):
        raise SelfCheckFailure(
            f'{label}: 入组 {kept} 条, 但输入 {len(records)} 条, 有记录丢失。')
    verified(label)
    return out


def sum_matches(parts, total, label: str) -> bool:
    """Non-raising form of assert_partition, for diagnostics/tests."""
    try:
        assert_partition(parts, total, label)
        return True
    except SelfCheckFailure:
        return False
