"""Tests for scripts/checks.py.

The first test reproduces the round-15 bug in miniature: a grouping key that
omits a distinguishing field, so a plain dict silently keeps only the last
record per key. The guard must fail loudly on both the losing construction and
on the partition identity.

Run: .venv-c2/bin/python scripts/test_checks.py
"""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from checks import (  # noqa: E402
    SelfCheckFailure, assert_partition, assert_same_total, assert_total,
    bucket_multi, bucket_unique, sum_matches,
)


class PartitionTest(unittest.TestCase):
    def test_catches_round15_bug(self):
        """The exact identity a reviewer had to notice by hand."""
        # Reported in the paper: SnrCommonCount = 5,466 split by mapping.
        with self.assertRaises(SelfCheckFailure) as cm:
            assert_partition({'A': 1537, 'B': 2035}, 5466, 'mapping groups')
        msg = str(cm.exception)
        self.assertIn('3,572', msg)      # thousands-separated in the report
        self.assertIn('5,466', msg)
        self.assertIn('-1,894', msg)     # the deficit is shown explicitly
        self.assertIn('分组键', msg)      # names the likely cause

    def test_corrected_counts_pass(self):
        assert_partition({'A': 2355, 'B': 3111}, 5466, 'mapping groups')

    def test_accepts_sequence(self):
        assert_partition([2355, 3111], 5466, 'seq ok')

    def test_rejects_negative_and_non_integer_parts(self):
        with self.assertRaises(SelfCheckFailure):
            assert_partition([-1, 5467], 5466, 'negative part')
        with self.assertRaises(SelfCheckFailure):
            assert_partition([2355.5, 3110.5], 5466, 'non-integer part')


class BucketTest(unittest.TestCase):
    def test_bucket_unique_rejects_silent_overwrite(self):
        """A key missing a distinguishing field must raise, not overwrite."""
        rows = [
            {'q': 'x', 'rep': 0, 'v': 1},
            {'q': 'x', 'rep': 1, 'v': 2},   # same q, different replicate
            {'q': 'y', 'rep': 0, 'v': 3},
        ]
        with self.assertRaises(SelfCheckFailure) as cm:
            bucket_unique(rows, lambda r: r['q'], 'rows by q')
        self.assertIn('distractor_replicate', str(cm.exception))

        got = bucket_unique(rows, lambda r: (r['q'], r['rep']), 'rows by q+rep')
        self.assertEqual(len(got), 3)

    def test_bucket_multi_keeps_every_record(self):
        rows = [{'q': 'x', 'v': i} for i in range(5)]
        got = bucket_multi(rows, lambda r: r['q'], 'rows by q')
        self.assertEqual(len(got['x']), 5)
        self.assertEqual(sum(len(v) for v in got.values()), len(rows))

    def test_bug_is_silent_without_the_guard(self):
        """Three replicates sharing a question: naive dict keeps one of three."""
        rows = [{'q': 'x', 'rep': i} for i in range(3)]
        naive = {r['q']: r for r in rows}
        self.assertEqual(len(naive), 1)          # silent loss, no error

        with self.assertRaises(SelfCheckFailure):
            bucket_unique(rows, lambda r: r['q'], 'naive key')

        split = bucket_multi(rows, lambda r: r['q'], 'splitting key')
        self.assertEqual(sum(len(v) for v in split.values()), 3)


class TotalsTest(unittest.TestCase):
    def test_assert_total(self):
        assert_total(5466, 5466, 'ok')
        with self.assertRaises(SelfCheckFailure):
            assert_total(3572, 5466, 'bad')

    def test_assert_same_total(self):
        assert_same_total(5466, 5466, 'ok')
        with self.assertRaises(SelfCheckFailure):
            assert_same_total(2355, 3111, 'bad')

    def test_sum_matches_is_non_raising(self):
        self.assertTrue(sum_matches({'A': 2355, 'B': 3111}, 5466, 'ok'))
        self.assertFalse(sum_matches({'A': 1537, 'B': 2035}, 5466, 'bad'))


if __name__ == '__main__':
    unittest.main(verbosity=2)
