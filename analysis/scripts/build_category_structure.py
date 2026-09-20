"""Where the ranking failures live: source-event structure and its transfer.

Input : generated/counterfactual_inputs.json  frozen panel (8 models, baseline view)
        generated/repair_inputs.json          re-paired panel (3 models)
        generated/ten_model_panel.json        the eight paper models
Output: generated/category_structure.tex, generated/category_structure.json

Sec. 5.1 fixes the phenomenon at the answer level: a correct mixture answer can
still be a ranking failure. This script asks whether those failures are spread
evenly over the 49 source events or concentrate on particular ones, and whether
that concentration is a property of the event rather than of the single
distractor pairing the frozen panel happens to use.

Categories are grouped into quintiles by their frozen-panel rate, using every
category rather than hand-picked extremes, so the grouping carries no
selection inflation. The test is transfer: the quintile ordering defined on the
frozen panel is re-scored on the independently re-paired panel, and the full
per-category ordering is correlated across panels. To rule out the reading that
the split merely rescales how confidently the answer is given, the same
contrast is recomputed inside bins of the mixture margin m_f.
"""
import collections
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from texnames import write_macros

ROOT = Path(__file__).resolve().parents[1]
GEN = ROOT / 'generated'
SEED, DRAWS = 20260907, 10000
MIN_CORRECT = 30   # categories with fewer correct answers cannot be ranked
N_QUINT = 5


def rng_ok(u):
    return u['mf'] is not None and u['mt'] is not None and u['mj']


def is_rf(u):
    """Ranking failure: correct answer, yet no threshold orders the views."""
    return u['mf'] > 0 and min(u['mf'], *u['mj']) <= u['mt']


def avg_rank(x):
    """Ordinal ranks with ties averaged, so tied categories order consistently."""
    x = np.asarray(x, float)
    order = np.argsort(x, kind='mergesort')
    sx = x[order]
    ranks = np.empty(len(x))
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and sx[j + 1] == sx[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def spearman(a, b):
    return float(np.corrcoef(avg_rank(a), avg_rank(b))[0, 1])


def tally(units, cats):
    """Per category: correct answers, ranking failures, all answers."""
    num, den, tot = collections.Counter(), collections.Counter(), collections.Counter()
    for u in units:
        c = u['target_category']
        if c not in cats:
            continue
        tot[c] += 1
        if u['mf'] > 0:
            den[c] += 1
            num[c] += is_rf(u)
    return num, den, tot


def group_stats(units, grp):
    """Pooled rate over a group of categories, with counts."""
    num, den, tot = tally(units, set(grp))
    n = sum(num.values())
    d = sum(den.values())
    a = sum(tot.values())
    return dict(rf=100.0 * n / d if d else float('nan'), rf_n=n, correct=d, all=a,
                acc=100.0 * d / a if a else float('nan'))


def boot_ci(units, a_grp, b_grp, cluster, seed):
    """Percentile CI for the A-minus-B gap, resampling whole clusters."""
    keys = sorted({u[cluster] for u in units})
    idx = {k: i for i, k in enumerate(keys)}
    A = np.zeros((len(keys), 2))
    B = np.zeros((len(keys), 2))
    for u in units:
        if u['mf'] <= 0:
            continue
        c = u['target_category']
        row = A if c in a_grp else B if c in b_grp else None
        if row is None:
            continue
        i = idx[u[cluster]]
        row[i, 1] += 1
        row[i, 0] += is_rf(u)
    ix = np.random.default_rng(seed).integers(0, len(keys), (DRAWS, len(keys)))
    ra, rb = A[ix].sum(1), B[ix].sum(1)
    d = 100.0 * (ra[:, 0] / np.maximum(ra[:, 1], 1) - rb[:, 0] / np.maximum(rb[:, 1], 1))
    lo, hi = np.percentile(d, [2.5, 97.5])
    return float(lo), float(hi)


def quintiles(rows):
    """Split an ordered category list into N_QUINT near-equal groups."""
    out, i = [], 0
    for k in range(N_QUINT):
        n = len(rows) // N_QUINT + (1 if k < len(rows) % N_QUINT else 0)
        out.append(rows[i:i + n])
        i += n
    return out


def mix(units, grp, key):
    c = collections.Counter()
    for u in units:
        if u['mf'] > 0 and u['target_category'] in grp:
            c[u[key]] += 1
    tot = sum(c.values())
    return {k: 100.0 * v / tot for k, v in c.items()}


def slashes(d, order):
    return '/'.join('%d' % round(d.get(k, 0.0)) for k in order)


def label(c):
    return c if c.isupper() else c[0].upper() + c[1:]


def main():
    frozen_doc = json.loads((GEN / 'counterfactual_inputs.json').read_text())
    repair_doc = json.loads((GEN / 'repair_inputs.json').read_text())
    paper_models = json.loads((GEN / 'ten_model_panel.json').read_text())['paper_models']

    fu = [u for u in frozen_doc['panels']['baseline'] if rng_ok(u)]
    assert {u['model_id'] for u in fu} == set(paper_models), 'frozen panel model mismatch'
    ru = [u for u in repair_doc['units'] if rng_ok(u)]

    fnum, fden, ftot = tally(fu, {u['target_category'] for u in fu})
    assert all(fden[c] >= MIN_CORRECT for c in fden), 'a category fell below the minimum sample'
    rows = sorted(fden, key=lambda c: (-fnum[c] / fden[c], c))
    qs = quintiles(rows)
    assert not any(set(a) & set(b) for i, a in enumerate(qs) for b in qs[i + 1:])

    fr = [group_stats(fu, q) for q in qs]
    rr = [group_stats(ru, q) for q in qs]
    assert all(fr[i]['rf'] >= fr[i + 1]['rf'] for i in range(N_QUINT - 1)), 'frozen trend not monotone'
    assert all(rr[i]['rf'] >= rr[i + 1]['rf'] for i in range(N_QUINT - 1)), 'repair trend not monotone'
    gap = fr[0]['rf'] - fr[-1]['rf']
    glo, ghi = boot_ci(fu, set(qs[0]), set(qs[-1]), 'source_tuple_id', SEED)
    rgap = rr[0]['rf'] - rr[-1]['rf']
    rlo, rhi = boot_ci(ru, set(qs[0]), set(qs[-1]), 'role_index', SEED)

    # Transfer of the whole ordering, and of the groups, to the re-paired panel.
    shared = [c for c in rows if fden[c] >= MIN_CORRECT]
    fa = [fnum[c] / fden[c] for c in shared]
    ra = [fnum[c] / fden[c] for c in shared]
    rnum, rden, _ = tally(ru, set(shared))
    ra = [rnum[c] / rden[c] for c in shared]
    rho = spearman(fa, ra)
    rng = np.random.default_rng(SEED)
    perm = np.array([spearman(fa, rng.permutation(np.asarray(ra))) for _ in range(DRAWS)])
    pval = (1 + int((perm >= rho).sum())) / (DRAWS + 1)

    models = sorted({u['model_id'] for u in ru})
    per_model, ok = {}, 0
    for m in models:
        um = [u for u in ru if u['model_id'] == m]
        a = group_stats(um, qs[0])['rf']
        b = group_stats(um, qs[-1])['rf']
        per_model[m] = [a, b]
        ok += a > b
    assert ok == len(models), 'a re-paired model reverses the category gap'

    # Is the split only a rescaling of how confidently the answer is given?
    bins = [(0, .5), (.5, 1), (1, 2), (2, 3), (3, 5)]
    mgaps = []
    for lo, hi in bins:
        sel = [u for u in fu if lo <= u['mf'] < hi]
        a, b = group_stats(sel, qs[0]), group_stats(sel, qs[-1])
        assert a['correct'] >= 50 and b['correct'] >= 50, f'margin bin [{lo},{hi}) too sparse'
        mgaps.append(a['rf'] - b['rf'])

    # The quintiles must partition the same answers the main table counts.
    tot_rf = sum(st['rf_n'] for st in fr)
    tot_ok = sum(st['correct'] for st in fr)
    assert tot_ok == len([u for u in fu if u['mf'] > 0]), 'quintiles do not cover the panel'
    assert tot_rf == len([u for u in fu if is_rf(u)]), 'quintile failures do not sum to the panel'

    mac = {
        # The count of ranked categories is the same 49 the main table reports;
        # \CatCount already provides it, so it is not redefined here.
        'CatHighest': label(rows[0]),
        'CatLowest': label(rows[-1]),
        'CatHighestFail': f'{100*fnum[rows[0]]/fden[rows[0]]:.1f}\\%',
        'CatHighestN': str(fden[rows[0]]),
        'CatSecond': label(rows[1]),
        'CatSecondFail': f'{100*fnum[rows[1]]/fden[rows[1]]:.1f}\\%',
        'CatThird': label(rows[2]),
        'CatThirdFail': f'{100*fnum[rows[2]]/fden[rows[2]]:.1f}\\%',
        'CatLowestFail': f'{100*fnum[rows[-1]]/fden[rows[-1]]:.1f}\\%',
        'CatLowestN': str(fden[rows[-1]]),
        'CatGap': f'{gap:+.1f}',
        'CatGapCI': f'[{glo:.1f},\\,{ghi:.1f}]',
        'CatRepGap': f'{rgap:+.1f}',
        'CatRepGapCI': f'[{rlo:.1f},\\,{rhi:.1f}]',
        'CatRepModels': str(len(models)),
        'CatRepModelsOk': str(ok),
        'CatRho': f'{rho:+.2f}',
        'CatShared': str(len(shared)),
        'CatMarginLo': f'{min(mgaps):+.1f}',
        'CatMarginHi': f'{max(mgaps):+.1f}',
        'CatSnrTop': slashes(mix(fu, qs[0], 'snr_db'), (-5, 0, 5)),
        'CatSnrBot': slashes(mix(fu, qs[-1], 'snr_db'), (-5, 0, 5)),
        'CatDistTop': slashes(mix(fu, qs[0], 'distractor_count'), (1, 2)),
        'CatDistBot': slashes(mix(fu, qs[-1], 'distractor_count'), (1, 2)),
        'CatOvTop': slashes(mix(fu, qs[0], 'overlap'), ('partial', 'full')),
        'CatOvBot': slashes(mix(fu, qs[-1], 'overlap'), ('partial', 'full')),
    }
    for k, st in enumerate(fr, 1):
        w = ('One', 'Two', 'Three', 'Four', 'Five')[k - 1]
        mac[f'CatQW{w}'] = f"{st['rf']:.1f}\\%"
        mac[f'CatQA{w}'] = f"{st['acc']:.1f}\\%"
        mac[f'CatQN{w}'] = str(st['correct'])
        mac[f'CatQC{w}'] = str(len(qs[k - 1]))
        mac[f'CatQR{w}'] = f"{rr[k - 1]['rf']:.1f}\\%"
    write_macros(GEN / 'category_structure.tex', mac, 'Generated by scripts/build_category_structure.py')

    lines = ['% Generated by scripts/build_category_structure.py',
             '\\begin{tabular}{lrrrrr}', '\\toprule',
             'Event quintile & Acc. & Correct & RFail & Re-paired \\\\', '\\midrule']
    for k, (q, st, rt) in enumerate(zip(qs, fr, rr), 1):
        tag = 'highest' if k == 1 else 'lowest' if k == N_QUINT else ''
        name = f'Q{k}' + (f' ({tag})' if tag else '')
        lines.append(f'{name} & {st["acc"]:.1f} & {st["correct"]} & '
                     f'{st["rf"]:.1f} & {rt["rf"]:.1f} \\\\')
    lines += ['\\bottomrule', '\\end{tabular}']
    (GEN / 'category_table.tex').write_text('\n'.join(lines) + '\n')
    (GEN / 'category_structure.json').write_text(json.dumps(
        {'categories': len(rows), 'quintiles': qs,
         'frozen': fr, 'repair': rr,
         'gap': gap, 'gap_ci': [glo, ghi], 'repair_gap': rgap, 'repair_gap_ci': [rlo, rhi],
         'spearman': rho, 'perm_p': pval, 'shared': len(shared),
         'repair_models': per_model, 'margin_bins': {str(b): g for b, g in zip(bins, mgaps)}}, indent=1) + '\n')

    print(f"{'Q':>2s} {'cats':>4s} {'correct':>7s} {'acc':>6s} {'frozen':>7s} {'repaired':>8s}")
    for k, (st, rt) in enumerate(zip(fr, rr), 1):
        print(f"Q{k} {len(qs[k - 1]):4d} {st['correct']:7d} {st['acc']:5.1f}% "
              f"{st['rf']:6.1f}% {rt['rf']:7.1f}%")
    print(f"gap Q1-Q5 frozen {gap:+.1f} CI [{glo:.1f},{ghi:.1f}]  "
          f"re-paired {rgap:+.1f} CI [{rlo:.1f},{rhi:.1f}]  ({ok}/{len(models)} models)")
    print(f"Spearman {rho:+.3f} over {len(shared)} shared categories (perm p={pval:.2e})")
    print(f"gap by mixture margin: {[round(g, 1) for g in mgaps]}")
    print(f"highest {label(rows[0])}  lowest {label(rows[-1])}")
    print('wrote generated/category_structure.tex')


if __name__ == '__main__':
    main()
