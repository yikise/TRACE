"""Consolidated H-based statistics, robustness, and equation-integrity checks.

Input : generated/ten_model_panel.json, generated/position_bias.json
Output: generated/robust_numbers.tex, generated/robust_results.json

Reports
  * the distribution of the ordering gap H among correct answers, including how
    close to zero it sits (the failure criterion is a strict inequality, so the
    sensitivity of the rate to a near-zero band is a fair question)
  * the pooled ranking-failure rate over all ten models and over the seven whose
    answer-key asymmetry is below the degeneracy threshold
  * the number of correct answers that satisfy more than one condition of the
    equation as printed in the paper (the conditions overlap; the implemented
    rule is disjoint). This is the arithmetic behind the equation rewrite.
"""
import json
from collections import defaultdict
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from texnames import write_macros

ROOT = Path(__file__).resolve().parents[1]
GEN = ROOT / 'generated'
SEED, DRAWS = 20260907, 10000


def parts(u):
    dt = u['mf'] - u['mt']
    H = dt - max(0.0, max(u['mf'] - x for x in u['mj']))
    G = dt - max([abs(u['mf'] - x) for x in u['mj']])
    return dt, H, G


def code_state(u):
    dt, H, G = parts(u)
    if dt <= 0:
        return 'ungrounded'
    if H <= 0:
        return 'confused'
    if G <= 0:
        return 'context'
    return 'grounded'


def printed_conditions(u):
    """The conditions as literally printed in the paper's Eq. (3)."""
    dt, H, G = parts(u)
    out = []
    if dt <= 0:
        out.append('ungrounded')
    if G <= 0:
        out.append('context')
    return out


def main():
    panel = json.loads((GEN / 'ten_model_panel.json').read_text())
    units = [u for u in panel['panels']['baseline'] if u['mt'] is not None and u['mj']]
    cor = [u for u in units if u['mf'] > 0]
    pb = json.loads((GEN / 'position_bias.json').read_text())
    degen = {r['model'] for r in pb['per_model'] if r['degenerate']}
    models = panel['models']
    clean = [m for m in models if m not in degen]

    dt = np.array([parts(u)[0] for u in cor])
    H = np.array([parts(u)[1] for u in cor])
    G = np.array([parts(u)[2] for u in cor])

    res = {'n_correct': len(cor), 'n_units': len(units), 'seed': SEED, 'draws': DRAWS}
    res['H'] = {
        'positive_share': float(np.mean(H > 0)),
        'nonpositive_share': float(np.mean(H <= 0)),
        'exact_zero': int(np.sum(H == 0)),
        'exact_zero_share': float(np.mean(H == 0)),
        'abs_lt_0.05': float(np.mean(np.abs(H) < 0.05)),
        'abs_lt_0.10': float(np.mean(np.abs(H) < 0.10)),
        'abs_lt_0.25': float(np.mean(np.abs(H) < 0.25)),
        'median_positive': float(np.median(H[H > 0])),
        'median_nonpositive': float(np.median(H[H <= 0])),
        'confused_share': float(np.mean((H <= 0) & (dt > 0))),
        'ungrounded_share': float(np.mean((H <= 0) & (dt <= 0))),
    }
    # sensitivity to a near-zero dead band
    res['band'] = {str(e): float(np.mean(H <= -e)) for e in [0.0, 1e-9, 0.01, 0.05, 0.1, 0.25, 0.5]}

    # the same two summaries on the eight design models only, so that the
    # invariance paragraph quotes a population consistent with \RankFail
    dsel = set(panel['paper_models'])
    hd = np.array([parts(u)[1] for u in cor if u['model_id'] in dsel])
    res['H_design8'] = {
        'n': int(len(hd)),
        'positive_share': float(np.mean(hd > 0)),
        'nonpositive_share': float(np.mean(hd <= 0)),
        'exact_zero': int(np.sum(hd == 0)),
        'exact_zero_share': float(np.mean(hd == 0)),
        'band': {str(e): float(np.mean(hd <= -e)) for e in [0.05, 0.1, 0.25]},
    }

    # equation overlap
    amb = sum(1 for u in cor if len(printed_conditions(u)) > 1)
    outside = sum(1 for u in cor if printed_conditions(u) and code_state(u) not in printed_conditions(u))
    mism = sum(1 for u in cor if code_state(u) != (
        'ungrounded' if parts(u)[0] <= 0 else
        'confused' if parts(u)[1] <= 0 else
        'context' if parts(u)[2] <= 0 else 'grounded'))
    res['equation'] = {'ambiguous': amb, 'ambiguous_share': amb / len(cor),
                       'outside_printed': outside, 'disjoint_mismatch': mism,
                       'confused_count': int(np.sum((H <= 0) & (dt > 0))),
                       'context_count': int(np.sum((G <= 0) & (H > 0)))}

    # pooled rates with bootstrap over source tuples
    by = defaultdict(list)
    for u in units:
        by[u['source_tuple_id']].append(u)
    keys = sorted(by)

    def pooled(sel, draws=DRAWS):
        rng = np.random.default_rng(SEED)
        per = {}
        for t in keys:
            ss = [code_state(u) for u in by[t] if u['model_id'] in sel and u['mf'] > 0]
            if ss:
                per[t] = ss
        tk = [t for t in keys if t in per]
        allst = [s for t in tk for s in per[t]]
        n = len(allst)
        pt = (allst.count('confused') + allst.count('ungrounded')) / n
        bs = []
        for _ in range(draws):
            pick = rng.integers(0, len(tk), len(tk))
            p = [s for i in pick for s in per[tk[i]]]
            bs.append((p.count('confused') + p.count('ungrounded')) / len(p))
        return {'n': n, 'rankfail': pt, 'lo': float(np.percentile(bs, 2.5)),
                'hi': float(np.percentile(bs, 97.5))}

    a = pooled(set(models))
    c = pooled(set(clean))
    res['pooled_all10'] = a
    res['pooled_clean'] = c
    res['clean_models'] = clean
    res['degenerate_models'] = sorted(degen)

    print(f"correct answers: {len(cor):,}")
    print(f"H>0 {100*res['H']['positive_share']:.1f}%  H<=0 {100*res['H']['nonpositive_share']:.1f}%  "
          f"H==0 exactly {res['H']['exact_zero']:,} ({100*res['H']['exact_zero_share']:.2f}%)")
    print(f"|H|<0.05 {100*res['H']['abs_lt_0.05']:.2f}%  |H|<0.25 {100*res['H']['abs_lt_0.25']:.2f}%")
    print(f"confused {100*res['H']['confused_share']:.2f}%  ungrounded {100*res['H']['ungrounded_share']:.2f}%")
    print('band sensitivity:', {k: f'{100*v:.2f}%' for k, v in res['band'].items()})
    print(f"\nequation: {amb:,} answers ({100*res['equation']['ambiguous_share']:.2f}%) match more than one printed condition; "
          f"{outside:,} have an implemented state outside the printed list; disjoint mismatch {mism}")
    print(f"\npooled all 10 : n={a['n']:,} rankfail={100*a['rankfail']:.1f}% [{100*a['lo']:.1f},{100*a['hi']:.1f}]")
    print(f"pooled clean  : n={c['n']:,} rankfail={100*c['rankfail']:.1f}% [{100*c['lo']:.1f},{100*c['hi']:.1f}]")
    print(f"degenerate    : {sorted(degen)}")

    (GEN / 'robust_results.json').write_text(json.dumps(res, indent=1) + '\n')
    mac = {
        'HCorrectCount': f"{len(cor):,}",
        'HPositive': f"{100*res['H']['positive_share']:.1f}\\%",
        'HNonPositive': f"{100*res['H']['nonpositive_share']:.1f}\\%",
        'HExactZero': f"{res['H']['exact_zero']:,}",
        'HExactZeroShare': f"{100*res['H']['exact_zero_share']:.2f}\\%",
        'HDesignNonPositive': f"{100*res['H_design8']['nonpositive_share']:.1f}\\%",
        'HDesignExactZero': f"{res['H_design8']['exact_zero']:,}",
        'HDesignExactZeroShare': f"{100*res['H_design8']['exact_zero_share']:.2f}\\%",
        'HDesignBandFiveHundredths': f"{100*res['H_design8']['band']['0.05']:.1f}\\%",
        'HDesignBandQuarter': f"{100*res['H_design8']['band']['0.25']:.1f}\\%",
        'HAbsLtFiveHundredths': f"{100*res['H']['abs_lt_0.05']:.2f}\\%",
        'HAbsLtOneTenth': f"{100*res['H']['abs_lt_0.10']:.2f}\\%",
        'HAbsLtQuarter': f"{100*res['H']['abs_lt_0.25']:.2f}\\%",
        'HMedianPositive': f"{res['H']['median_positive']:.2f}",
        'HMedianNonPositive': f"{res['H']['median_nonpositive']:.2f}",
        'HConfusedShare': f"{100*res['H']['confused_share']:.2f}\\%",
        'HUngroundedShare': f"{100*res['H']['ungrounded_share']:.2f}\\%",
        'EqAmbiguousCount': f"{amb:,}",
        'EqAmbiguousShare': f"{100*res['equation']['ambiguous_share']:.2f}\\%",
        'EqOutsidePrinted': f"{outside:,}",
        'EqConfusedCount': f"{res['equation']['confused_count']:,}",
        'EqContextCount': f"{res['equation']['context_count']:,}",
        'RobustAllN': f"{a['n']:,}",
        'RobustAllRankFail': f"{100*a['rankfail']:.1f}\\%",
        'RobustAllCI': f"{100*a['lo']:.1f}--{100*a['hi']:.1f}\\%",
        'RobustCleanN': f"{c['n']:,}",
        'RobustCleanRankFail': f"{100*c['rankfail']:.1f}\\%",
        'RobustCleanCI': f"{100*c['lo']:.1f}--{100*c['hi']:.1f}\\%",
    }
    BAND = {'0.05': 'FiveHundredths', '0.1': 'OneTenth', '0.25': 'Quarter'}
    for e, word in BAND.items():
        mac[f'HBand{word}'] = f"{100*res['band'][e]:.1f}\\%"
    write_macros(GEN / 'robust_numbers.tex', mac,
                 'Generated by scripts/build_robust_numbers.py')
    print(f'\nwrote generated/robust_numbers.tex ({len(mac)} macros)')


if __name__ == '__main__':
    main()
