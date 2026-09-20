#!/usr/bin/env python3
"""Does the category gap survive replacing the target recordings themselves?

Sec. 5.2 shows ranking failures cluster on specific source events, and that the
clustering transfers to a re-paired panel -- but both panels hold the target
recording fixed, so a category effect and a "these few clips happen to be hard"
effect stay confounded. This panel separates them. Every frozen target role in
the hardest (Q1) and easiest (Q5) quintiles is re-scored with a *different*
audited recording of the same category substituted into the identical mixture:
backgrounds, SNR, overlap, distractor pairing, query and label mapping are
byte-identical, and only the target waveform changes.

Inputs are the sufficient statistics in generated/subst_aggregates.json, produced
on the analysis host by the panel's aggregate_subst.py. Every statistic here is a
ratio of pooled counts and the only resampling unit is the source tuple, so the
per-(model, tuple, quintile) and per-role counters reconstruct each rate exactly;
that script asserts the reduction reproduces the direct counts.

Writes generated/subst_numbers.tex and generated/subst_results.json.
"""
import collections
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
GEN = ROOT / 'generated'
SEED, DRAWS = 20260907, 10000
DOC = json.loads((GEN / 'subst_aggregates.json').read_text())
CELLS = DOC['cells']
ROLES = DOC['roles']

# Denominator/numenator key suffixes: ITT counts every unit, 'both' restricts to
# units both the frozen and the substituted target answer correctly, and the
# 'b_' prefix selects the frozen (baseline) target rather than its substitute.
ITT = ('itt_den', 'itt_num')
BOTH = ('both_den', 'both_num')


def rate(rows, keys):
    den = sum(r[keys[0]] for r in rows)
    num = sum(r[keys[1]] for r in rows)
    return (100.0 * num / den if den else float('nan')), den, num


def pick(q=None, both=False, base=False):
    dk, nk = BOTH if both else ITT
    if base:
        dk, nk = 'b_' + dk, 'b_' + nk
    rows = [c for c in CELLS if q is None or c['quintile'] == q]
    return rate(rows, (dk, nk))


def main():
    # Headline rates. The ITT denominator moves between panels because swapping
    # the target changes which mixtures the model answers correctly.
    r1f, d1f, _ = pick('Q1', base=True)
    r5f, d5f, _ = pick('Q5', base=True)
    r1s, d1s, _ = pick('Q1')
    r5s, d5s, _ = pick('Q5')
    gap_f, gap_s = r1f - r5f, r1s - r5s

    r1fb, d1fb, _ = pick('Q1', both=True, base=True)
    r5fb, d5fb, _ = pick('Q5', both=True, base=True)
    r1sb, _, _ = pick('Q1', both=True)
    r5sb, _, _ = pick('Q5', both=True)

    # Cluster bootstrap over source tuples. A tuple's cells are resampled
    # together, and every rate is recomputed inside the draw.
    tuples = sorted({c['tuple'] for c in CELLS})
    by_tuple = collections.defaultdict(list)
    for c in CELLS:
        by_tuple[c['tuple']].append(c)
    ix = np.random.default_rng(SEED).integers(0, len(tuples), (DRAWS, len(tuples)))

    def draw(rows, q, keys):
        den = num = 0
        for c in rows:
            if c['quintile'] == q:
                den += c[keys[0]]
                num += c[keys[1]]
        return (100.0 * num / den if den else np.nan)

    gf, gs, c1, c5 = (np.empty(DRAWS) for _ in range(4))
    gfb, gsb = np.empty(DRAWS), np.empty(DRAWS)
    for k in range(DRAWS):
        rows = [c for t in (tuples[i] for i in ix[k]) for c in by_tuple[t]]
        gf[k] = draw(rows, 'Q1', ('b_itt_den', 'b_itt_num')) - draw(rows, 'Q5', ('b_itt_den', 'b_itt_num'))
        gs[k] = draw(rows, 'Q1', ITT) - draw(rows, 'Q5', ITT)
        c1[k] = draw(rows, 'Q1', ITT) - draw(rows, 'Q1', ('b_itt_den', 'b_itt_num'))
        c5[k] = draw(rows, 'Q5', ITT) - draw(rows, 'Q5', ('b_itt_den', 'b_itt_num'))
        gfb[k] = draw(rows, 'Q1', ('b_both_den', 'b_both_num')) - draw(rows, 'Q5', ('b_both_den', 'b_both_num'))
        gsb[k] = draw(rows, 'Q1', BOTH) - draw(rows, 'Q5', BOTH)

    def ci(v):
        lo, hi = np.percentile(v, [2.5, 97.5])
        return float(lo), float(hi)

    fgl, fgh = ci(gf)
    sgl, sgh = ci(gs)
    c1l, c1h = ci(c1)
    c5l, c5h = ci(c5)
    fbl, fbh = ci(gfb)
    sbl, sbh = ci(gsb)

    # Difference-in-differences on the paired draws: does substituting targets
    # narrow the gap at all, and does the substituted gap stay clear of zero?
    did = gs - gf
    dl, dh = ci(did)
    frac_narrow = float(np.mean(did < 0))
    frac_above10 = float(np.mean(gs > 10))
    frac_positive = float(np.mean(gs > 0))

    # Per-model ordering after substitution.
    models = sorted({c['model'] for c in CELLS})
    per_model = {}
    for m in models:
        rows = [c for c in CELLS if c['model'] == m]
        per_model[m] = {
            'q1_base': rate([c for c in rows if c['quintile'] == 'Q1'], ('b_itt_den', 'b_itt_num'))[0],
            'q5_base': rate([c for c in rows if c['quintile'] == 'Q5'], ('b_itt_den', 'b_itt_num'))[0],
            'q1_subst': rate([c for c in rows if c['quintile'] == 'Q1'], ITT)[0],
            'q5_subst': rate([c for c in rows if c['quintile'] == 'Q5'], ITT)[0],
        }
    ordered = sum(1 for v in per_model.values() if v['q1_subst'] > v['q5_subst'])
    # A model whose own Q1 failure rate rises when targets are replaced.
    rose = [m for m, v in per_model.items() if v['q1_subst'] > v['q1_base']]

    # Role-level transfer across the 56 roles with enough correct answers on both.
    rl = [(r['b_itt_num'] / r['b_itt_den'], r['itt_num'] / r['itt_den']) for r in ROLES
          if r['b_itt_den'] >= 5 and r['itt_den'] >= 5]
    pear = (float(np.corrcoef([x[0] for x in rl], [x[1] for x in rl])[0, 1])
            if len(rl) >= 3 else float('nan'))
    # Roles whose failure rate moves by more than 5pp in either direction.
    moved_up = sum(1 for a, b in rl if b > a + 0.05)
    moved_dn = sum(1 for a, b in rl if b < a - 0.05)

    print(f'Q1 frozen {r1f:.1f}% (n={d1f}) -> subst {r1s:.1f}% (n={d1s})  '
          f'change {r1s - r1f:+.1f}pp CI [{c1l:+.1f},{c1h:+.1f}]')
    print(f'Q5 frozen {r5f:.1f}% (n={d5f}) -> subst {r5s:.1f}% (n={d5s})  '
          f'change {r5s - r5f:+.1f}pp CI [{c5l:+.1f},{c5h:+.1f}]')
    print(f'gap frozen {gap_f:+.1f}pp CI [{fgl:+.1f},{fgh:+.1f}]')
    print(f'gap subst  {gap_s:+.1f}pp CI [{sgl:+.1f},{sgh:+.1f}]')
    print(f'DiD {did.mean():+.1f}pp CI [{dl:+.1f},{dh:+.1f}], narrowed in '
          f'{100 * frac_narrow:.1f}% of draws; substituted gap >10pp in '
          f'{100 * frac_above10:.1f}%, >0 in {100 * frac_positive:.1f}%')
    print(f'[paired both correct] gap frozen {r1fb - r5fb:+.1f}pp CI [{fbl:+.1f},{fbh:+.1f}] '
          f'-> subst {r1sb - r5sb:+.1f}pp CI [{sbl:+.1f},{sbh:+.1f}]')
    print(f'models keeping Q1>Q5 after substitution: {ordered}/{len(models)}; Q1 rose for {rose}')
    print(f'role-level r={pear:+.2f} over {len(rl)} roles; moved >5pp: {moved_up} up, {moved_dn} down')
    for m, v in per_model.items():
        print(f"  {m:20s} frozen {v['q1_base']:5.1f} vs {v['q5_base']:5.1f} | "
              f"subst {v['q1_subst']:5.1f} vs {v['q5_subst']:5.1f}")

    out = {
        'seed': SEED, 'draws': DRAWS, 'n_units': DOC['n_units'],
        'n_conditions': DOC['n_conditions'], 'n_clips': DOC['n_clips'],
        'n_models': len(models), 'n_roles': DOC['n_roles'], 'n_tuples': DOC['n_tuples'],
        'q1_frozen': r1f, 'q5_frozen': r5f, 'q1_subst': r1s, 'q5_subst': r5s,
        'q1_frozen_n': d1f, 'q5_frozen_n': d5f, 'q1_subst_n': d1s, 'q5_subst_n': d5s,
        'gap_frozen': gap_f, 'gap_frozen_ci': [fgl, fgh],
        'gap_subst': gap_s, 'gap_subst_ci': [sgl, sgh],
        'did': float(did.mean()), 'did_ci': [dl, dh],
        'frac_narrowed': frac_narrow, 'frac_gap_above_10': frac_above10,
        'frac_gap_positive': frac_positive,
        'gap_frozen_both': r1fb - r5fb, 'gap_frozen_both_ci': [fbl, fbh],
        'gap_subst_both': r1sb - r5sb, 'gap_subst_both_ci': [sbl, sbh],
        'change_q1': r1s - r1f, 'change_q1_ci': [c1l, c1h],
        'change_q5': r5s - r5f, 'change_q5_ci': [c5l, c5h],
        'models_ordered': ordered, 'models_q1_rose': rose,
        'role_pearson': pear, 'role_n': len(rl),
        'roles_moved_up': moved_up, 'roles_moved_down': moved_dn,
        'per_model': per_model,
    }
    (GEN / 'subst_results.json').write_text(json.dumps(out, indent=1, sort_keys=True) + '\n')

    mac = {
        'SubstUnits': f"{DOC['n_units']:,}",
        'SubstConditions': f"{DOC['n_conditions']:,}",
        'SubstClips': str(DOC['n_clips']),
        'SubstRoles': str(DOC['n_roles']),
        'SubstModels': str(len(models)),
        'SubstModelsOrdered': str(ordered),
        'SubstQOneFrozen': f'{r1f:.1f}\\%',
        'SubstQFiveFrozen': f'{r5f:.1f}\\%',
        'SubstQOneSubst': f'{r1s:.1f}\\%',
        'SubstQFiveSubst': f'{r5s:.1f}\\%',
        # Denominators, because the ITT comparison moves them: the frozen and
        # replaced arms are not scored on the same number of correct answers.
        'SubstQOneFrozenN': f'{d1f:,}',
        'SubstQFiveFrozenN': f'{d5f:,}',
        'SubstQOneSubstN': f'{d1s:,}',
        'SubstQFiveSubstN': f'{d5s:,}',
        'SubstGapFrozen': f'{gap_f:+.1f}',
        'SubstGapFrozenCI': f'[{fgl:+.1f},\\,{fgh:+.1f}]',
        'SubstGapSubst': f'{gap_s:+.1f}',
        'SubstGapSubstCI': f'[{sgl:+.1f},\\,{sgh:+.1f}]',
        'SubstDiD': f'{did.mean():+.1f}',
        'SubstDiDCI': f'[{dl:+.1f},\\,{dh:+.1f}]',
        'SubstDiDNarrowed': f'{100 * frac_narrow:.1f}\\%',
        'SubstGapAboveTen': f'{100 * frac_above10:.1f}\\%',
        # Unsigned: the prose says "falls"/"rises", so a sign would read as a
        # double negative.
        'SubstChangeQOne': f'{abs(r1s - r1f):.1f}',
        'SubstChangeQOneCI': f'[{c1l:+.1f},\\,{c1h:+.1f}]',
        'SubstChangeQFive': f'{abs(r5s - r5f):.1f}',
        'SubstChangeQFiveCI': f'[{c5l:+.1f},\\,{c5h:+.1f}]',
        'SubstGapFrozenBoth': f'{r1fb - r5fb:+.1f}',
        'SubstGapSubstBoth': f'{r1sb - r5sb:+.1f}',
        'SubstGapSubstBothCI': f'[{sbl:+.1f},\\,{sbh:+.1f}]',
        'SubstRoleRho': f'{pear:+.2f}',
        'SubstRoleN': str(len(rl)),
    }
    lines = ['% Generated by scripts/build_subst_numbers.py']
    lines += [f'\\newcommand{{\\{k}}}{{{v}}}' for k, v in mac.items()]
    (GEN / 'subst_numbers.tex').write_text('\n'.join(lines) + '\n')
    print(f'-> {GEN / "subst_numbers.tex"}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
