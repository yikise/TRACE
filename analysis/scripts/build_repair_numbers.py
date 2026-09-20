"""Re-paired-background replication analysis for the TRACE paper.

Input : generated/repair_inputs.json (three models, 2880 repair conditions)
        generated/counterfactual_inputs.json (eight-model frozen panel)
Output: generated/repair_numbers.tex, generated/repair_results.json

Question: holding the target clip, target gain, SNR, and full overlap fixed,
does the ranking-failure rate reproduce when the two distractors are redrawn
blind to model outcomes? And is a role's failure stable across backgrounds?
"""
import collections
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
GEN = ROOT / 'generated'
SEED, DRAWS = 20260907, 10000
NAMES = {'qwen25_omni_7b': 'Qwen2.5-Omni-7B',
         'qwen3_omni_30b': 'Qwen3-Omni-30B',
         'step_audio2_mini': 'Step-Audio 2 Mini'}


def prepare(u):
    u = dict(u)
    mf, mt, mj = u['mf'], u['mt'], u['mj']
    dt = mf - mt
    dj = [mf - m for m in mj]
    H = dt - max(0.0, max(dj))
    u.update(dt=dt, H=H, correct=mf > 0, rankfail=mf > 0 and H <= 0)
    return u


def main():
    rep = json.loads((GEN / 'repair_inputs.json').read_text())
    base = json.loads((GEN / 'counterfactual_inputs.json').read_text())
    units = [prepare(u) for u in rep['units']]
    models = rep['models']
    # cluster = the underlying original target role (192 of them)
    clusters = sorted({u['role_index'] for u in units})
    assert len(clusters) == 192, len(clusters)
    ix = np.random.default_rng(SEED).integers(0, len(clusters), (DRAWS, len(clusters)))
    cidx = {c: i for i, c in enumerate(clusters)}

    def ratio(us, num, den=lambda u: u['correct']):
        n = np.zeros(len(clusters)); d = np.zeros(len(clusters))
        for u in us:
            i = cidx[u['role_index']]
            if den(u):
                d[i] += 1
                if num(u):
                    n[i] += 1
        point = n.sum() / d.sum() if d.sum() else float('nan')
        s = n[ix].sum(1) / np.maximum(d[ix].sum(1), 1)
        lo, hi = np.percentile(s, [2.5, 97.5])
        return dict(count=int(n.sum()), denom=int(d.sum()), rate=float(point),
                    ci95=[float(lo), float(hi)])

    res = {'models': models, 'n_units': len(units),
           'n_roles': len(clusters),
           'bootstrap': {'cluster': 'original target role', 'draws': DRAWS, 'seed': SEED},
           'input_sha256': __import__('hashlib').sha256((GEN / 'repair_inputs.json').read_bytes()).hexdigest()}

    pooled = ratio(units, lambda u: u['rankfail'])
    res['pooled_rankfail'] = pooled
    res['per_model'] = {m: ratio([u for u in units if u['model_id'] == m],
                                 lambda u: u['rankfail']) for m in models}
    res['per_snr'] = {str(s): ratio([u for u in units if u['snr_db'] == s],
                                    lambda u: u['rankfail']) for s in [5.0, 0.0, -5.0]}

    # How many of the five backgrounds for a role fail at least once / always?
    byrole = collections.defaultdict(list)
    for u in units:
        byrole[(u['model_id'], u['role_index'], u['snr_db'])].append(u)
    assert all(len(v) == 5 for v in byrole.values()), \
        collections.Counter(len(v) for v in byrole.values())
    # Restrict to cells where the mixture answer stays correct on all five
    # backgrounds: within those, failure cannot be blamed on one recording
    # pairing, only on the acoustics drawn.
    stable = [v for v in byrole.values() if all(u['correct'] for u in v)]
    ever = sum(1 for v in stable if any(u['rankfail'] for u in v))
    always = sum(1 for v in stable if all(u['rankfail'] for u in v))
    res['role_stability'] = {
        'role_snr_cells': len(byrole), 'correct_on_all_five': len(stable),
        'fail_on_at_least_one': ever, 'fail_on_all_five': always,
        'share_fail_at_least_one_of_correct_cells': (ever / len(stable) if stable else None),
    }

    # Same-role comparison against the frozen panel's matching condition
    # (full overlap, two distractors, same SNR, same model, same role).
    frozen = [prepare(u) for u in base['panels']['baseline']]
    # role lookup shared by both panels, keyed by the frozen (tuple, target)
    role_of = {(u['original_source_tuple_id'], u['original_target_index']): u['role_index']
               for u in units}
    assert len(role_of) == 192, len(role_of)
    fkey = {}
    for u in frozen:
        u['role_index'] = role_of.get((u['source_tuple_id'], u['target_index']))
        if u['overlap'] == 'full' and u['distractor_count'] == 2 and u['role_index'] is not None:
            fkey[(u['model_id'], u['source_tuple_id'], u['target_index'], u['snr_db'])] = u
    pairs = []
    for u in units:
        k = (u['model_id'], u['original_source_tuple_id'],
             u['original_target_index'], u['snr_db'])
        f = fkey.get(k)
        if f is None:
            continue
        pairs.append((f, u))
    # matched comparison on cells where BOTH are correct
    both = [(f, r) for f, r in pairs if f['correct'] and r['correct']]
    res['matched_frozen_pairs'] = {'n_conditions': len(pairs), 'n_both_correct': len(both),
                                   'frozen_rankfail': ratio([f for f, _ in both],
                                                            lambda u: u['rankfail']),
                                   'repair_rankfail': ratio([r for _, r in both],
                                                            lambda u: u['rankfail'])}

    # Paired difference on the same shared answers (repaired minus frozen).
    # Two separately bootstrapped marginals can coincide while individual cells
    # move in both directions, so the stability claim is reported as a paired
    # contrast with its own interval, clustered on the same roles.
    def pair_diff(pairs_):
        fn = np.zeros(len(clusters)); fd = np.zeros(len(clusters))
        rn = np.zeros(len(clusters)); rd = np.zeros(len(clusters))
        for f, r in pairs_:
            i = cidx[f['role_index']]
            fd[i] += 1; fn[i] += f['rankfail']
            rd[i] += 1; rn[i] += r['rankfail']
        pt = rn.sum() / max(rd.sum(), 1) - fn.sum() / max(fd.sum(), 1)
        s = (rn[ix].sum(1) / np.maximum(rd[ix].sum(1), 1)
             - fn[ix].sum(1) / np.maximum(fd[ix].sum(1), 1))
        lo, hi = np.percentile(s, [2.5, 97.5])
        return dict(point=float(pt), ci95=[float(lo), float(hi)],
                    frozen=float(fn.sum() / max(fd.sum(), 1)),
                    repair=float(rn.sum() / max(rd.sum(), 1)))
    res['matched_paired_diff'] = pair_diff(both)
    # does the frozen outcome predict the re-paired outcome?
    tab = collections.Counter((f['rankfail'], r['rankfail']) for f, r in both)
    res['matched_transition'] = {f'{int(a)}->{int(b)}': c for (a, b), c in sorted(tab.items())}

    # macros
    def pct(name, r):
        mac[name] = f"{100*r['rate']:.1f}\\%"
        mac[name + 'CI'] = '--'.join(f'{100*x:.1f}' for x in r['ci95']) + '\\%'
        mac[name + 'Count'] = f"{r['count']:,}"
        mac[name + 'Denom'] = f"{r['denom']:,}"

    mac = {}
    pct('ReplRankFail', pooled)
    mac['ReplModelCount'] = f'{len(models)}'
    mac['ReplRoleCount'] = f'{len(clusters):,}'
    mac['ReplUnitCount'] = f'{len(units):,}'
    # Display names are emitted alongside the numbers so the prose can name the
    # models it reports. The number macros carry an explicit `RankFail` suffix:
    # the earlier `\Repl<Model>` spelling expanded to a percentage, so a citation
    # like "(\ReplQwenTwoFiveSevenB, ...)" silently printed "(9.2%, ...)" and the
    # model identities never reached the page.
    REPAIR_DISPLAY = {
        'qwen25_omni_7b': ('QwenTwoFiveSevenB', 'Qwen2.5-Omni-7B'),
        'qwen3_omni_30b': ('QwenThreeOmni', 'Qwen3-Omni-30B'),
        'step_audio2_mini': ('StepAudioTwoMini', 'Step-Audio~2~Mini'),
    }
    for m in models:
        tag, display = REPAIR_DISPLAY[m]
        pct('Repl' + tag + 'RankFail', res['per_model'][m])
        mac['Repl' + tag + 'Name'] = display
    for s, tag in [(5.0, 'High'), (0.0, 'Mid'), (-5.0, 'Low')]:
        pct('ReplSnr' + tag, res['per_snr'][str(s)])
    pct('ReplFrozenMatched', res['matched_frozen_pairs']['frozen_rankfail'])
    pct('ReplRepairMatched', res['matched_frozen_pairs']['repair_rankfail'])
    _pd = res['matched_paired_diff']
    mac['ReplMatchedDelta'] = f"{100*_pd['point']:+.1f}"
    mac['ReplMatchedDeltaCI'] = '[' + ',\\, '.join(
        f'{100*x:+.1f}' for x in _pd['ci95']) + ']'
    # Paired transition on the shared correct answers. Reported so the
    # stability claim rests on the cell-level agreement pattern rather than
    # on two marginal rates, which can coincide while many cells switch.
    trans = res['matched_transition']
    n_both = res['matched_frozen_pairs']['n_both_correct']
    n_down = trans.get('1->0', 0)   # failed on frozen, ordered after re-pairing
    n_up = trans.get('0->1', 0)     # ordered on frozen, failed after re-pairing
    mac['ReplTransFailToOk'] = f'{n_down:,}'
    mac['ReplTransOkToFail'] = f'{n_up:,}'
    mac['ReplTransStableFail'] = f"{trans.get('1->1', 0):,}"
    mac['ReplTransSwitchShare'] = f'{100 * (n_down + n_up) / n_both:.1f}\\%'
    rs = res['role_stability']
    mac['ReplCorrectAllFive'] = f"{rs['correct_on_all_five']:,}"
    mac['ReplFailEverShare'] = f"{100*rs['share_fail_at_least_one_of_correct_cells']:.1f}\\%"
    mac['ReplFailAlways'] = f"{rs['fail_on_all_five']:,}"

    (GEN / 'repair_numbers.tex').write_text(
        '% Generated by scripts/build_repair_numbers.py\n' +
        '\n'.join(f'\\newcommand{{\\{k}}}{{{v}}}' for k, v in mac.items()) + '\n')
    (GEN / 'repair_results.json').write_text(json.dumps(res, indent=1) + '\n')
    print(json.dumps(res, indent=1))
    for k, v in mac.items():
        print(f'{k:28s} {v}')


if __name__ == '__main__':
    main()
