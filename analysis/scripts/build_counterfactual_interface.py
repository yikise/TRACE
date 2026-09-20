"""Same-audio replication over four wordings and two answer mappings.

Input : generated/counterfactual_interface_inputs.json  fixed-mixture
        query-swap panel (8 models x 64 tuples x 3 target roles x 4 wordings
        x 2 answer mappings, every view rendered from identical audio).
Output: generated/counterfactual_interface_results.json
        generated/counterfactual_interface_numbers.tex
        generated/counterfactual_interface_table.tex

The two mappings cover different target roles in the frozen panel, so their
marginal failure rates are descriptive only and cannot separate mapping bias
from role difficulty. Everything below the marginal block is paired: same
model, same source tuple, same target role, same audio, only the wording or
the answer mapping changes. Those paired statistics are what a robustness
claim about the interface has to rest on.
"""
from pathlib import Path
import collections
import hashlib
import json

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
G = ROOT / 'generated'
p = G / 'counterfactual_interface_inputs.json'
data = json.loads(p.read_text())
us = data['units']
g = collections.defaultdict(list)
for u in us:
    u['correct'] = u['mf'] > 0
    u['H'] = min(u['mf'], *u['mj']) - u['mt']
    u['fail'] = u['H'] <= 0
    g[(u['model_id'], u['source_tuple_id'], u['target_index'])].append(u)
assert len(g) == 1536
for v in g.values():
    assert len(v) == 8 and len({(u['prompt_template'], u['label_mapping']) for u in v}) == 8
    assert len({tuple(u['audio_paths']) for u in v}) == 1
    assert len({tuple(u['audio_hashes']) for u in v if len(u['audio_hashes']) == 4}) <= 1
keys = sorted({u['source_tuple_id'] for u in us})
assert len(keys) == 64
ix = np.random.default_rng(20260907).integers(0, 64, (10000, 64))


def rate(v):
    cs = [u for u in v if u['correct']]
    a = np.array([[sum(u['fail'] for u in cs if u['source_tuple_id'] == k),
                   sum(u['source_tuple_id'] == k for u in cs)] for k in keys], float)
    b = a[ix].sum(1)
    return {'count': int(a[:, 0].sum()), 'correct': len(cs),
            'rate': float(a[:, 0].sum() / len(cs)),
            'ci95': np.quantile(b[:, 0] / b[:, 1], [.025, .975]).tolist()}


templates = ['canonical', 'contain', 'hear', 'occur']
out = {'models': sorted({u['model_id'] for u in us}), 'n_views': len(us) * 4,
       'n_conditions': len(us), 'n_base_units': len(g), 'pooled': rate(us),
       'source_files': data['source_files'],
       'input_sha256': hashlib.sha256(p.read_bytes()).hexdigest(),
       'bootstrap': {'draws': 10000, 'seed': 20260907,
                     'cluster': 'source_tuple_id; all models, mappings, and wordings retained'},
       'templates': {t: rate([u for u in us if u['prompt_template'] == t]) for t in templates},
       'mappings': {m: rate([u for u in us if u['label_mapping'] == m]) for m in ['yes_A', 'yes_B']}}
common = [u for v in g.values() if all(u['correct'] for u in v) for u in v]
out['all_interfaces_correct'] = {
    'n_base_units': len(common) // 8, 'pooled': rate(common),
    'templates': {t: rate([u for u in common if u['prompt_template'] == t]) for t in templates}}


# ---------------- paired analysis ----------------
def rf(u):
    """Ranking-failure verdict: correct answer, yet no shared threshold."""
    return bool(u['correct'] and u['fail'])


def pos(u):
    """H > 0: the views are locally separable."""
    return bool(u['H'] > 0)


bykey = {(u['model_id'], u['source_tuple_id'], u['target_index'],
          u['prompt_template'], u['label_mapping']): u for u in us}
assert len(bykey) == len(us)
pairs = []
for k, A in bykey.items():
    if k[4] != 'yes_A':
        continue
    B = bykey.get(k[:4] + ('yes_B',))
    if B is None:
        continue
    assert A['audio_paths'] == B['audio_paths']
    assert not (A['audio_hashes'] and B['audio_hashes']) or A['audio_hashes'] == B['audio_hashes']
    pairs.append((k, A, B))
assert len(pairs) == 6144
ki = {k: i for i, k in enumerate(keys)}


def paired(items, fn):
    num = np.zeros(len(keys))
    den = np.zeros(len(keys))
    for k, A, B in items:
        i = ki[k[1]]
        num[i] += fn(A, B)
        den[i] += 1
    n = num[ix].sum(1)
    d = den[ix].sum(1)
    ok = d > 0
    return (float(num.sum() / den.sum()),
            np.quantile(n[ok] / d[ok], [.025, .975]).tolist(),
            int(num.sum()), int(den.sum()))


both = [(k, A, B) for k, A, B in pairs if A['correct'] and B['correct']]
sign_pt, sign_ci, _, _ = paired(pairs, lambda A, B: float(pos(A) == pos(B)))
agree_pt, agree_ci, _, _ = paired(pairs, lambda A, B: float(rf(A) == rf(B)))
diff_pt, diff_ci, _, _ = paired(pairs, lambda A, B: float(rf(B)) - float(rf(A)))
b_agree_pt, b_agree_ci, _, _ = paired(both, lambda A, B: float(rf(A) == rf(B)))
b_diff_pt, b_diff_ci, b_den, _ = paired(both, lambda A, B: float(rf(B)) - float(rf(A)))
ha = np.array([A['H'] for _, A, B in pairs])
hb = np.array([B['H'] for _, A, B in pairs])
pear = float(np.corrcoef(ha, hb)[0, 1])
rank = lambda x: np.argsort(np.argsort(x))
spear = float(np.corrcoef(rank(ha), rank(hb))[0, 1])


def cohen(a, b):
    """Agreement corrected for the rate each mapping attains on its own."""
    a = np.asarray(a, bool)
    b = np.asarray(b, bool)
    po = float(np.mean(a == b))
    pe = float(np.mean(a) * np.mean(b) + (1 - np.mean(a)) * (1 - np.mean(b)))
    return po, pe, (po - pe) / (1 - pe), float(np.mean(a)), float(np.mean(b))


# Two verdicts on the same pairs. The sign of H asks whether the views are
# separable at all; the rankfail verdict asks whether a correct answer is
# actually supported by the ordering. They have very different base rates, so
# raw agreement on one is not comparable to raw agreement on the other.
ka = cohen([pos(A) for _, A, B in pairs], [pos(B) for _, A, B in pairs])
kb = cohen([rf(A) for _, A, B in pairs], [rf(B) for _, A, B in pairs])
kappa = ka[2]
kappa_rf = kb[2]
kappa_rf_agree = kb[0]
pos_a, pos_b = ka[3], ka[4]
rf_a, rf_b = kb[3], kb[4]

# wording agreement: same role, same mapping, four semantically equivalent wordings
wgroups = collections.defaultdict(list)
for u in us:
    wgroups[(u['model_id'], u['source_tuple_id'], u['target_index'], u['label_mapping'])].append(u)
assert len(wgroups) == 3072
assert all(len(v) == 4 for v in wgroups.values())
unan = float(np.mean([len({rf(u) for u in v}) == 1 for v in wgroups.values()]))
wp = [(A, B) for v in wgroups.values() for i, A in enumerate(v) for B in v[i + 1:]]
wp_agree = float(np.mean([rf(A) == rf(B) for A, B in wp]))
ct = [out['all_interfaces_correct']['templates'][t]['rate'] for t in templates]
wspread = (max(ct) - min(ct)) * 100

out['paired'] = {
    'n_pairs': len(pairs), 'n_both_correct': len(both),
    'sign_agree': sign_pt, 'sign_ci': sign_ci,
    'verdict_agree': agree_pt, 'verdict_ci': agree_ci,
    'verdict_diff_pp': diff_pt * 100, 'verdict_diff_ci_pp': [v * 100 for v in diff_ci],    'verdict_agree_both': b_agree_pt, 'verdict_ci_both': b_agree_ci,
    'diff_both_pp': b_diff_pt * 100, 'diff_both_ci_pp': [v * 100 for v in b_diff_ci],
    'rfail_a_both': float(np.mean([rf(A) for _, A, B in both])) * 100,
    'rfail_b_both': float(np.mean([rf(B) for _, A, B in both])) * 100,
    'pearson_h': pear, 'spearman_h': spear,
    'kappa': kappa, 'kappa_rf': kappa_rf, 'kappa_rf_agree': kappa_rf_agree,
    'pos_a': pos_a, 'pos_b': pos_b, 'rf_a': rf_a, 'rf_b': rf_b,
    'wording_unanimous': unan, 'wording_pair_agree': wp_agree,
    'wording_common_spread_pp': wspread,
    'wording_common_rates': {t: out['all_interfaces_correct']['templates'][t]['rate'] * 100 for t in templates},
}
(G / 'counterfactual_interface_results.json').write_text(json.dumps(out, indent=2) + '\n')

mac = {}


def m(name, r):
    mac[name] = f"{100 * r['rate']:.1f}\\%"
    mac[name + 'CI'] = '--'.join(f'{100 * v:.1f}' for v in r['ci95']) + '\\%'


m('InterfaceFailure', out['pooled'])
m('InterfaceCommonFailure', out['all_interfaces_correct']['pooled'])
mac['InterfaceCommonCount'] = f"{out['all_interfaces_correct']['n_base_units']:,}"
mac['InterfaceMin'] = f"{100 * min(v['rate'] for v in out['templates'].values()):.1f}\\%"
mac['InterfaceMax'] = f"{100 * max(v['rate'] for v in out['templates'].values()):.1f}\\%"
for n, k in [('InterfaceYesA', 'yes_A'), ('InterfaceYesB', 'yes_B')]:
    m(n, out['mappings'][k])
P = out['paired']
mac['InterfacePairN'] = f"{P['n_pairs']:,}"
mac['InterfaceBothN'] = f"{P['n_both_correct']:,}"
mac['InterfaceSignAgree'] = f"{100 * P['sign_agree']:.1f}\\%"
mac['InterfaceSignAgreeCI'] = '--'.join(f'{100 * v:.1f}' for v in P['sign_ci']) + '\\%'
mac['InterfaceVerdictAgree'] = f"{100 * P['verdict_agree']:.1f}\\%"
mac['InterfaceVerdictAgreeBoth'] = f"{100 * P['verdict_agree_both']:.1f}\\%"
mac['InterfaceKappa'] = f"{P['kappa']:.2f}"
mac['InterfaceKappaRf'] = f"{P['kappa_rf']:.2f}"
mac['InterfaceRfAgree'] = f"{100 * P['kappa_rf_agree']:.1f}\\%"
mac['InterfaceSignA'] = f"{100 * P['pos_a']:.1f}\\%"
mac['InterfaceSignB'] = f"{100 * P['pos_b']:.1f}\\%"
mac['InterfaceRfA'] = f"{100 * P['rf_a']:.1f}\\%"
mac['InterfaceRfB'] = f"{100 * P['rf_b']:.1f}\\%"
mac['InterfaceAllDiff'] = f"{P['verdict_diff_pp']:+.1f}"
mac['InterfaceAllDiffCI'] = '{:+.1f}, {:+.1f}'.format(*P['verdict_diff_ci_pp'])
mac['InterfacePearson'] = f"{P['pearson_h']:+.2f}"
mac['InterfaceSpearman'] = f"{P['spearman_h']:+.2f}"
mac['InterfaceBothA'] = f"{P['rfail_a_both']:.1f}\\%"
mac['InterfaceBothB'] = f"{P['rfail_b_both']:.1f}\\%"
mac['InterfaceBothDiff'] = f"{P['diff_both_pp']:+.1f}"
mac['InterfaceBothDiffCI'] = '{:+.1f}, {:+.1f}'.format(*P['diff_both_ci_pp'])
mac['InterfaceWordingUnanimous'] = f"{100 * P['wording_unanimous']:.1f}\\%"
mac['InterfaceWordingSpread'] = f"{P['wording_common_spread_pp']:.1f}"
mac['InterfaceWordingCount'] = str(len(templates))
# Per-wording common-correct failure rates, so the spread can be read against
# the four values it is computed from rather than quoted alone.
for _t, _r in P['wording_common_rates'].items():
    mac['InterfaceWording' + _t.capitalize()] = f"{_r:.1f}\\%"
(G / 'counterfactual_interface_numbers.tex').write_text(
    '% Generated by build_counterfactual_interface.py\n' +
    '\n'.join('\\newcommand{\\' + k + '}{' + v + '}' for k, v in mac.items()) + '\n')

rows = ['\\begin{tabular}{lrr}', '\\toprule',
        'Question wording & RFail (\\%) & 95\\% CI \\\\', '\\midrule']
labels = {'canonical': 'Canonical', 'contain': 'Contain', 'hear': 'Hear', 'occur': 'Occur'}
for t in templates:
    r = out['templates'][t]
    rows.append(labels[t] + f" & {100 * r['rate']:.1f} & " +
                '--'.join(f'{100 * v:.1f}' for v in r['ci95']) + ' \\\\')
rows += ['\\bottomrule', '\\end{tabular}']
(G / 'counterfactual_interface_table.tex').write_text('\n'.join(rows) + '\n')
print(json.dumps({k: v for k, v in out.items() if k != 'source_files'}, indent=2))
