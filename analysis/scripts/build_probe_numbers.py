"""Construct-validity probe: run TRACE on detectors whose decision rule is known.

Input : generated/view_energies.json  (exact per-view stem RMS, 3456 conditions)
Output: generated/probe_numbers.tex, generated/probe_results.json

Every probe is a synthetic detector whose view score is an explicit function of
the stem energies, so its ground truth is known by construction. With
L(z) = log(1 + z/f) and per-condition floor f:

  oracle        s = L(E_t)                  reads the queried sound only
  shortcut      s = L(max_j E_j)            reads rivals only, never the target
  loudness      s = L(E_t + sum_j E_j)      sums per-stem RMS, no source identity

E_i is the RMS of stem i, NOT the mixture waveform's own RMS. Summing stems
overstates the true mixture RMS by ~47% on average over this panel, because
independent sources add in power rather than amplitude. The key stays 'loudness'
for pipeline compatibility (probe_boot.json / build_probe_macros.py) and the
paper calls it the energy-sum proxy: it reads exactly one scalar carrying no
source identity, which is what the control needs. Recomputing it from the true
view RMS instead moves the rate from 45.0% to 51.0%.
  additive(a)   s = L(E_t) - a*L(max_j E_j) reads the target; rival subtracts
  normalize(c)  s = E_t / (f + c*E_j)       reads the target; rival divides it

additive and normalize are the two families that matter for interpretation.
For any score of the *additively separable* form s = A(E_t) + B(E_j) with
A(0)=B(0)=0, the context-dominated state is unreachable among correct answers:
context needs m_j + m_t >= 2 m_f, but separability gives m_j + m_t = m_f. A
multiplicative (divisive-normalisation) score has no such constraint, so a
detector that provably reads the target can still be placed in the
context-dominated state.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
GEN = ROOT / 'generated'
STATE_OF = ['ungrounded', 'confused', 'context', 'grounded']
FLOOR_FRAC = 1e-3


def load():
    return json.loads((GEN / 'view_energies.json').read_text())['units']


def view_energies(u):
    idx = u['included_indices']
    t = u['target_index']
    stem = {int(k): v for k, v in u['stem_rms'].items()}
    peak = max(stem.values()) or 1.0
    f = FLOOR_FRAC * peak

    def surv(view, i):
        if view == 'remove_target':
            return i != t
        if view.startswith('remove_off_'):
            return i != int(view.rsplit('_', 1)[1])
        return True

    def energies(view):
        e_t = stem[t] if surv(view, t) else 0.0
        return e_t, [stem[i] for i in idx if i != t and surv(view, i)]

    offs = sorted(k for k in u['view_total_rms'] if k.startswith('remove_off_'))
    return f, energies('full'), energies('remove_target'), [energies(k) for k in offs]


def scores(u, fn):
    f, full, tr, offs = view_energies(u)
    return fn(f, *full), fn(f, *tr), [fn(f, *o) for o in offs]


def classify(mf, mt, mj):
    if mf <= 0:
        return None
    dt = mf - mt
    H = dt - max(0.0, max(mf - x for x in mj))
    G = dt - max(abs(mf - x) for x in mj)
    if dt <= 0:
        return 'ungrounded'
    if H <= 0:
        return 'confused'
    if G <= 0:
        return 'context'
    return 'grounded'


def run(units, fn):
    states = [s for s in (classify(*scores(u, fn)) for u in units) if s is not None]
    n = len(states)
    if not n:
        return {'n_correct': 0, 'rankfail': None, 'shares': {s: None for s in STATE_OF}}
    return {'n_correct': n,
            'rankfail': (states.count('confused') + states.count('ungrounded')) / n,
            'shares': {s: states.count(s) / n for s in STATE_OF}}


def main():
    units = load()
    Lf = lambda f, z: float(np.log1p(max(z, 0.0) / f))
    probes = {
        'oracle': lambda f, e_t, r: Lf(f, e_t),
        'shortcut': lambda f, e_t, r: Lf(f, max(r)) if r else Lf(f, 0.0),
        'loudness': lambda f, e_t, r: Lf(f, e_t + sum(r)),
    }
    for a in [0.5, 1.0, 2.0]:
        probes[f'additive_a{a}'] = (lambda a: lambda f, e_t, r:
                                    Lf(f, e_t) - a * (Lf(f, max(r)) if r else 0.0))(a)
    for c in [1.0, 10.0]:
        probes[f'normalize_c{c}'] = (lambda c: lambda f, e_t, r:
                                     e_t / (f + c * (max(r) if r else 0.0)))(c)

    truth = {'oracle': 'reads the queried sound only',
             'shortcut': 'reads rivals only, never the queried sound',
             'loudness': 'sums per-stem RMS, no source identity',
             'additive': 'reads the queried sound, rival subtracted',
             'normalize': 'reads the queried sound, rival divides it'}

    res = {'n_conditions': len(units), 'floor_frac': FLOOR_FRAC,
           'input_sha256': hashlib.sha256((GEN / 'view_energies.json').read_bytes()).hexdigest(),
           'probes': {}, 'ground_truth': truth}
    for name, fn in probes.items():
        res['probes'][name] = run(units, fn)
        r = res['probes'][name]
        print(f"{name:16s} n={r['n_correct']:5d} rankfail={100*(r['rankfail'] or 0):5.1f}% "
              + ' '.join(f'{s[:4]}:{100*(r["shares"][s] or 0):5.1f}' for s in STATE_OF))

    mac = {}
    def put(k, v):
        mac[k] = v
    def pct(k, name, st=None):
        r = res['probes'][name]
        v = r['rankfail'] if st is None else r['shares'][st]
        mac[k] = 'n/a' if v is None else f'{100*v:.1f}\\%'
    mac['ProbeConditionCount'] = f'{len(units):,}'
    pct('ProbeOracleRankFail', 'oracle'); pct('ProbeOracleGrounded', 'oracle', 'grounded')
    pct('ProbeShortcutRankFail', 'shortcut'); pct('ProbeShortcutUngrounded', 'shortcut', 'ungrounded')
    pct('ProbeLoudnessRankFail', 'loudness'); pct('ProbeLoudnessConfused', 'loudness', 'confused')
    for a in ['0_5', '1_0', '2_0']:
        k = f'additive_a{a.replace("_", ".")}'
        tag = 'Add' + a.replace('_', '')
        pct(f'Probe{tag}Context', k, 'context'); pct(f'Probe{tag}Grounded', k, 'grounded')
    for c in ['1_0', '10_0']:
        k = f'normalize_c{c.replace("_", ".")}'
        tag = 'Norm' + c.replace('_', '')
        pct(f'Probe{tag}Context', k, 'context'); pct(f'Probe{tag}Grounded', k, 'grounded')
        pct(f'Probe{tag}RankFail', k)
    (GEN / 'probe_numbers.tex').write_text(
        '% Generated by scripts/build_probe_numbers.py\n' +
        '\n'.join(f'\\newcommand{{\\{k}}}{{{v}}}' for k, v in mac.items()) + '\n')
    (GEN / 'probe_results.json').write_text(json.dumps(res, indent=1) + '\n')


if __name__ == '__main__':
    main()
