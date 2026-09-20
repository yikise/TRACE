#!/usr/bin/env python3
"""Revision-response analyses for the TRACE paper (no new model inference).

Reads generated/counterfactual_inputs.json and writes
generated/revision_numbers.tex and generated/revision_results.json.

Computes:
  R1. RMS-matched ranking-failure SNR gradient (common-correct units)
  R2. Phase-scrambled ranking-failure SNR gradient (extra control)
  R3. AHA common-correct state transition matrix (base -> AHA)
  R4. Threshold-matched control: base margins shifted so its present-question
      recall matches AHA; state shares recomputed on the reduced correct set
  R5. Distractor-count chance-candidates control: d=2 ranking failure with
      one randomly kept rival vs both rivals
  R6. H statistics: per-model ranking-failure rates, near-zero |H| mass
"""
import json, os
from collections import defaultdict, Counter
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INP = os.path.join(ROOT, 'generated', 'counterfactual_inputs.json')
OUT_TEX = os.path.join(ROOT, 'generated', 'revision_numbers.tex')
OUT_JSON = os.path.join(ROOT, 'generated', 'revision_results.json')
SEED = 20260907
DRAWS = 10000

ORDER = ['audio_flamingo_3', 'mimo_audio_7b', 'qwen2_audio_7b', 'qwen25_omni_3b',
         'qwen25_omni_7b', 'qwen_audio_aha_7b', 'qwen3_omni_30b', 'step_audio2_mini']
BASE, AHA = 'qwen25_omni_7b', 'qwen_audio_aha_7b'
STATES = ['grounded', 'context', 'confused', 'ungrounded']


def prepare(u):
    mf, mt, mj = u['mf'], u['mt'], u['mj']
    dt = mf - mt
    dj = [mf - m for m in mj]
    H = dt - max(0.0, max(dj))
    G = dt - max(abs(x) for x in dj)
    u['dt'], u['H'], u['G'] = dt, H, G
    u['correct'] = mf > 0
    if not u['correct']:
        u['state'] = None
    elif dt <= 0:
        u['state'] = 'ungrounded'
    elif H <= 0:
        u['state'] = 'confused'
    elif G <= 0:
        u['state'] = 'context'
    else:
        u['state'] = 'grounded'
    u['nongrounded'] = u['correct'] and u['state'] != 'grounded'
    u['rankfail'] = u['correct'] and H <= 0
    return u


def load():
    d = json.load(open(INP))
    return ({'base': [prepare(u) for u in d['panels']['baseline']],
             'rms': [prepare(u) for u in d['panels']['global_rms']],
             'phase': [prepare(u) for u in d['panels']['phase_scrambled']]},
            d['models'])


def tuple_ids(us):
    return sorted({u['source_tuple_id'] for u in us})


def boot_ratio(us, num, den, tuples, draws):
    idx = {t: i for i, t in enumerate(tuples)}
    n = np.zeros(len(tuples)); dn = np.zeros(len(tuples))
    for u in us:
        i = idx[u['source_tuple_id']]
        if den(u):
            dn[i] += 1
            if num(u):
                n[i] += 1
    point = n.sum() / dn.sum() if dn.sum() else float('nan')
    samp = (n[draws].sum(1) / np.maximum(dn[draws].sum(1), 1))
    lo, hi = np.percentile(samp, [2.5, 97.5])
    return point, lo, hi, int(n.sum()), int(dn.sum())


def boot_diff(us_a, us_b, num, den, tuples, draws):
    idx = {t: i for i, t in enumerate(tuples)}
    def counts(us):
        n = np.zeros(len(tuples)); dn = np.zeros(len(tuples))
        for u in us:
            i = idx[u['source_tuple_id']]
            if den(u):
                dn[i] += 1
                if num(u):
                    n[i] += 1
        return n, dn
    na, da = counts(us_a); nb, db = counts(us_b)
    point = na.sum() / da.sum() - nb.sum() / db.sum()
    sa = na[draws].sum(1) / np.maximum(da[draws].sum(1), 1)
    sb = nb[draws].sum(1) / np.maximum(db[draws].sum(1), 1)
    lo, hi = np.percentile(sa - sb, [2.5, 97.5])
    return point, lo, hi


def common_snr(us):
    g = defaultdict(dict)
    for u in us:
        g[(u['model_id'], u['question_id'], u['overlap'], u['distractor_count'],
           u['distractor_replicate'])][u['snr_db']] = u
    return [v for v in g.values() if len(v) == 3 and all(x['correct'] for x in v.values())]


def main():
    panels, models = load()
    us, rms, phase = panels['base'], panels['rms'], panels['phase']
    tuples = tuple_ids(us)
    rng = np.random.default_rng(SEED)
    draws = rng.integers(0, len(tuples), size=(DRAWS, len(tuples)))
    mac, res = {}, {}

    def pct(k, v, d=1): mac[k] = f'{100*v:.{d}f}\\%'
    def ci(k, lo, hi, d=1): mac[k] = f'{100*lo:.{d}f}--{100*hi:.{d}f}\\%'
    def fx(v, d): t = f'{100*v:.{d}f}'; return '0.0' if t == '-0.0' else t
    def pp(k, v, d=1): mac[k] = fx(v, d)
    def ppci(k, lo, hi, d=1): mac[k] = f'{fx(lo, d)},\\,{fx(hi, d)}'
    def num(k, v): mac[k] = f'{v:,}'

    # ---------- R1/R2: ranking-failure SNR gradient on RMS / phase panels
    for pname, panel in [('Rms', rms), ('Phase', phase)]:
        common = common_snr(panel)
        num(f'{pname}SnrCommonCount', len(common))
        for snr, key in [(5.0, 'High'), (0.0, 'Mid'), (-5.0, 'Low')]:
            sub = [v[snr] for v in common]
            p, lo, hi, n, dn = boot_ratio(sub, lambda u: u['rankfail'], lambda u: True, tuples, draws)
            pct(f'{pname}RankFail{key}', p); ci(f'{pname}RankFail{key}CI', lo, hi)
            res[f'{pname.lower()}_rankfail_snr'] = res.get(f'{pname.lower()}_rankfail_snr', {})
            res[f'{pname.lower()}_rankfail_snr'][snr] = dict(rate=p, lo=lo, hi=hi, n=n, dn=dn)
        pt, lo, hi = boot_diff([v[-5.0] for v in common], [v[5.0] for v in common],
                               lambda u: u['rankfail'], lambda u: True, tuples, draws)
        pp(f'{pname}RankFailDelta', pt); ppci(f'{pname}RankFailDeltaCI', lo, hi)
        res[f'{pname.lower()}_rankfail_delta'] = dict(delta=pt, lo=lo, hi=hi)

    # ---------- R3: AHA common-correct transition matrix
    pair = defaultdict(dict)
    for u in us:
        if u['model_id'] in (BASE, AHA):
            pair[u['condition_id']][u['model_id']] = u
    both = [v for v in pair.values() if len(v) == 2]
    bc = [v for v in both if v[BASE]['correct'] and v[AHA]['correct']]
    num('AHACommonCorrectCount', len(bc))
    mac['AHACommonCorrectShare'] = f'{100*len(bc)/sum(1 for v in both if v[BASE]["correct"]):.1f}\\%'
    trans = Counter((v[BASE]['state'], v[AHA]['state']) for v in bc)
    res['aha_transition'] = {f'{a}->{b}': c for (a, b), c in sorted(trans.items())}
    for st in STATES:
        row = sum(trans[(st, b)] for b in STATES)
        num(f'AHATransFrom{st.capitalize()}', row)
        for b in STATES:
            if row:
                pct(f'AHATrans{st.capitalize()}To{b.capitalize()}', trans[(st, b)] / row)
    # within common-correct: paired state shares (base vs AHA on identical conditions)
    ub = [v[BASE] for v in bc]; ua = [v[AHA] for v in bc]
    for tag, sub in [('BaseCC', ub), ('AHACC', ua)]:
        p, lo, hi, n, dn = boot_ratio(sub, lambda u: u['nongrounded'], lambda u: True, tuples, draws)
        pct(f'{tag}NonGrounded', p); ci(f'{tag}NonGroundedCI', lo, hi)
        p2, lo2, hi2, n2, dn2 = boot_ratio(sub, lambda u: u['rankfail'], lambda u: True, tuples, draws)
        pct(f'{tag}RankFail', p2); ci(f'{tag}RankFailCI', lo2, hi2)
        res[tag.lower()] = dict(nongrounded=(p, lo, hi), rankfail=(p2, lo2, hi2))
    pt, lo, hi = boot_diff(ua, ub, lambda u: u['nongrounded'], lambda u: True, tuples, draws)
    pp('AHACCDeltaNonGrounded', pt); ppci('AHACCDeltaNonGroundedCI', lo, hi)
    pt, lo, hi = boot_diff(ua, ub, lambda u: u['rankfail'], lambda u: True, tuples, draws)
    pp('AHACCDeltaRankFail', pt); ppci('AHACCDeltaRankFailCI', lo, hi)
    # grounded-and-correct share of ALL present questions (absolute yield)
    for tag, m in [('Base', BASE), ('AHA', AHA)]:
        um = [u for u in us if u['model_id'] == m]
        gy = sum(1 for u in um if u['correct'] and u['state'] == 'grounded') / len(um)
        pct(f'{tag}GroundedYield', gy)
        res[f'{tag.lower()}_grounded_yield'] = gy

    # ---------- R4: threshold-matched control for base
    ub_all = [u for u in us if u['model_id'] == BASE]
    ua_all = [u for u in us if u['model_id'] == AHA]
    aha_acc = np.mean([u['correct'] for u in ua_all])
    mfs = np.sort(np.array([u['mf'] for u in ub_all]))
    tau = float(np.quantile(mfs, 1.0 - aha_acc))
    mac['BaseTauMatch'] = f'{tau:.3f}'
    res['tau_match'] = tau
    shifted = []
    for u in ub_all:
        v = dict(u)
        v['correct'] = u['mf'] > tau
        v['state'] = u['state'] if v['correct'] else None
        v['nongrounded'] = v['correct'] and u['state'] != 'grounded'
        v['rankfail'] = v['correct'] and u['H'] <= 0
        shifted.append(v)
    acc_t = np.mean([v['correct'] for v in shifted])
    pct('BaseAccTau', acc_t)
    p, lo, hi, n, dn = boot_ratio(shifted, lambda u: u['nongrounded'], lambda u: u['correct'], tuples, draws)
    pct('BaseTauNonGrounded', p); ci('BaseTauNonGroundedCI', lo, hi)
    p, lo, hi, n, dn = boot_ratio(shifted, lambda u: u['rankfail'], lambda u: u['correct'], tuples, draws)
    pct('BaseTauRankFail', p); ci('BaseTauRankFailCI', lo, hi)
    # which states does the threshold remove?
    removed = [u for u, v in zip(ub_all, shifted) if u['correct'] and not v['correct']]
    res['tau_removed_by_state'] = dict(Counter(u['state'] for u in removed))
    for st in STATES:
        if removed:
            pct(f'TauRemoved{st.capitalize()}', sum(1 for u in removed if u['state'] == st) / len(removed))
    pt, lo, hi = boot_diff(ua_all, shifted, lambda u: u['nongrounded'], lambda u: u['correct'], tuples, draws)
    pp('AHAvsTauDeltaNonGrounded', pt); ppci('AHAvsTauDeltaNonGroundedCI', lo, hi)
    res['tau_control'] = dict(acc=acc_t, removed=len(removed),
                              removed_by_state=dict(Counter(u['state'] for u in removed)))

    # ---------- R5: distractor chance-candidates control
    rng2 = np.random.default_rng(SEED + 1)
    # The kept rival is drawn ONCE per unit below, after d2 exists, and that single
    # assignment is reused everywhere. Previously `single_rival_rf` was called twice
    # against the same shared RNG stream, so the printed B column and the printed
    # delta came from two different random draws and the row did not satisfy
    # B - A = delta.
    kept_rival = {}
    def single_rival_rf(u):
        if not u['correct']:
            return False
        dj = u['mf'] - u['mj'][kept_rival[id(u)]]
        return (u['dt'] - max(0.0, dj)) <= 0
    g = defaultdict(lambda: defaultdict(list))
    for u in us:
        g[(u['model_id'], u['question_id'], u['snr_db'], u['overlap'])][u['distractor_count']].append(u)
    common_d = [v for v in g.values() if 1 in v and 2 in v
                and all(x['correct'] for lst in v.values() for x in lst)]
    d1 = [x for v in common_d for x in v[1]]; d2 = [x for v in common_d for x in v[2]]
    for u in d2:                       # one draw per unit, reused by every consumer
        kept_rival[id(u)] = int(rng2.integers(0, len(u['mj'])))
    d2_single = [dict(u, rankfail=single_rival_rf(u)) for u in d2]
    for crit, ck in [(lambda u: u['rankfail'], 'RankFail'), (lambda u: u['nongrounded'], 'NonGrounded')]:
        p1, _, _, _, _ = boot_ratio(d1, crit, lambda u: True, tuples, draws)
        p2, _, _, _, _ = boot_ratio(d2, crit, lambda u: True, tuples, draws)
        p2s, lo2s, hi2s, _, _ = boot_ratio(d2, single_rival_rf if ck == 'RankFail' else crit,
                                           lambda u: True, tuples, draws)
        pct(f'Dist{ck}OneC', p1); pct(f'Dist{ck}TwoC', p2)
        if ck == 'RankFail':
            pct('DistRankFailTwoSingleRival', p2s); ci('DistRankFailTwoSingleRivalCI', lo2s, hi2s)
            # paired delta of the one-kept-rival control against the d1 baseline,
            # with its own bootstrap CI (previously the uncontrolled Delta was reused)
            pt, lo, hi = boot_diff(d2_single, d1, lambda u: u['rankfail'], lambda u: True, tuples, draws)
            pp('DistRankFailSingleRivalDelta', pt); ppci('DistRankFailSingleRivalDeltaCI', lo, hi)
            res['dist_single_rival'] = dict(d1=p1, d2_both=p2, d2_single=p2s,
                                            delta_single_vs_d1=pt, delta_lo=lo, delta_hi=hi)
    # symmetric control: d1 with BOTH replicates pooled as candidate max
    def d1_pooled_rf(v):
        xs = v[1]
        if not all(x['correct'] for x in xs):
            return None
        # treat the two d1 replicates' rivals as two candidates
        djs = []
        for x in xs:
            djs.extend(x['mf'] - m for m in x['mj'])
        dt = xs[0]['dt']
        return (dt - max(0.0, max(djs))) <= 0
    d1p = [d1_pooled_rf(v) for v in common_d]
    d1p = [x for x in d1p if x is not None]
    mac['DistRankFailOnePooledRivals'] = f'{100*np.mean(d1p):.1f}\\%'
    res['dist_d1_pooled_rivals'] = float(np.mean(d1p))

    # ---------- R7: main-panel calibration for the fixed equal-gain interface panel
    sel = [u for u in us if u['snr_db'] == -5.0 and u['distractor_count'] == 2 and u['overlap'] == 'full']
    p, lo, hi, n, dn = boot_ratio(sel, lambda u: u['rankfail'], lambda u: u['correct'], tuples, draws)
    pct('RankFailLowSnrDTwoFull', p); ci('RankFailLowSnrDTwoFullCI', lo, hi)
    res['rankfail_lowsnr_d2_full'] = dict(rate=p, lo=lo, hi=hi, n=n, dn=dn)

    # ---------- R6: H statistics
    correct = [u for u in us if u['correct']]
    Hs = np.array([u['H'] for u in correct])
    for eps, tag in [(0.1, 'Tiny'), (0.25, 'Small'), (0.5, 'Mid')]:
        pct(f'HNearZero{tag}', np.mean(np.abs(Hs) < eps))
        res[f'h_nearzero_{eps}'] = float(np.mean(np.abs(Hs) < eps))
    per_model_rf = {}
    SAFE = {'audio_flamingo_3': 'AudioFlamingoThree', 'mimo_audio_7b': 'MimoAudioSevenB',
            'qwen2_audio_7b': 'QwenTwoAudio', 'qwen25_omni_3b': 'QwenTwoFiveOmniThreeB',
            'qwen25_omni_7b': 'QwenTwoFiveOmniSevenB', 'qwen_audio_aha_7b': 'QwenAudioAhaSevenB',
            'qwen3_omni_30b': 'QwenThreeOmni', 'step_audio2_mini': 'StepAudioTwoMini'}
    for m in ORDER:
        um = [u for u in us if u['model_id'] == m and u['correct']]
        per_model_rf[m] = float(np.mean([u['rankfail'] for u in um]))
        pct(f'RankFail{SAFE[m]}', per_model_rf[m], d=1)
    res['per_model_rankfail'] = per_model_rf
    qs = np.percentile(Hs, [5, 25, 50, 75, 95])
    res['h_quantiles'] = dict(zip(['5', '25', '50', '75', '95'], [float(q) for q in qs]))

    with open(OUT_TEX, 'w') as f:
        f.write('% Generated by scripts/build_revision_numbers.py\n')
        for k, v in mac.items():
            f.write(f'\\newcommand{{\\{k}}}{{{v}}}\n')
    with open(OUT_JSON, 'w') as f:
        json.dump(res, f, indent=1, default=float)
    for k, v in mac.items():
        print(f'{k:36s} {v}')


if __name__ == '__main__':
    main()
