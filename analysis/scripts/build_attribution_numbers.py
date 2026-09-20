#!/usr/bin/env python3
"""Attribution-state decomposition of correct mixture answers.

Reads generated/counterfactual_inputs.json (eight-model frozen export) and
writes generated/attribution_numbers.tex, generated/attribution_table.tex,
generated/attribution_results.json.

States for a correct mixture answer (m_f > 0):
  grounded             : G_abs = dt - max_j |dj| > 0
  context-dominated    : H > 0 and G_abs <= 0
  competitor-confused  : dt > 0 and H <= 0
  ungrounded           : dt <= 0
with H = dt - max(0, max_j dj).
"""
import json, math, os, sys
from collections import defaultdict, Counter
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INP = os.path.join(ROOT, 'generated', 'counterfactual_inputs.json')
OUT_TEX = os.path.join(ROOT, 'generated', 'attribution_numbers.tex')
OUT_TAB = os.path.join(ROOT, 'generated', 'attribution_table.tex')
OUT_JSON = os.path.join(ROOT, 'generated', 'attribution_results.json')
SEED = 20260907
DRAWS = 10000

NAMES = {
    'audio_flamingo_3': 'Audio Flamingo 3',
    'mimo_audio_7b': 'MiMo-Audio-7B',
    'qwen25_omni_3b': 'Qwen2.5-Omni-3B',
    'qwen25_omni_7b': 'Qwen2.5-Omni-7B',
    'qwen_audio_aha_7b': '\\quad + AHA (LoRA)',
    'qwen2_audio_7b': 'Qwen2-Audio-7B',
    'qwen3_omni_30b': 'Qwen3-Omni-30B',
    'step_audio2_mini': 'Step-Audio 2 Mini',
}
ORDER = ['audio_flamingo_3', 'mimo_audio_7b', 'qwen2_audio_7b', 'qwen25_omni_3b',
         'qwen25_omni_7b', 'qwen_audio_aha_7b', 'qwen3_omni_30b', 'step_audio2_mini']

# absent-query yes rates (annotation-disjoint absent panel, pooled over wordings/mappings)
ABSENT_YES = {
    'audio_flamingo_3': 0.062, 'mimo_audio_7b': 0.096, 'qwen25_omni_3b': 0.121,
    'qwen25_omni_7b': 0.262, 'qwen2_audio_7b': 0.512, 'qwen3_omni_30b': 0.254,
    'qwen_audio_aha_7b': 0.127, 'step_audio2_mini': 0.531,
}


def prepare(u):
    mf, mt, mj = u['mf'], u['mt'], u['mj']
    dt = mf - mt
    dj = [mf - m for m in mj]
    max_dj = max(dj)
    max_abs = max(abs(x) for x in dj)
    H = dt - max(0.0, max_dj)
    G = dt - max_abs
    u['dt'], u['maxdj'], u['H'], u['G'] = dt, max_dj, H, G
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
    # paradigm checks
    u['target_only_pass'] = dt > 0                      # score-based target-only check
    u['flip_fail'] = u['correct'] and (mt > 0 or any(m <= 0 for m in mj))  # answer-flip check
    return u


def load():
    d = json.load(open(INP))
    us = [prepare(u) for u in d['panels']['baseline']]
    rms = [prepare(u) for u in d['panels']['global_rms']]
    return d['models'], us, rms


def tuple_ids(us):
    return sorted({u['source_tuple_id'] for u in us})


def boot_ratio(us, num, den, tuples, rng_draws):
    """Bootstrap ratio of pooled counts over source tuples."""
    idx = {t: i for i, t in enumerate(tuples)}
    n = np.zeros(len(tuples)); dn = np.zeros(len(tuples))
    for u in us:
        i = idx[u['source_tuple_id']]
        if den(u):
            dn[i] += 1
            if num(u):
                n[i] += 1
    point = n.sum() / dn.sum() if dn.sum() else float('nan')
    samp = (n[rng_draws].sum(1) / np.maximum(dn[rng_draws].sum(1), 1))
    lo, hi = np.percentile(samp, [2.5, 97.5])
    return point, lo, hi, int(n.sum()), int(dn.sum())


def boot_diff(us_a, us_b, num, den, tuples, rng_draws):
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
    sa = na[rng_draws].sum(1) / np.maximum(da[rng_draws].sum(1), 1)
    sb = nb[rng_draws].sum(1) / np.maximum(db[rng_draws].sum(1), 1)
    lo, hi = np.percentile(sa - sb, [2.5, 97.5])
    return point, lo, hi


def spearman(x, y):
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def main():
    models, us, rms = load()
    tuples = tuple_ids(us)
    rng = np.random.default_rng(SEED)
    draws = rng.integers(0, len(tuples), size=(DRAWS, len(tuples)))
    mac = {}; res = {}

    def number(k, v): mac[k] = f'{v:,}'
    def pct(k, v, d=1): mac[k] = f'{100*v:.{d}f}\\%'
    def ci(k, lo, hi, d=1): mac[k] = f'{100*lo:.{d}f}--{100*hi:.{d}f}\\%'
    def fx(v, d): t = f'{100*v:.{d}f}'; return '0.0' if t == '-0.0' else t
    def pp(k, v, d=1): mac[k] = fx(v, d)
    def ppci(k, lo, hi, d=1): mac[k] = f'{fx(lo, d)},\\,{fx(hi, d)}'

    correct = lambda u: u['correct']
    number('AttrModelCount', len(models))
    number('AttrConditionCount', len(us))
    ncorrect = sum(1 for u in us if u['correct'])
    number('AttrCorrectCount', ncorrect)
    pct('AttrAccuracy', ncorrect / len(us))

    # --- state shares among correct answers
    for st, key in [('grounded', 'Grounded'), ('context', 'Context'),
                    ('confused', 'Confused'), ('ungrounded', 'Ungrounded')]:
        p, lo, hi, n, dn = boot_ratio(us, lambda u, s=st: u['state'] == s, correct, tuples, draws)
        pct(key, p); ci(key + 'CI', lo, hi); number(key + 'Count', n)
        res[st] = dict(rate=p, lo=lo, hi=hi, count=n, denom=dn)
    p, lo, hi, n, dn = boot_ratio(us, lambda u: u['nongrounded'], correct, tuples, draws)
    pct('NonGrounded', p); ci('NonGroundedCI', lo, hi); number('NonGroundedCount', n)
    res['nongrounded'] = dict(rate=p, lo=lo, hi=hi, count=n, denom=dn)
    p, lo, hi, n, dn = boot_ratio(us, lambda u: u['H'] <= 0, correct, tuples, draws)
    pct('RankFail', p); ci('RankFailCI', lo, hi); number('RankFailCount', n)
    rankfail_pooled = p

    # ungrounded with strictly higher support after target removal
    p, lo, hi, n, dn = boot_ratio(us, lambda u: u['dt'] < 0, correct, tuples, draws)
    pct('UngroundedStrict', p); number('UngroundedStrictCount', n)
    # ungrounded that still answer 'present' on the target-removed view
    p, lo, hi, n, dn = boot_ratio(us, lambda u: u['mt'] > 0, lambda u: u['state'] == 'ungrounded', tuples, draws)
    pct('UngroundedStillYes', p)

    # --- per-model table (lean: Acc / confused / ungrounded / rank-fail)
    rows = []; per_model = {}
    for m in ORDER:
        um = [u for u in us if u['model_id'] == m]
        acc = sum(1 for u in um if u['correct']) / len(um)
        nc = sum(1 for u in um if u['correct'])
        shares = {st: sum(1 for u in um if u['state'] == st) / nc
                  for st in ['grounded', 'context', 'confused', 'ungrounded']}
        ng = 1 - shares['grounded']
        rf = sum(1 for u in um if u['correct'] and u['H'] <= 0) / nc
        per_model[m] = dict(acc=acc, nongrounded=ng, rankfail=rf, yes=ABSENT_YES[m], **shares)
        rows.append((NAMES[m], acc, shares['confused'], shares['ungrounded'], rf))
    pooled = (ncorrect / len(us), res['confused']['rate'], res['ungrounded']['rate'], rankfail_pooled)
    with open(OUT_TAB, 'w') as f:
        f.write('\\begin{tabular}{lrrrr}\n\\toprule\n')
        f.write('Model & Acc. & Conf. & Ungr. & RFail \\\\\n\\midrule\n')
        for r in rows:
            f.write(f'{r[0]} & ' + ' & '.join(f'{100*x:.1f}' for x in r[1:]) + ' \\\\\n')
        f.write('\\midrule\nPooled & ' + ' & '.join(f'{100*x:.1f}' for x in pooled) + ' \\\\\n')
        f.write('\\bottomrule\n\\end{tabular}\n')
    ngs = [per_model[m]['nongrounded'] for m in ORDER]
    pp('NonGroundedMin', min(ngs)); pp('NonGroundedMax', max(ngs))
    mac['NonGroundedMinModel'] = NAMES[ORDER[int(np.argmin(ngs))]]
    mac['NonGroundedMaxModel'] = NAMES[ORDER[int(np.argmax(ngs))]]

    # --- correlation with absent yes-rate (8 models)
    ys = [ABSENT_YES[m] for m in ORDER]
    ung = [per_model[m]['ungrounded'] for m in ORDER]
    conf = [per_model[m]['confused'] for m in ORDER]
    rf = [per_model[m]['ungrounded'] + per_model[m]['confused'] for m in ORDER]
    mac['SpearmanUngrounded'] = f'{spearman(ys, ung):.2f}'
    mac['SpearmanRankFail'] = f'{spearman(ys, rf):.2f}'
    mac['SpearmanNonGrounded'] = f'{spearman(ys, ngs):.2f}'
    mac['SpearmanContext'] = f'{spearman(ys, [per_model[m]["context"] for m in ORDER]):.2f}'
    res['spearman'] = dict(ungrounded=spearman(ys, ung), rankfail=spearman(ys, rf),
                           nongrounded=spearman(ys, ngs))

    # --- AHA vs same base (Qwen2.5-Omni-7B)
    for tag, m in [('Base', 'qwen25_omni_7b'), ('AHA', 'qwen_audio_aha_7b')]:
        pm = per_model[m]
        pct(tag + 'Acc', pm['acc']); pct(tag + 'Ungrounded', pm['ungrounded'])
        pct(tag + 'Confused', pm['confused']); pct(tag + 'Context', pm['context'])
        pct(tag + 'NonGrounded', pm['nongrounded']); pct(tag + 'Yes', pm['yes'])
        pct(tag + 'RankFail', pm['ungrounded'] + pm['confused'])
    ua = [u for u in us if u['model_id'] == 'qwen_audio_aha_7b']
    ub = [u for u in us if u['model_id'] == 'qwen25_omni_7b']
    for st, key in [('ungrounded', 'Ungrounded'), ('context', 'Context'), ('confused', 'Confused')]:
        pt, lo, hi = boot_diff(ua, ub, lambda u, s=st: u['state'] == s, correct, tuples, draws)
        pp('AHADelta' + key, pt); ppci('AHADelta' + key + 'CI', lo, hi)
    pt, lo, hi = boot_diff(ua, ub, lambda u: u['nongrounded'], correct, tuples, draws)
    pp('AHADeltaNonGrounded', pt); ppci('AHADeltaNonGroundedCI', lo, hi)
    pt, lo, hi = boot_diff(ua, ub, lambda u: u['H'] <= 0, correct, tuples, draws)
    pp('AHADeltaRankFail', pt); ppci('AHADeltaRankFailCI', lo, hi)

    # --- paradigm comparison
    # target-only score check: passes (dt>0) but non-grounded / rank-failure
    p, lo, hi, n, dn = boot_ratio(us, lambda u: u['dt'] > 0, lambda u: u['correct'] and u['H'] <= 0, tuples, draws)
    pct('TargetOnlyMissRank', p); ci('TargetOnlyMissRankCI', lo, hi); number('TargetOnlyMissRankCount', n)
    p, lo, hi, n, dn = boot_ratio(us, lambda u: u['dt'] > 0, lambda u: u['nongrounded'], tuples, draws)
    pct('TargetOnlyMissNonGrounded', p); ci('TargetOnlyMissNonGroundedCI', lo, hi); number('TargetOnlyMissNonGroundedCount', n)
    # answer-flip check flags among grounded answers
    p, lo, hi, n, dn = boot_ratio(us, lambda u: u['flip_fail'], lambda u: u['state'] == 'grounded', tuples, draws)
    pct('FlipFlagGrounded', p); ci('FlipFlagGroundedCI', lo, hi); number('FlipFlagGroundedCount', n)
    p, lo, hi, n, dn = boot_ratio(us, lambda u: u['flip_fail'], correct, tuples, draws)
    pct('FlipFlagAll', p); ci('FlipFlagAllCI', lo, hi); number('FlipFlagAllCount', n)
    # answer-flip check passes among non-grounded (misses)
    p, lo, hi, n, dn = boot_ratio(us, lambda u: not u['flip_fail'], lambda u: u['nongrounded'], tuples, draws)
    pct('FlipMissNonGrounded', p); ci('FlipMissNonGroundedCI', lo, hi); number('FlipMissNonGroundedCount', n)
    p, lo, hi, n, dn = boot_ratio(us, lambda u: not u['flip_fail'], lambda u: u['correct'] and u['H'] <= 0, tuples, draws)
    pct('FlipMissRank', p); number('FlipMissRankCount', n)
    # grounded but target-removed view still 'present' (unmasking / bias)
    p, lo, hi, n, dn = boot_ratio(us, lambda u: u['mt'] > 0, lambda u: u['state'] == 'grounded', tuples, draws)
    pct('GroundedTargetStillYes', p)
    p, lo, hi, n, dn = boot_ratio(us, lambda u: any(m <= 0 for m in u['mj']), lambda u: u['state'] == 'grounded', tuples, draws)
    pct('GroundedCompetitorReject', p)

    # --- same-sample coverage / false-alarm cross-tab (Sec. 3.3)
    # MATCH's C-I rule, expressed on our views: the target-removed view is
    # answered 'absent' (m_t <= 0). As a check it flags an answer when that
    # view is still read as 'present' (m_t > 0). It therefore COVERS a
    # ranking failure iff m_t > 0, and misses only those with m_t <= 0.
    p, lo, hi, n, dn = boot_ratio(us, lambda u: u['mt'] > 0, lambda u: u['correct'] and u['H'] <= 0, tuples, draws)
    pct('CICoverRank', p); ci('CICoverRankCI', lo, hi); number('CICoverRankCount', n)
    # what C-I flags in total, and how much of that is ordering-consistent
    p, lo, hi, n, dn = boot_ratio(us, lambda u: u['mt'] > 0, correct, tuples, draws)
    pct('CIFlagAll', p); number('CIFlagAllCount', n)
    p, lo, hi, n, dn = boot_ratio(us, lambda u: u['mt'] > 0, lambda u: u['correct'] and u['H'] > 0, tuples, draws)
    pct('CIFlagOrdered', p); number('CIFlagOrderedCount', n)
    # every-view-correct rule: misses no ranking failure by construction; the
    # share of its flags that are ordering-consistent is FlipFlagGrounded.
    # the ungrounded share of ranking failures (the complement of Confused);
    # equivalently the coverage of the target-only score check, which alarms
    # exactly when dt <= 0
    p, lo, hi, n, dn = boot_ratio(us, lambda u: u['dt'] <= 0, lambda u: u['correct'] and u['H'] <= 0, tuples, draws)
    pct('UngroundedShareRank', p); number('UngroundedShareRankCount', n)
    pct('ScoreCoverRank', p); ci('ScoreCoverRankCI', lo, hi); number('ScoreCoverRankCount', n)
    # precision of the two answer-level rules: share of their alarms that are
    # genuine ordering conflicts
    p, lo, hi, n, dn = boot_ratio(us, lambda u: u['H'] <= 0, lambda u: u['correct'] and u['mt'] > 0, tuples, draws)
    pct('CIFlagPrecision', p); number('CIFlagPrecisionCount', n)
    # ranking failures that even the answer-correctness rule on the
    # target-removed view (C-I) cannot see
    p, lo, hi, n, dn = boot_ratio(us, lambda u: u['mt'] <= 0 and u['dt'] > 0,
                                 lambda u: u['correct'] and u['H'] <= 0, tuples, draws)
    pct('OnlyRivalMissRank', p); ci('OnlyRivalMissRankCI', lo, hi); number('OnlyRivalMissRankCount', n)
    # share of each check's flags that are ordering-consistent answers
    n_ci_flag = sum(1 for u in us if u['correct'] and u['mt'] > 0)
    n_ci_ord = sum(1 for u in us if u['correct'] and u['mt'] > 0 and u['H'] > 0)
    pct('CIFlagOrderedShare', n_ci_ord / n_ci_flag)
    n_flip = sum(1 for u in us if u['correct'] and u['flip_fail'])
    n_flip_ord = sum(1 for u in us if u['correct'] and u['flip_fail'] and u['H'] > 0)
    pct('FlipFlagOrderedShare', n_flip_ord / n_flip)
    pct('FlipFlagOrderedAll', n_flip_ord / len([u for u in us if u['correct']]))
    pct('CIFlagOrderedAll', n_ci_ord / len([u for u in us if u['correct']]))
    number('AttrMixtureCount', 3456)

    # --- threshold sensitivity: the reported share is conditional on a correct
    # mixture answer (mf > 0). Because H is computed from the SAME margins, its
    # SIGN does not depend on where that correctness threshold is placed; only
    # which answers enter the set does. We therefore re-report the share while
    # varying the threshold tau over the mf distribution, so a reader can see
    # how much of the number is the threshold and how much is the statistic.
    allh = float(np.mean([u['H'] <= 0 for u in us]))
    pct('ThreshFreeRankShare', allh)
    number('ThreshFreeN', len(us))
    TAU_LEVELS = [-1.0, -0.5, 0.0, 0.5, 1.0, 2.0]
    TAU_TAGS = {  # TeX control sequences cannot contain digits
        -1.0: 'NegOne', -0.5: 'NegHalf', 0.0: 'Zero',
        0.5: 'Half', 1.0: 'One', 2.0: 'Two',
    }
    tau_rows = []
    for tau in TAU_LEVELS:
        sel = [u for u in us if u['mf'] > tau]
        if not sel:
            continue
        rate = float(np.mean([u['H'] <= 0 for u in sel]))
        tau_rows.append((tau, len(sel), rate))
    for tau, n_sel, rate in tau_rows:
        tag = TAU_TAGS[tau]
        pct(f'ThreshShift{tag}Share', rate)
        number(f'ThreshShift{tag}N', n_sel)
    mac['ThreshShiftMinShare'] = f'{100*min(r for _, _, r in tau_rows):.1f}\\%'
    mac['ThreshShiftMaxShare'] = f'{100*max(r for _, _, r in tau_rows):.1f}\\%'
    res['threshold_sensitivity'] = dict(
        free_share=allh, free_n=len(us),
        rows=[dict(tau=t, n=n_, share=r) for t, n_, r in tau_rows])

    # --- stress axes on common-correct subsets
    def common_units(us_, keyf, levels):
        """Group by `keyf`, one record per level value.

        The inner dict is keyed by the level, so a collision would silently
        replace a record and shrink the subset. Round 15 shipped exactly that
        bug downstream (a grouping key missing distractor_replicate), so the
        collision is now fatal here, at the source of the totals every other
        script cross-checks against.
        """
        pick = (lambda u: u['snr_db']) if levels == 'snr' else (
            (lambda u: u['distractor_count']) if levels == 'd' else
            (lambda u: u['overlap']))
        groups = defaultdict(dict)
        for u in us_:
            k, lv = keyf(u), pick(u)
            if lv in groups[k]:
                raise ValueError(
                    f'common_units: 键 {k!r} 的层 {lv!r} 出现多次; '
                    f'内层字典会静默覆盖记录, 令共同正确子集偏小。'
                    f'请检查分组键是否缺少区分字段(如 distractor_replicate)。')
            groups[k][lv] = u
        # Every record must land in exactly one bucket: nothing dropped.
        if sum(len(v) for v in groups.values()) != len(us_):
            raise ValueError('common_units: 有记录未进入任何分组。')
        return groups

    # SNR: pair over (model, question, overlap, d, replicate)
    keyf = lambda u: (u['model_id'], u['question_id'], u['overlap'], u['distractor_count'], u['distractor_replicate'])
    g = common_units(us, keyf, 'snr')
    common = [v for v in g.values() if len(v) == 3 and all(x['correct'] for x in v.values())]
    number('SnrCommonCount', len(common))
    snr_res = {}
    for snr, key in [(5.0, 'High'), (0.0, 'Mid'), (-5.0, 'Low')]:
        sub = [v[snr] for v in common]
        for crit, ck in [('nongrounded', 'NonGrounded'), ('H', 'RankFail'), ('ungrounded', 'Ungrounded'), ('context', 'Context')]:
            f = (lambda u: u['nongrounded']) if crit == 'nongrounded' else (lambda u: u['H'] <= 0) if crit == 'H' else (lambda u, c=crit: u['state'] == c)
            p, lo, hi, n, dn = boot_ratio(sub, f, lambda u: True, tuples, draws)
            pct(f'Snr{ck}{key}', p); ci(f'Snr{ck}{key}CI', lo, hi)
            snr_res[(crit, snr)] = (p, lo, hi)
    for crit, ck in [('nongrounded', 'NonGrounded'), ('H', 'RankFail'), ('ungrounded', 'Ungrounded')]:
        f = (lambda u: u['nongrounded']) if crit == 'nongrounded' else (lambda u: u['H'] <= 0) if crit == 'H' else (lambda u: u['state'] == 'ungrounded')
        pt, lo, hi = boot_diff([v[-5.0] for v in common], [v[5.0] for v in common], f, lambda u: True, tuples, draws)
        pp(f'Snr{ck}Delta', pt); ppci(f'Snr{ck}DeltaCI', lo, hi)
        inc = 0
        lifts = []
        for m in ORDER:
            cm = [v for v in common if v[5.0]['model_id'] == m]
            a = np.mean([f(v[-5.0]) for v in cm]); b = np.mean([f(v[5.0]) for v in cm])
            inc += a > b
            lifts.append(a - b)
        number(f'Snr{ck}IncreasingModels', inc)
        # smallest per-model change, so "a positive change in every model" can be
        # stated with its worst case rather than asserted unquantified
        pp(f'Snr{ck}MinModelLift', min(lifts))
    # accuracy across SNR (all units) for the "accuracy vs grounding" contrast
    for snr, key in [(5.0, 'High'), (0.0, 'Mid'), (-5.0, 'Low')]:
        sub = [u for u in us if u['snr_db'] == snr]
        pct(f'SnrAcc{key}', np.mean([u['correct'] for u in sub]))
        # non-grounded among all correct at this SNR
        p, lo, hi, n, dn = boot_ratio(sub, lambda u: u['nongrounded'], correct, tuples, draws)
        pct(f'SnrNonGroundedAll{key}', p)
    # RMS-matched SNR gradient for non-grounded
    g = common_units(rms, keyf, 'snr')
    common_r = [v for v in g.values() if len(v) == 3 and all(x['correct'] for x in v.values())]
    for snr, key in [(5.0, 'High'), (-5.0, 'Low')]:
        p, lo, hi, n, dn = boot_ratio([v[snr] for v in common_r], lambda u: u['nongrounded'], lambda u: True, tuples, draws)
        pct(f'RmsNonGrounded{key}', p)
    pt, lo, hi = boot_diff([v[-5.0] for v in common_r], [v[5.0] for v in common_r], lambda u: u['nongrounded'], lambda u: True, tuples, draws)
    pp('RmsNonGroundedDelta', pt); ppci('RmsNonGroundedDeltaCI', lo, hi)
    p, lo, hi, n, dn = boot_ratio(rms, lambda u: u['nongrounded'], correct, tuples, draws)
    pct('RmsNonGrounded', p); ci('RmsNonGroundedCI', lo, hi)

    # Distractor count: pair (model, question, snr, overlap); d2 has one replicate, d1 has two
    def dgroups(us_):
        g = defaultdict(lambda: defaultdict(list))
        for u in us_:
            g[(u['model_id'], u['question_id'], u['snr_db'], u['overlap'])][u['distractor_count']].append(u)
        return g
    g = dgroups(us)
    common_d = [v for v in g.values() if all(x['correct'] for lst in v.values() for x in lst) and 1 in v and 2 in v]
    number('DistCommonCount', len(common_d))
    d1 = [x for v in common_d for x in v[1]]; d2 = [x for v in common_d for x in v[2]]
    for crit, ck in [('nongrounded', 'NonGrounded'), ('H', 'RankFail')]:
        f = (lambda u: u['nongrounded']) if crit == 'nongrounded' else (lambda u: u['H'] <= 0)
        p1, lo1, hi1, _, _ = boot_ratio(d1, f, lambda u: True, tuples, draws)
        p2, lo2, hi2, _, _ = boot_ratio(d2, f, lambda u: True, tuples, draws)
        pct(f'Dist{ck}One', p1); pct(f'Dist{ck}Two', p2)
        pt, lo, hi = boot_diff(d2, d1, f, lambda u: True, tuples, draws)
        pp(f'Dist{ck}Delta', pt); ppci(f'Dist{ck}DeltaCI', lo, hi)
    pct('DistAccOne', np.mean([u['correct'] for u in us if u['distractor_count'] == 1]))
    pct('DistAccTwo', np.mean([u['correct'] for u in us if u['distractor_count'] == 2]))

    # Overlap: pair (model, question, snr, d, replicate)
    keyo = lambda u: (u['model_id'], u['question_id'], u['snr_db'], u['distractor_count'], u['distractor_replicate'])
    g = common_units(us, keyo, 'o')
    common_o = [v for v in g.values() if len(v) == 2 and all(x['correct'] for x in v.values())]
    number('OverlapCommonCount', len(common_o))
    for crit, ck in [('nongrounded', 'NonGrounded'), ('H', 'RankFail')]:
        f = (lambda u: u['nongrounded']) if crit == 'nongrounded' else (lambda u: u['H'] <= 0)
        pf, _, _, _, _ = boot_ratio([v['full'] for v in common_o], f, lambda u: True, tuples, draws)
        pq, _, _, _, _ = boot_ratio([v['partial'] for v in common_o], f, lambda u: True, tuples, draws)
        pct(f'Overlap{ck}Full', pf); pct(f'Overlap{ck}Partial', pq)
        pt, lo, hi = boot_diff([v['full'] for v in common_o], [v['partial'] for v in common_o], f, lambda u: True, tuples, draws)
        pp(f'Overlap{ck}Delta', pt); ppci(f'Overlap{ck}DeltaCI', lo, hi)

    # --- hero example: context-dominated case with large |dj| and small dt, single distractor, high margins
    cands = [u for u in us if u['state'] == 'context' and u['distractor_count'] == 1 and u['mf'] > 1.0]
    cands.sort(key=lambda u: (min(u['dt'], 0) - min(u['maxdj'], 0)), reverse=True)
    # prefer: dt close to 0 or slightly positive; removing competitor raises margin a lot
    cands.sort(key=lambda u: (-u['maxdj']) - abs(u['dt']), reverse=True)
    res['hero_candidates'] = [{k: u[k] for k in ['model_id', 'condition_id', 'target_category', 'snr_db', 'overlap', 'mf', 'mt', 'mj', 'dt', 'maxdj', 'H', 'G']} for u in cands[:12]]
    ucands = [u for u in us if u['state'] == 'ungrounded' and u['distractor_count'] == 1 and u['mf'] > 1.0 and u['mt'] > u['mf']]
    ucands.sort(key=lambda u: u['mt'] - u['mf'], reverse=True)
    res['ungrounded_candidates'] = [{k: u[k] for k in ['model_id', 'condition_id', 'target_category', 'snr_db', 'overlap', 'mf', 'mt', 'mj', 'dt', 'maxdj', 'H', 'G']} for u in ucands[:12]]

    # --- target-category dependence: which sounds are answered correctly without being heard
    cat_n = Counter(u['target_category'] for u in us if u['correct'])
    cat_k = Counter(u['target_category'] for u in us if u['nongrounded'])
    cat_rate = sorted(((cat_k[c] / cat_n[c], c) for c in cat_n), reverse=True)
    top_cats = [c for _, c in cat_rate[:10]]; bot_cats = [c for _, c in cat_rate[-10:]]
    res['category_rates'] = [dict(category=c, rate=r, n=cat_n[c]) for r, c in cat_rate]
    number('CatCount', len(cat_n))
    for cs, key in [(top_cats, 'CatTop'), (bot_cats, 'CatBot')]:
        p, lo, hi, n, dn = boot_ratio(us, lambda u: u['nongrounded'], lambda u, cs=cs: u['correct'] and u['target_category'] in cs, tuples, draws)
        pct(key, p); ci(key + 'CI', lo, hi); number(key + 'Count', dn)
    mac['CatTopList'] = ', '.join(top_cats)
    mac['CatBotList'] = ', '.join(bot_cats)
    pct('CatMaxRate', cat_rate[0][0]); mac['CatMaxName'] = cat_rate[0][1]
    consistent = 0
    for m in ORDER:
        um = [u for u in us if u['model_id'] == m and u['correct']]
        a = np.mean([u['nongrounded'] for u in um if u['target_category'] in top_cats])
        b = np.mean([u['nongrounded'] for u in um if u['target_category'] in bot_cats])
        consistent += a > b
    number('CatConsistentModels', consistent)
    # rank agreement of per-model category rates
    per_cat = {}
    for m in ORDER:
        um = [u for u in us if u['model_id'] == m and u['correct']]
        n_ = Counter(u['target_category'] for u in um); k_ = Counter(u['target_category'] for u in um if u['nongrounded'])
        per_cat[m] = {c: k_[c] / n_[c] for c in n_ if n_[c] >= 10}
    rhos = []
    for i, a in enumerate(ORDER):
        for b in ORDER[i + 1:]:
            ks = sorted(set(per_cat[a]) & set(per_cat[b]))
            rhos.append(spearman([per_cat[a][c] for c in ks], [per_cat[b][c] for c in ks]))
    mac['CatPairwiseRho'] = f'{np.median(rhos):.2f}'

    # --- AHA paired: where does the accuracy loss fall?
    BASE, AHA = 'qwen25_omni_7b', 'qwen_audio_aha_7b'
    pair = defaultdict(dict)
    for u in us:
        if u['model_id'] in (BASE, AHA):
            pair[u['condition_id']][u['model_id']] = u
    lost = [v[BASE] for v in pair.values() if len(v) == 2 and v[BASE]['correct'] and not v[AHA]['correct']]
    kept = [v[BASE] for v in pair.values() if len(v) == 2 and v[BASE]['correct'] and v[AHA]['correct']]
    gained = [v[AHA] for v in pair.values() if len(v) == 2 and not v[BASE]['correct'] and v[AHA]['correct']]
    number('AHALostCount', len(lost)); number('AHAKeptCount', len(kept)); number('AHAGainedCount', len(gained))
    p, lo, hi, n, dn = boot_ratio(lost, lambda u: u['nongrounded'], lambda u: True, tuples, draws)
    pct('AHALostNonGrounded', p); ci('AHALostNonGroundedCI', lo, hi)
    p, lo, hi, n, dn = boot_ratio(kept, lambda u: u['nongrounded'], lambda u: True, tuples, draws)
    pct('AHAKeptNonGrounded', p); ci('AHAKeptNonGroundedCI', lo, hi)
    pt, lo, hi = boot_diff(lost, kept, lambda u: u['nongrounded'], lambda u: True, tuples, draws)
    pp('AHALostKeptDelta', pt); ppci('AHALostKeptDeltaCI', lo, hi)

    res['per_model'] = per_model
    res['macros'] = mac
    with open(OUT_TEX, 'w') as f:
        f.write('% Generated by scripts/build_attribution_numbers.py\n')
        for k, v in mac.items():
            f.write(f'\\newcommand{{\\{k}}}{{{v}}}\n')
    with open(OUT_JSON, 'w') as f:
        json.dump(res, f, indent=1, default=float)
    for k, v in mac.items():
        print(f'{k:36s} {v}')


if __name__ == '__main__':
    main()
