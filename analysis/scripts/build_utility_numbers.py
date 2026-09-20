"""Diagnostic utility of the H decomposition, and answer-mapping robustness.

Two analyses requested by external review.

U1. Diagnostic utility. Table 1's relation (H<=0 implies the answer-flip check
    fails) is definitional, so it cannot by itself show that splitting
    counterfactual errors into ordering conflicts and threshold-repairable
    errors carries information beyond a binary error label. Here we test that
    directly: among answers that are correct on the frozen mixture AND flagged
    by the answer-flip check, we split them by sign(H) and compare their
    ranking-failure risk on the independently re-paired panel, which redraws
    both distractors while holding the target clip, gain and SNR fixed. The
    re-paired panel covers only the full-overlap two-distractor cells for three
    models, so the baseline side is restricted to exactly those cells and each
    baseline unit is joined to its five re-paired replicates.

    Estimator. Rates are POOLED ratios, sum(failures)/sum(eligible redraws),
    matching the convention of the paper's other statistics (e.g. the frozen
    vs re-paired comparison in Sec. 5.4). A baseline unit is eligible when at
    least one of its five redraws is still answered correctly, and the
    denominator is the number of correctly answered redraws. The adjusted
    contrast standardises conflict units onto threshold-repairable units within
    (model, mixture-margin bin, SNR) cells, so the contrast cannot be read off
    the fact that conflicts sit at lower margins or in different models. The
    per-model split is reported as an unadjusted contrast only; it is not
    claimed to be significant model by model.

U2. Answer-mapping robustness. The mapping is balanced 50/50 over the panel
    but not over the correct subset. We report the ranking-failure share and
    the competition gradient separately under each mapping, using the ground
    truth y_present recorded in fixed_panel.json (never inferred). The
    common-correct key mirrors SnrCommonCount in build_attribution_numbers.py,
    i.e. it includes distractor_replicate, so the two mapping groups partition
    the same 5,466 units reported in Table 4.

Both use source-tuple cluster bootstrap, 10,000 draws, SEED=20260907.
"""
import json, os, re, sys
from collections import defaultdict
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from checks import (  # noqa: E402
    SelfCheckFailure, assert_partition, assert_total, bucket_multi,
    bucket_unique, report, verified,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NUM = os.path.join(ROOT, 'generated', 'attribution_numbers.tex')
SNR_COMMON_MACRO = 'SnrCommonCount'
BASE = os.path.join(ROOT, 'generated', 'counterfactual_inputs.json')
REPAIR = os.path.join(ROOT, 'generated', 'repair_inputs.json')
PANEL = os.path.join(ROOT, 'generated', 'fixed_panel.json')
OUT_TEX = os.path.join(ROOT, 'generated', 'utility_numbers.tex')
OUT_JSON = os.path.join(ROOT, 'generated', 'utility_results.json')
SEED = 20260907
DRAWS = 10000
BIN = 0.5
MIN_CELL_UNITS = 3
FIVE_REPAIR_REDRAWS = 5
ORDER = ['audio_flamingo_3', 'mimo', 'qwen25_omni_3b', 'qwen25_omni_7b',
         'qwen2_audio_7b', 'qwen3_omni_30b', 'qwen_audio_aha_7b',
         'step_audio2_mini']


def prepare(u):
    mf, mt, mj = u['mf'], u['mt'], u['mj']
    dt = mf - mt
    max_dj = max(mf - x for x in mj)
    u = dict(u)
    u['dt'] = dt
    u['H'] = dt - max(0.0, max_dj)
    u['correct'] = mf > 0
    u['flip_fail'] = u['correct'] and (mt > 0 or any(x <= 0 for x in mj))
    return u


def load():
    base = [prepare(u) for u in json.load(open(BASE))['panels']['baseline']]
    repair = [prepare(u) for u in json.load(open(REPAIR))['units']]
    truth = {}
    for r in json.load(open(PANEL))['rows']:
        truth[r['question_id']] = r['y_present']
    return base, repair, truth


def mf_bin(x):
    return round(x / BIN) * BIN


def read_macro(path, name):
    """Read a published \\newcommand value as an int, or None if absent.

    Lets one build script cross-check its own total against a macro another
    script already published, which is how the round-15 miscount would have
    been caught automatically.
    """
    if not os.path.exists(path):
        return None
    m = re.search(r'\\newcommand\{\\' + re.escape(name) + r'\}\{([^}]*)\}',
                  open(path).read())
    if not m:
        return None
    digits = re.sub(r'[^0-9]', '', m.group(1))
    return int(digits) if digits else None


def main():
    base, repair, truth = load()
    rng = np.random.default_rng(SEED)
    res = {}
    mac = {}

    # ---------------- U1: diagnostic utility ----------------
    # bucket_multi: five repair redraws deliberately share one baseline key, so
    # every record must be collected (a plain dict would keep only the last).
    rep_key = bucket_multi(
        repair,
        lambda r: (r['model_id'], r['original_source_tuple_id'],
                   r['original_target_index'], r['snr_db']),
        'U1 重配对键 (每键 5 个副本)')
    assert_total(sum(len(v) for v in rep_key.values()), len(repair),
                 'U1 重配对副本总数')

    units = []
    for u in base:
        if not (u['correct'] and u['flip_fail']):
            continue
        if u['overlap'] != 'full' or u['distractor_count'] != 2:
            continue
        rs = rep_key.get((u['model_id'], u['source_tuple_id'],
                          u['target_index'], u['snr_db']))
        if not rs:
            continue
        units.append(dict(
            u,
            n_ok=sum(1 for r in rs if r['correct']),
            n_fail=sum(1 for r in rs if r['correct'] and r['H'] <= 0),
            conflict=u['H'] <= 0))

    by_tup = defaultdict(list)
    for x in units:
        by_tup[x['source_tuple_id']].append(x)
    tups = sorted(by_tup)

    def pooled(sample, conflict):
        g = [x for t in sample for x in by_tup[t] if x['conflict'] == conflict]
        dn = sum(x['n_ok'] for x in g)
        return (sum(x['n_fail'] for x in g) / dn) if dn else None

    def raw_delta(sample):
        a, b = pooled(sample, True), pooled(sample, False)
        return None if a is None or b is None else a - b

    def matched_delta(sample):
        """Pooled weighting, standardised within (model, margin bin, SNR)."""
        A = [x for t in sample for x in by_tup[t] if x['conflict']]
        B = [x for t in sample for x in by_tup[t] if not x['conflict']]
        cells = defaultdict(lambda: [0, 0, 0])  # fails, eligible redraws, units
        for x in B:
            c = cells[(x['model_id'], mf_bin(x['mf']), x['snr_db'])]
            c[0] += x['n_fail']
            c[1] += x['n_ok']
            c[2] += 1
        cells = {k: v for k, v in cells.items() if v[1] and v[2] >= MIN_CELL_UNITS}
        num = den = 0
        for x in A:
            c = cells.get((x['model_id'], mf_bin(x['mf']), x['snr_db']))
            if not c or not x['n_ok']:
                continue
            num += x['n_fail'] - x['n_ok'] * (c[0] / c[1])
            den += x['n_ok']
        return (num / den, den, len(cells)) if den else (None, 0, len(cells))

    boot_raw, boot_m = [], []
    pt_raw = raw_delta(tups)
    got = matched_delta(tups)
    pt_m, used_pt, cells_pt = (got if got else (None, 0, 0))
    for _ in range(DRAWS):
        s = [tups[i] for i in rng.integers(0, len(tups), len(tups))]
        a = raw_delta(s)
        if a is not None:
            boot_raw.append(a)
        b = matched_delta(s)
        if b[0] is not None:
            boot_m.append(b[0])
    lo_r, hi_r = np.percentile(np.array(boot_raw), [2.5, 97.5])
    lo_m, hi_m = np.percentile(np.array(boot_m), [2.5, 97.5])

    nA = sum(1 for x in units if x['conflict'])
    nB = sum(1 for x in units if not x['conflict'])
    nA_ok = sum(1 for x in units if x['conflict'] and x['n_ok'])
    nB_ok = sum(1 for x in units if not x['conflict'] and x['n_ok'])
    dA = sum(x['n_ok'] for x in units if x['conflict'])
    dB = sum(x['n_ok'] for x in units if not x['conflict'])
    fA, fB = pooled(tups, True), pooled(tups, False)

    # unadjusted contrast per model, on the same pooled convention
    per_model = {}
    for m in ORDER:
        ga = [x for x in units if x['conflict'] and x['model_id'] == m]
        gb = [x for x in units if not x['conflict'] and x['model_id'] == m]
        da, db_ = sum(x['n_ok'] for x in ga), sum(x['n_ok'] for x in gb)
        a = sum(x['n_fail'] for x in ga) / da if da else None
        b = sum(x['n_fail'] for x in gb) / db_ if db_ else None
        if a is not None and b is not None:
            per_model[m] = dict(conflict=a, repairable=b, delta=a - b,
                                n_conflict=len(ga), n_repairable=len(gb))
    n_pos = sum(1 for v in per_model.values() if v['delta'] > 0)

    # Every baseline unit must land in exactly one H-sign group, and each unit's
    # eligible redraws must be a subset of its five redraws.
    assert_partition({'conflict': nA, 'repairable': nB}, len(units),
                     'U1 H 符号二分 = 入选基线单元')
    assert_total(nA + nB, len(units), 'U1 入选基线单元合计')
    for x in units:
        if not (0 <= x['n_ok'] <= FIVE_REPAIR_REDRAWS):
            raise SelfCheckFailure(
                f'U1 重配分母越界: {x["condition_id"]} n_ok={x["n_ok"]} '
                f'不在 [0, {FIVE_REPAIR_REDRAWS}]')
    verified('U1 每个单元的重配分母在 0..5')

    mac['UtilConflictN'] = f'{nA:,}'
    mac['UtilRepairableN'] = f'{nB:,}'
    mac['UtilTotalN'] = f'{nA+nB:,}'
    mac['UtilConflictEligibleN'] = f'{nA_ok:,}'
    mac['UtilRepairableEligibleN'] = f'{nB_ok:,}'
    mac['UtilConflictRedrawN'] = f'{dA:,}'
    mac['UtilRepairableRedrawN'] = f'{dB:,}'
    mac['UtilTupleN'] = f'{len(tups)}'
    mac['UtilConflictFail'] = f'{100*fA:.1f}\\%'
    mac['UtilRepairableFail'] = f'{100*fB:.1f}\\%'
    mac['UtilRawDelta'] = f'{100*pt_raw:+.1f}'
    mac['UtilRawDeltaCI'] = f'{100*lo_r:+.1f},\\,{100*hi_r:+.1f}'
    mac['UtilMatchedDelta'] = f'{100*pt_m:+.1f}'
    mac['UtilMatchedDeltaCI'] = f'{100*lo_m:+.1f},\\,{100*hi_m:+.1f}'
    mac['UtilModelsPositive'] = f'{n_pos}'
    mac['UtilMatchedUsed'] = f'{used_pt:,}'
    mac['UtilCellCount'] = f'{cells_pt:,}'
    res['utility'] = dict(
        n_conflict=nA, n_repairable=nB, n_eligible_conflict=nA_ok,
        n_eligible_repairable=nB_ok, denom_conflict=dA, denom_repairable=dB,
        n_tuples=len(tups), conflict_fail=fA, repairable_fail=fB,
        raw=dict(point=pt_raw, lo=float(lo_r), hi=float(hi_r)),
        matched=dict(point=pt_m, lo=float(lo_m), hi=float(hi_m), used=used_pt),
        per_model=per_model, models_positive=n_pos)

    # ---------------- U2: answer-mapping robustness ----------------
    for u in base:
        u['mapping'] = truth[u['question_id']]
    # Common-correct key mirrors SnrCommonCount in build_attribution_numbers.py
    # (same five fields, including distractor_replicate). bucket_multi keeps all
    # three SNR levels per key; the uniqueness of the key itself is asserted so
    # a missing field can never silently collapse records again.
    key5 = lambda u: (u['model_id'], u['question_id'], u['overlap'],  # noqa: E731
                      u['distractor_count'], u['distractor_replicate'])
    # The 5-field key intentionally holds one record per SNR level, so the key
    # that must be unique adds SNR. Proving the 6-field key unique is what
    # establishes that the 5-field key is complete, i.e. that nothing collapsed
    # into it -- the round-15 failure mode.
    uniq = bucket_unique(base, lambda u: key5(u) + (u['snr_db'],),
                         'U2 唯一键 (model,question,overlap,d,replicate,snr)')
    assert_total(len(uniq), len(base), 'U2 唯一键覆盖全部基线单元')
    common = bucket_multi(base, key5, 'U2 SNR 三元组')
    for k, v in common.items():
        if len(v) != len({x['snr_db'] for x in v}):
            raise SelfCheckFailure(
                f'U2 SNR 三元组: 键 {k!r} 内 SNR 重复; 键仍缺字段。')
    assert_total(sum(len(v) for v in common.values()), len(base),
                 'U2 三元组未丢失任何基线单元')
    # Convert the guarded groups into {snr_db: unit} maps, the shape the
    # statistics below expect. The length check above already proved each key
    # holds exactly one unit per SNR, so this rebuild cannot drop a record.
    triples = [{x['snr_db']: x for x in v}
               for v in common.values() if len(v) == 3]
    common = [v for v in triples if all(x['correct'] for x in v.values())]
    res['mapping'] = {'n_common_total': len(common)}
    group_sizes = {}
    for mp in ('A', 'B'):
        cor = [u for u in base if u['mapping'] == mp and u['correct']]
        share = 100 * sum(1 for u in cor if u['H'] <= 0) / len(cor)
        sub = [v for v in common if all(x['mapping'] == mp for x in v.values())]
        hi = 100 * sum(1 for v in sub if v[5.0]['H'] <= 0) / len(sub)
        mid = 100 * sum(1 for v in sub if v[0.0]['H'] <= 0) / len(sub)
        lo = 100 * sum(1 for v in sub if v[-5.0]['H'] <= 0) / len(sub)
        tag = 'A' if mp == 'A' else 'B'
        mac[f'Map{tag}Correct'] = f'{len(cor):,}'
        mac[f'Map{tag}Fail'] = f'{share:.1f}\\%'
        mac[f'Map{tag}CommonN'] = f'{len(sub):,}'
        mac[f'Map{tag}High'] = f'{hi:.1f}\\%'
        mac[f'Map{tag}Mid'] = f'{mid:.1f}\\%'
        mac[f'Map{tag}Low'] = f'{lo:.1f}\\%'
        mac[f'Map{tag}Gradient'] = f'{lo-hi:+.1f}'
        group_sizes[mp] = len(sub)
        res['mapping'][mp] = dict(n=len(cor), fail=share, high=hi, mid=mid,
                                  low=lo, gradient=lo - hi, n_common=len(sub))
    # The guard for the round-15 bug: the two mapping groups must reconstruct
    # the 5,466 common-correct units that Table 4 already reports. Also verify
    # against the published macro itself, so the two scripts cannot drift.
    assert_partition(group_sizes, len(common),
                     'U2 映射分组合计 = 共同正确单位',
                     detail='两组单元数相加必须等于论文表 4 的 5,466')
    published = read_macro(NUM, SNR_COMMON_MACRO)
    if published is None:
        # Deliberately fatal rather than a silent skip: this cross-check is the
        # guard that catches a mis-built partition, so quietly not running it
        # would reproduce the very failure mode it exists to prevent.
        raise SelfCheckFailure(
            f'找不到 {os.path.relpath(NUM, ROOT)} 中的 \\{SNR_COMMON_MACRO}, '
            f'无法交叉核对共同正确单位总数。\n'
            f'  请先运行 scripts/build_attribution_numbers.py 生成该宏; '
            f'两个脚本必须使用同一筛选口径。')
    assert_total(len(common), published,
                 'U2 共同正确单位 vs 已发表的 SnrCommonCount',
                 detail=f'来自 {os.path.relpath(NUM, ROOT)}')
    nA_all = sum(1 for u in base if u['mapping'] == 'A')
    mac['MapBalanceA'] = f'{100*nA_all/len(base):.1f}\\%'
    res['mapping']['balance_a'] = nA_all / len(base)

    with open(OUT_TEX, 'w') as f:
        f.write('% Generated by scripts/build_utility_numbers.py\n')
        for k, v in mac.items():
            f.write(f'\\newcommand{{\\{k}}}{{{v}}}\n')
    with open(OUT_JSON, 'w') as f:
        json.dump(res, f, indent=1, sort_keys=True)

    print('=== U1 诊断效用 (answer-flip 失败样本内, 池化口径) ===')
    print(f'  排序冲突 H<=0 : 单元={nA} 有效单元={nA_ok} 重配分母={dA} 失败率={100*fA:.1f}%')
    print(f'  阈值可修复 H>0: 单元={nB} 有效单元={nB_ok} 重配分母={dB} 失败率={100*fB:.1f}%')
    print(f'  未调整差 = {100*pt_raw:+.2f}pp  95%CI=[{100*lo_r:+.2f},{100*hi_r:+.2f}]')
    print(f'  模型内标准化差 = {100*pt_m:+.2f}pp  95%CI=[{100*lo_m:+.2f},{100*hi_m:+.2f}] (纳入重配 {used_pt})')
    print(f'  逐模型未调整差为正: {n_pos}/{len(per_model)}')
    for m, v in per_model.items():
        print(f'    {m:20s} {100*v["conflict"]:5.1f}% vs {100*v["repairable"]:5.1f}%  = {100*v["delta"]:+6.1f}pp')
    print(f'  合计基线单元={nA+nB}  tuple={len(tups)}')
    print('=== U2 映射稳健性 ===')
    print(f'  共同正确单位总数 = {res["mapping"]["n_common_total"]:,} (应=5,466)')
    for mp in ('A', 'B'):
        d = res['mapping'][mp]
        print(f"  present={mp}: 正确n={d['n']:5d} 失败率={d['fail']:.2f}% | 共同正确n={d['n_common']:,} "
              f"+5dB={d['high']:.1f}% 0dB={d['mid']:.1f}% -5dB={d['low']:.1f}% 梯度={d['gradient']:+.2f}pp")
    print(f"  分组相加 = {res['mapping']['A']['n_common']+res['mapping']['B']['n_common']:,}")
    print(f"  面板配平 A = {100*res['mapping']['balance_a']:.1f}%")
    report()
    print(f'-> {OUT_TEX}')


if __name__ == '__main__':
    main()
