"""Matched present/absent 2x2: same acoustics and wording, different query.

Input : generated/fixed_panel.json    present arm, 10 models x 192 questions
        generated/absent_inputs.jsonl absent arm, 10 models x 512 conditions
        generated/absent_manifest.jsonl ground truth for the absent conditions
Output: generated/matched_numbers.tex, generated/matched_2x2.json

The absent panel reuses the rendered mixtures of the fixed panel: all 64 distinct
audio_sha256 values of the absent manifest also occur among the fixed panel's
`full` views. The same question wording and the same answer-key mapping are used
on both sides. The queried category necessarily changes with the arm, so the two
arms share acoustics but not the query; a "present" answer is a true positive on
the present arm and a false alarm on the absent arm. The pair reports both rates
on identical acoustics. It is not a decomposition of yes-answering into acoustic
and query-prior components: the absent categories are a different set, so a
model that ignored audio and answered from the category word alone could still
produce a rate difference.

Because three models have a near-degenerate answer-key preference, the absent
false-alarm rate is averaged over the two label mappings.
"""
import json
from pathlib import Path

import numpy as np

from texnames import model_stem, write_macros

ROOT = Path(__file__).resolve().parents[1]
GEN = ROOT / 'generated'
PAPER = ['audio_flamingo_3', 'mimo_audio_7b', 'qwen25_omni_3b', 'qwen25_omni_7b',
         'qwen2_audio_7b', 'qwen3_omni_30b', 'qwen_audio_aha_7b', 'step_audio2_mini']


def rate(rs):
    if not rs:
        return None
    return sum(1.0 if r['margin'] > 0 else (0.5 if r['margin'] == 0 else 0.0) for r in rs) / len(rs)


def tag(m):
    # letter-only stems: TeX control sequences cannot contain digits
    return model_stem(m)


def main():
    fx = json.loads((GEN / 'fixed_panel.json').read_text())
    present = fx['rows']
    models = fx['models']
    held = fx['held_out_models']
    absent = [json.loads(l) for l in (GEN / 'absent_inputs.jsonl').read_text().splitlines()]
    manifest = [json.loads(l) for l in (GEN / 'absent_manifest.jsonl').read_text().splitlines()]

    # the two arms must share acoustics, or the comparison is not matched
    fx_sha = {r['audio_sha256'] for r in present}
    ab_sha = {m['views'][0]['audio_sha256'] for m in manifest}
    assert ab_sha <= fx_sha, sorted(ab_sha - fx_sha)
    for m in manifest:
        assert m['semantic_answer_present'] is False
        assert m['annotation_audit']['superclass_disjoint'] is True

    # restrict the absent arm to the canonical wording so both arms ask the same question
    absent_canon = [r for r in absent if r['prompt_template'] == 'canonical']

    out = {'shared_acoustics': len(ab_sha), 'present_rows': len(present),
           'absent_rows': len(absent_canon), 'models': {}}
    print(f"shared acoustic hashes: {len(ab_sha)}   present rows {len(present)}   absent rows {len(absent_canon)}")
    print(f"\n{'model':22s} {'TPR(present)':>13} {'FPR(absent)':>12} {'YesA':>7} {'YesB':>7} {'d-prime':>8}  held")
    for m in models:
        p = [r for r in present if r['model_id'] == m]
        a = [r for r in absent_canon if r['model_id'] == m]
        tpr = rate(p)
        per_lm = {lm: rate([r for r in a if r['label_mapping'] == lm]) for lm in ('yes_A', 'yes_B')}
        fpr = float(np.mean(list(per_lm.values())))
        from statistics import NormalDist
        clip = lambda x: min(max(x, 1e-6), 1 - 1e-6)
        dprime = NormalDist().inv_cdf(clip(tpr)) - NormalDist().inv_cdf(clip(fpr))
        out['models'][m] = {'tpr': tpr, 'fpr': fpr, 'fpr_yes_A': per_lm['yes_A'],
                            'fpr_yes_B': per_lm['yes_B'], 'dprime': dprime,
                            'n_present': len(p), 'n_absent': len(a), 'held_out': m in held}
        print(f"{m:22s} {100*tpr:12.1f}% {100*fpr:11.1f}% {100*per_lm['yes_A']:6.1f}% "
              f"{100*per_lm['yes_B']:6.1f}% {dprime:8.2f}  {str(m in held):>5}")

    for label, grp in (('design8', [m for m in models if m not in held]), ('heldout2', held)):
        out[label] = {'tpr': float(np.mean([out['models'][m]['tpr'] for m in grp])),
                      'fpr': float(np.mean([out['models'][m]['fpr'] for m in grp]))}
        print(f"\n{label}: TPR(present)={100*out[label]['tpr']:.1f}%  FPR(absent)={100*out[label]['fpr']:.1f}%")

    (GEN / 'matched_2x2.json').write_text(json.dumps(out, indent=1) + '\n')
    mac = {'MatchSharedAcoustics': str(len(ab_sha)),
           'MatchPresentRows': f"{len(present):,}", 'MatchAbsentRows': f"{len(absent_canon):,}",
           'MatchPerModelPresent': str(len(present) // len(models)),
           'MatchPerModelAbsent': str(len(absent_canon) // len(models)),
           'MatchAbsentPooled': f"{len(manifest):,}",
           'MatchPaperTPR': f"{100*out['design8']['tpr']:.1f}\\%",
           'MatchPaperFPR': f"{100*out['design8']['fpr']:.1f}\\%",
           'MatchHeldTPR': f"{100*out['heldout2']['tpr']:.1f}\\%",
           'MatchHeldFPR': f"{100*out['heldout2']['fpr']:.1f}\\%",
           'MatchAllTPR': f"{100*np.mean([out['models'][m]['tpr'] for m in models]):.1f}\\%",
           'MatchAllFPR': f"{100*np.mean([out['models'][m]['fpr'] for m in models]):.1f}\\%"}
    for m in models:
        mac[f'Match{tag(m)}TPR'] = f"{100*out['models'][m]['tpr']:.1f}\\%"
        mac[f'Match{tag(m)}FPR'] = f"{100*out['models'][m]['fpr']:.1f}\\%"
        mac[f'Match{tag(m)}Dprime'] = f"{out['models'][m]['dprime']:.2f}"
    write_macros(GEN / 'matched_numbers.tex', mac,
                 'Generated by scripts/build_matched_numbers.py')
    print(f'\nwrote generated/matched_numbers.tex ({len(mac)} macros)')


if __name__ == '__main__':
    main()
