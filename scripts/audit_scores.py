#!/usr/bin/env python3
"""Validate all released score rows and reproduce TRACE's main headline (CPU only)."""
from pathlib import Path
import argparse, collections, gzip, hashlib, json, math
ROOT = Path(__file__).resolve().parents[1]

def read_rows(path):
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', encoding='utf-8') as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)

def verdict(mf, mt, rivals):
    if not rivals or not all(math.isfinite(x) for x in [mf, mt, *rivals]):
        raise ValueError('Need finite full/target/rival margins and at least one rival')
    h = min(mf, *rivals) - mt
    return {'correct': mf > 0, 'H': h,
            'failure': mf > 0 and h <= 0,
            'ungrounded': mf > 0 and mf - mt <= 0,
            'confused': mf > 0 and mf - mt > 0 and h <= 0}

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, default=ROOT)
    ap.add_argument('--output', type=Path)
    args = ap.parse_args()
    root = args.root.resolve()
    index = json.loads((root / 'data/scores/INDEX.json').read_text())
    manifest_cache = {}
    main_results = []
    views_total = 0
    for entry in index:
        panel, model = entry['panel'], entry['model']
        path = root / entry['file']
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry['sha256']:
            raise ValueError(f'Checksum mismatch: {entry["file"]}')
        if panel not in manifest_cache:
            manifest_cache[panel] = {(c['condition_id'], v['intervention']): (c, v)
                for c in read_rows(root / f'data/manifests/{panel}.jsonl') for v in c['views']}
        expected = manifest_cache[panel]
        groups = collections.defaultdict(dict)
        seen = set()
        for row in read_rows(path):
            key = row['condition_id'], row['intervention']
            if key in seen or key not in expected:
                raise ValueError(f'Duplicate/unknown view: {panel}/{model}/{key}')
            seen.add(key)
            c, v = expected[key]
            assert row['model_id'] == model
            for field in ['margin', 'logp_present', 'logp_absent']:
                assert math.isfinite(row[field]), (panel, model, key, field)
            assert abs(row['margin'] - row['logp_present'] + row['logp_absent']) < 1e-8
            if row.get('audio_sha256') and v.get('audio_sha256'):
                assert row['audio_sha256'] == v['audio_sha256']
            assert row['y_present'] == c['y_present'] and row['y_absent'] == c['y_absent']
            groups[row['condition_id']][row['intervention']] = row['margin']
        assert seen == set(expected), (panel, model, len(seen), len(expected))
        assert len(seen) == entry['views'] and len(groups) == entry['conditions']
        views_total += len(seen)
        if panel == 'baseline':
            counts = collections.Counter()
            for views in groups.values():
                result = verdict(views['full'], views['remove_target'],
                                 [v for k, v in views.items() if k.startswith('remove_off_')])
                for k in ['correct', 'failure', 'ungrounded', 'confused']:
                    counts[k] += result[k]
            main_results.append({'model': model, 'conditions': len(groups), **counts,
                'accuracy_percent': 100 * counts['correct'] / len(groups),
                'failure_percent': 100 * counts['failure'] / counts['correct']})
    totals = {k: sum(r[k] for r in main_results) for k in
              ['conditions', 'correct', 'failure', 'ungrounded', 'confused']}
    assert len(main_results) == 8
    assert totals == dict(conditions=27648, correct=18911, failure=1184, ungrounded=681, confused=503), totals
    report = {'status': 'pass', 'score_files': len(index), 'score_views': views_total,
              'main': main_results, 'pooled': totals,
              'ranking_failure_percent': 100 * totals['failure'] / totals['correct'],
              'target_only_missed_percent': 100 * totals['confused'] / totals['failure'],
              'inference_rerun': False}
    print(json.dumps(report, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + '\n')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
