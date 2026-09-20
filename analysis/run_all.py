#!/usr/bin/env python3
"""Reproduce every TRACE number and Table 1-4 from the frozen inputs.

Run:
    python3 run_all.py                 # numbers + all tables + tests
    python3 run_all.py --figures       # also rebuild Fig. 1 (matplotlib only)
    python3 run_all.py --check         # verify input checksums, run nothing
    python3 run_all.py --dry-run       # print the plan
    python3 run_all.py --only build_attribution_numbers.py   # one or more steps

What this does NOT do
---------------------
* No LaTeX. Nothing here invokes latexmk/pdflatex; `generated/*.tex` are macro
  and tabular-body fragments the paper \\input{}s, and they are checked for
  digit-free control sequences by `texnames.write_macros`.
* No model inference. Every input under `inputs/frozen/` is a frozen export of
  already-scored margins; the scripts only recompute statistics.
* No Mac-specific fonts. The optional figure uses matplotlib's bundled DejaVu
  Sans with `pdf.fonttype=42`, so it renders identically off-macOS.

Pipeline order matters in two places and is enforced here:
  * `build_attribution_numbers.py` must precede `build_utility_numbers.py`
    (which reads \\SnrCommonCount back out of attribution_numbers.tex as a
    cross-check), `make_checks_table.py` (which imports its `load()`), and
    `make_contrast_table.py`.
  * `make_risk_table.py` must follow `build_category_structure.py` and
    `build_subst_numbers.py`, whose macros it asserts cell by cell.
`make_subst_table.py` is deliberately absent: `subst_table.tex` is not
\\input by the paper.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
GEN = ROOT / 'generated'
SCRIPTS = ROOT / 'scripts'
CONFIG = ROOT / 'config' / 'inputs.json'
LOGS = ROOT / 'logs'

# Steps in dependency order. Each entry is (script, what it produces).
STEPS = [
    ('test_checks.py',
     'unit tests for scripts/checks.py (the round-15 grouping-key guards)'),
    ('build_attribution_numbers.py',
     'Table 2 rows + the state shares every other script cross-checks'),
    ('build_revision_numbers.py',
     'revision-response macros (RMS/phase SNR gradients, AHA transitions)'),
    ('build_repair_numbers.py',
     're-paired-background replication numbers'),
    ('build_category_structure.py',
     'source-event quintiles + transfer to the re-paired panel (Table 3)'),
    ('build_probe_numbers.py',
     'controlled-detector probe point estimates (oracle/shortcut/loudness)'),
    ('build_utility_numbers.py',
     'H-sign diagnostic utility + answer-mapping robustness'),
    ('build_subst_numbers.py',
     'target-substitution panel statistics (Table 3)'),
    ('build_robust_numbers.py',
     'H distribution, robustness and equation-integrity checks'),
    ('build_matched_numbers.py',
     'matched present/absent 2x2 on shared acoustics'),
    ('build_counterfactual_interface.py',
     'same-audio wording/mapping replication'),
    ('make_checks_table.py',
     'Table 1: overlap decomposition of counterfactual checks'),
    ('make_contrast_table.py',
     'Table 4 + the Fig. 1(a) case macros + the Table 2 formatting pass'),
    ('make_risk_table.py',
     'Table 3: the single risk-quintile / target-substitution float'),
]

FIGURE_STEP = ('make_overview_figure.py', 'Fig. 1 (overview); requires matplotlib only')


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


# The analysis is meant to be copied off this machine, and scripts print the
# absolute paths they write to. Redact the two local prefixes so a shipped log
# never carries a home directory or a host path.
_HOME = str(Path.home())
_ROOT = str(ROOT)


def redact(text: str) -> str:
    return text.replace(_ROOT, '.').replace(_HOME, '~')


def load_config() -> dict:
    if not CONFIG.exists():
        raise SystemExit(f'missing {CONFIG}; restore the released config/inputs.json')
    return json.loads(CONFIG.read_text())


def stage_inputs(cfg: dict) -> None:
    """Copy the frozen inputs into generated/ and verify their checksums.

    generated/ is the single working directory the scripts read and write, so
    the frozen inputs are re-staged on every run. That keeps inputs and freshly
    produced outputs distinguishable: `config/inputs.json` lists exactly which
    files are inputs (checksummed here) and which are outputs.
    """
    GEN.mkdir(parents=True, exist_ok=True)
    bad = []
    for e in cfg['inputs']:
        src = ROOT / e['path']
        if not src.exists():
            raise SystemExit(f'frozen input missing: {e["path"]}')
        got = sha256(src)
        if got != e['sha256']:
            bad.append((e['path'], e['sha256'], got))
            continue
        dst = ROOT / e['staged_to']
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists() or sha256(dst) != got:
            shutil.copy2(src, dst)
    if bad:
        lines = '\n'.join(f'  {p}: config={a[:16]} actual={b[:16]}' for p, a, b in bad)
        raise SystemExit('frozen input checksum mismatch -- refusing to run:\n' + lines)
    n = len(cfg['inputs'])
    print(f'[stage] verified {n} frozen inputs (sha256) and staged them into generated/')


def check_only(cfg: dict) -> int:
    ok = True
    for e in cfg['inputs']:
        src = ROOT / e['path']
        if not src.exists():
            print(f'  MISSING {e["path"]}')
            ok = False
            continue
        got = sha256(src)
        flag = 'ok ' if got == e['sha256'] else 'BAD'
        if got != e['sha256']:
            ok = False
        print(f'  {flag} {e["path"]:52s} {got[:16]}  {e["bytes"]:>10,d} B')
    for name in cfg.get('outputs', []) + cfg.get('figures', []):
        p = GEN / name
        if p.exists():
            print(f'  out {("generated/" + name):52s} {sha256(p)[:16]}')
    print('[check] inputs ' + ('OK' if ok else 'FAILED'))
    return 0 if ok else 1


def run_step(script: str, logdir: Path, env: dict) -> tuple[int, float]:
    """Run one script with the same interpreter, streaming output to log+stdout.

    The log is redacted at the end: several scripts echo the path they wrote to,
    and this analysis is shipped to another machine.
    """
    log = logdir / (script[:-3] + '.log')
    t0 = time.time()
    with open(log, 'w') as fh:
        fh.write(f'$ {sys.executable.replace(_HOME, "~")} scripts/{script}\n')
        fh.flush()
        proc = subprocess.Popen(
            [sys.executable, '-u', str(SCRIPTS / script)],
            cwd=str(ROOT), env=env, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            fh.write(line)
        rc = proc.wait()
    log.write_text(redact(log.read_text()))
    return rc, time.time() - t0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--figures', action='store_true',
                    help='also rebuild Fig. 1 (matplotlib only; no LaTeX, no Mac fonts)')
    ap.add_argument('--check', action='store_true', help='verify input checksums and exit')
    ap.add_argument('--dry-run', action='store_true', help='print the plan and exit')
    ap.add_argument('--only', nargs='+', metavar='SCRIPT',
                    help='run only these scripts (comma-separated names also accepted)')
    args = ap.parse_args()

    cfg = load_config()
    steps = list(STEPS)
    if args.figures:
        steps.append(FIGURE_STEP)
    if args.only:
        want = {s.strip() for arg in args.only for s in arg.split(',') if s.strip()}
        known = {s for s, _ in steps}
        unknown = want - known
        if unknown:
            raise SystemExit(f'unknown step(s): {sorted(unknown)}\nknown: {sorted(known)}')
        steps = [s for s in steps if s[0] in want]

    if args.check:
        return check_only(cfg)

    print(f'TRACE number reproduction  (root={ROOT})')
    print(f'python: {sys.executable}')
    if args.dry_run:
        for i, (s, what) in enumerate(steps, 1):
            print(f'  {i:2d}. {s:36s} {what}')
        return 0

    LOGS.mkdir(parents=True, exist_ok=True)
    # Each step's log is opened with 'w', so it is overwritten rather than
    # appended and no pre-run cleanup is needed (or wanted: an unrelated log in
    # logs/ may be the only copy of an earlier diagnostic).

    stage_inputs(cfg)

    env = dict(os.environ)
    env['MPLBACKEND'] = 'Agg'
    env['MPLCONFIGDIR'] = str(LOGS / 'mplcache')
    # Keep the analysis self-contained on a read-only-ish checkout: no .pyc
    # next to the scripts, no user-site surprises.
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    env.pop('PYTHONPATH', None)

    results = []
    t_start = time.time()
    for i, (script, what) in enumerate(steps, 1):
        print(f'\n{"=" * 78}\n[{i}/{len(steps)}] {script}\n  {what}\n{"=" * 78}')
        rc, dt = run_step(script, LOGS, env)
        results.append({'script': script, 'rc': rc, 'seconds': round(dt, 1)})
        print(f'\n[{i}/{len(steps)}] {script}: {"OK" if rc == 0 else f"FAILED (rc={rc})"} '
              f'in {dt:.1f}s')
        if rc != 0:
            print(f'  full output: logs/{script[:-3]}.log')
            break

    manifest = {
        'schema': 1,
        # Basename only: the interpreter's absolute location is a host path and
        # this file travels with the analysis.
        'python': Path(sys.executable).name,
        'python_version': sys.version.split()[0],
        'figures_built': bool(args.figures),
        'total_seconds': round(time.time() - t_start, 1),
        'steps': results,
        'inputs': {e['name']: e['sha256'] for e in cfg['inputs']},
        'outputs': {},
        'figures': {},
    }
    for name in cfg.get('outputs', []):
        p = GEN / name
        if p.exists():
            manifest['outputs'][name] = sha256(p)
    for name in cfg.get('figures', []):
        p = GEN / name
        if p.exists():
            manifest['figures'][name] = sha256(p)
    (LOGS / 'run_manifest.json').write_text(
        redact(json.dumps(manifest, indent=1)) + '\n')

    failed = [r for r in results if r['rc'] != 0]
    print(f'\n{"=" * 78}')
    if failed:
        print(f'FAILED: {failed[0]["script"]} (rc={failed[0]["rc"]}); '
              f'stopped after {len(results)}/{len(steps)} steps')
        print(f'  logs: {LOGS}')
        return 1
    print(f'all {len(results)} steps OK in {manifest["total_seconds"]:.0f}s; '
          f'{len(manifest["outputs"])} outputs, {len(manifest["figures"])} figures')
    print(f'  manifest: logs/run_manifest.json')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
