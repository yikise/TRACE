#!/usr/bin/env python3
"""Rebuild TRACE panel audio from the frozen manifests and bundled source clips.

Reconstruction is bit-exact with the frozen renderers
(``experiments/qsaec/render_stress.py``, ``design_repair_panel.py``,
``design_subst_panel.py``). Every generated view is checked against the
``audio_sha256`` recorded in the frozen manifest and the run aborts on the first
mismatch.

Rules that are reproduced exactly:

* clips are read as mono PCM16 and divided by ``32767``;
* a tuple contributes its three 2.0 s centre crops, with the stored
  ``common_constituent_gain`` divided back out;
* a view accumulates ``gain * source`` into a 4.0 s buffer at the recorded
  onsets, iterating ``included_indices`` in the recorded order;
* the target keeps ``target_gain`` while every distractor shares
  ``distractor_gain``;
* ``baseline``, ``repair`` and ``subst`` apply no level correction, exactly as
  the frozen renderers do; ``global_rms`` multiplies by the recorded
  ``normalization_gain`` and then by the recorded ``shared_safety_gain``;
* ``subst`` replaces the target event with the substitution clip scaled by the
  ``rms_scale`` stored in ``design/subst_design.jsonl``;
* ``fixed``, ``prompt`` and ``absent`` reuse the frozen query-swap views that
  already ship under ``data/audio_sources/fixed/`` instead of re-deriving them.

Output layout under ``--output-dir``::

    <output-dir>/<panel>/stress_manifest.jsonl
    <output-dir>/<panel>/FROZEN.json
    <output-dir>/<panel>/<view audio paths, relative to that manifest>

Each panel gets its own subdirectory because some panels share condition ids
(``baseline`` and ``global_rms`` have identical condition ids). ``run_stress.py``
resolves a view as ``manifest.parent / view["audio"]``, which is exactly this
layout, so the emitted manifest is directly scorable::

    python -m experiments.qsaec.run_stress \\
        --manifest <output-dir>/baseline/stress_manifest.jsonl --model ... --output ...

A non-empty panel directory is never overwritten. Use ``--check-only`` to
hash-verify without writing anything, or ``--verify-existing`` to re-verify an
existing tree in place.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import sys
import wave
from pathlib import Path
from typing import Any, Sequence

import numpy as np

PANELS = ("baseline", "global_rms", "repair", "subst", "absent", "prompt", "fixed")
SYNTHESIS_PANELS = ("baseline", "global_rms", "repair", "subst")
COPY_PANELS = ("absent", "prompt", "fixed")
STEM_BITMASKS = (1, 2, 4)
PACKAGE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_ROOT = PACKAGE_ROOT / "data"


class VerificationError(RuntimeError):
    """Raised when a rebuild does not reproduce a frozen digest."""


def _read_wav(path: Path) -> tuple[int, list[float]]:
    """Read mono PCM16 exactly like the frozen renderer (int16 -> float via /32767)."""
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        width = handle.getsampwidth()
        rate = handle.getframerate()
        raw = handle.readframes(handle.getnframes())
    if channels != 1 or width != 2:
        raise ValueError(f"expected mono PCM16 crop: {path}")
    return rate, (np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32767.0).tolist()


def _wav_bytes(samples: Any, rate: int) -> bytes:
    """Deterministic mono PCM16 bytes, identical to the frozen ``_write_wav``."""
    raw = np.rint(np.clip(np.asarray(samples), -1.0, 1.0) * 32767).astype("<i2").tobytes()
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(raw)
    return buffer.getvalue()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def onsets_for(row: dict, rate: int) -> dict[int, int]:
    """Frozen onset rule: full overlap starts every included source at 1.0 s."""
    included = [int(i) for i in row["included_indices"]]
    target_index = int(row["target_index"])
    if row["overlap"] == "full":
        return {index: int(1.0 * rate) for index in included}
    signs = (-1, 1) if row["y_present"] == "A" else (1, -1)
    out = {target_index: int(1.0 * rate)}
    distractors = [index for index in included if index != target_index]
    for position, index in enumerate(distractors):
        out[index] = int((1.0 + 0.75 * signs[position]) * rate)
    return out


def gains_for(row: dict) -> dict[int, float]:
    """Target keeps ``target_gain``; every distractor shares ``distractor_gain``."""
    target_index = int(row["target_index"])
    out = {target_index: float(row["target_gain"])}
    for index in row["included_indices"]:
        index = int(index)
        if index != target_index:
            out[index] = float(row["distractor_gain"])
    return out


def mix_view(
    events: Sequence[Sequence[float]],
    included: Sequence[int],
    gains: dict[int, float],
    onsets: dict[int, int],
    frames: int,
    omitted: int | None,
) -> np.ndarray:
    """Accumulate sources into a 4.0 s buffer in the frozen iteration order.

    ``omitted`` is the recorded ``removed_index``; that source contributes
    nothing, which is how every frozen removal view is defined.
    """
    buffer = np.zeros(frames, dtype=np.float64)
    for index in included:
        if index == omitted:
            continue
        values = np.asarray(events[index], dtype=np.float64)
        onset = onsets[index]
        buffer[onset:onset + len(values)] += gains[index] * values
    return buffer


class SourcePool:
    """Reads, verifies and caches the bundled source clips."""

    def __init__(self, data_root: Path) -> None:
        self.data_root = data_root
        self.audio_sources = data_root / "audio_sources"
        index_path = self.audio_sources / "INDEX.jsonl"
        if not index_path.is_file():
            raise FileNotFoundError(f"missing source index: {index_path}")
        self.index: dict[str, dict] = {}
        for line in index_path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                self.index[row["ref"]] = row
        self.rate: int | None = None
        self._stems: dict[tuple[str, int], list[float]] = {}
        self._clips: dict[str, list[float]] = {}

    def ref_path(self, ref: str) -> Path:
        path = self.data_root / ref
        if not path.is_file():
            raise FileNotFoundError(f"missing bundled source clip: {path}")
        return path

    def clip(self, ref: str) -> list[float]:
        if ref not in self._clips:
            rate, audio = _read_wav(self.ref_path(ref))
            self._set_rate(rate)
            self._clips[ref] = audio
        return self._clips[ref]

    def stem(self, tuple_id: str, event_index: int) -> list[float]:
        """One 2.0 s centre crop of a tuple event, with the common gain removed."""
        key = (tuple_id, event_index)
        if key not in self._stems:
            bit = STEM_BITMASKS[event_index]
            ref = f"audio_sources/stems/{tuple_id}/subset_{bit:03b}.wav"
            rate, audio = _read_wav(self.ref_path(ref))
            self._set_rate(rate)
            crop = int(2.0 * rate)
            start = (len(audio) - crop) // 2
            gain = float(self.index[ref]["common_constituent_gain"])
            self._stems[key] = [value / gain for value in audio[start:start + crop]]
        return self._stems[key]

    def stems(self, tuple_id: str) -> list[list[float]]:
        return [self.stem(tuple_id, index) for index in range(3)]

    def _set_rate(self, rate: int) -> None:
        if self.rate is None:
            self.rate = rate
        elif self.rate != rate:
            raise ValueError(f"sample-rate mismatch: {rate} != {self.rate}")


def load_design(data_root: Path, name: str) -> dict[str, dict]:
    path = data_root / "design" / name
    if not path.is_file():
        raise FileNotFoundError(f"missing design file: {path}")
    return {
        row["rid"]: row
        for row in (json.loads(line) for line in path.read_text().splitlines() if line.strip())
    }


def render_row(panel: str, row: dict, pool: SourcePool, subst_design: dict[str, dict]) -> list[tuple[dict, bytes]]:
    """Return ``(view, wav_bytes)`` for every view of a manifest row."""
    if panel in COPY_PANELS:
        out = []
        for view in row["views"]:
            rel = view["audio"]
            ref = f"audio_sources/{rel}"
            out.append((view, pool.ref_path(ref).read_bytes()))
        return out

    target_index = int(row["target_index"])
    included = [int(i) for i in row["included_indices"]]

    # Reading the sources is what establishes the sample rate.
    if panel == "subst":
        design = subst_design.get(row["condition_id"].rsplit("_snr", 1)[0])
        if design is None:
            raise KeyError(f"no subst design row for {row['condition_id']}")
        events = pool.stems(row["source_tuple_id"])
        scale = float(design["rms_scale"])
        events[target_index] = [value * scale for value in pool.clip(design["substitute_ref"])]
    elif panel == "repair":
        target = row["repair_target"]
        events = [pool.stem(target[0], int(target[1]))]
        events += [pool.stem(d[0], int(d[1])) for d in row["repair_distractors"]]
    else:
        events = pool.stems(row["source_tuple_id"])

    rate = pool.rate
    if rate is None:
        raise RuntimeError("source pool has no sample rate")
    onsets = onsets_for(row, rate)
    gains = gains_for(row)

    frames = int(4.0 * rate)
    out = []
    for view in row["views"]:
        omitted = view.get("removed_index")
        mixed = mix_view(
            events, included, gains, onsets, frames,
            None if omitted is None else int(omitted),
        )
        # The frozen renderers record both corrections per view. They are 1.0
        # everywhere except ``global_rms`` (normalization + peak safety) and the
        # ``repair`` panel (peak safety only); skipping the multiply when the
        # factor is exactly 1.0 keeps the float bits identical.
        normalization = float(view.get("normalization_gain", 1.0))
        if normalization != 1.0:
            mixed = mixed * normalization
        safety = float(view.get("shared_safety_gain", 1.0))
        if safety != 1.0:
            mixed = mixed * safety
        out.append((view, _wav_bytes(mixed, rate)))
    return out


def _check(panel: str, rel: str, payload: bytes, expected: str) -> None:
    got = hashlib.sha256(payload).hexdigest()
    if got != expected:
        raise VerificationError(
            f"{panel}: frozen digest mismatch for {rel}\n  computed {got}\n  frozen   {expected}"
        )


def _prepare_panel_dir(target: Path, *, verify_existing: bool) -> None:
    if not target.exists():
        return
    if not target.is_dir():
        raise SystemExit(f"refusing to write: not a directory: {target}")
    if any(target.iterdir()) and not verify_existing:
        raise SystemExit(
            f"refusing to overwrite non-empty target: {target}\n"
            "pass --verify-existing to re-verify it in place, "
            "or choose a different --output-dir"
        )


def _select_rows(rows: list[dict], condition_ids: Sequence[str], limit: int | None) -> list[dict]:
    if condition_ids:
        by_id = {row["condition_id"]: row for row in rows}
        missing = [cid for cid in condition_ids if cid not in by_id]
        if missing:
            raise SystemExit(f"unknown condition ids: {missing[:5]}")
        return [by_id[cid] for cid in condition_ids]
    return rows[:limit] if limit else rows


def render_panel(
    panel: str,
    rows: list[dict],
    pool: SourcePool,
    output_dir: Path | None,
    *,
    check_only: bool,
    verify_existing: bool,
) -> dict:
    subst_design = load_design(pool.data_root, "subst_design.jsonl") if panel == "subst" else {}
    panel_root = (output_dir / panel) if output_dir else None
    if panel_root is not None and not check_only:
        _prepare_panel_dir(panel_root, verify_existing=verify_existing)

    views_done = 0
    views_written = 0
    emitted: list[dict] = []

    for row in rows:
        new_views = []
        for view, payload in render_row(panel, row, pool, subst_design):
            rel = view["audio"]
            _check(panel, rel, payload, view["audio_sha256"])
            views_done += 1
            item = dict(view)
            item["audio"] = rel
            new_views.append(item)
            if check_only or panel_root is None:
                continue
            dest = panel_root / rel
            if dest.is_file():
                if _sha256_file(dest) != view["audio_sha256"]:
                    raise VerificationError(f"{panel}: existing file differs from frozen digest: {dest}")
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(payload)
            views_written += 1
        new_row = dict(row)
        new_row["views"] = new_views
        emitted.append(new_row)

    if panel_root is not None and not check_only:
        payload = "".join(json.dumps(row, sort_keys=True) + "\n" for row in emitted)
        manifest_path = panel_root / "stress_manifest.jsonl"
        if manifest_path.is_file():
            if manifest_path.read_text() != payload:
                raise VerificationError(f"{panel}: existing stress_manifest.jsonl differs")
        else:
            manifest_path.write_text(payload)
        (panel_root / "FROZEN.json").write_text(json.dumps({
            "panel": panel,
            "conditions": len(emitted),
            "views": views_done,
            "manifest_sha256": _sha256_file(manifest_path),
            "source_data_root": str(pool.data_root),
            "reconstruction": (
                "rebuilt from bundled source clips" if panel in SYNTHESIS_PANELS
                else "reused frozen query-swap views"
            ),
        }, indent=2, sort_keys=True) + "\n")

    return {
        "panel": panel,
        "conditions": len(emitted),
        "expected_views": sum(len(row["views"]) for row in rows),
        "views_verified": views_done,
        "views_written": views_written,
        "check_only": check_only,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Rebuild TRACE panel audio from frozen manifests.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Output layout")[0],
    )
    parser.add_argument("--panel", action="append", choices=PANELS,
                        help="panel to build or verify; repeat for several (default: all)")
    parser.add_argument("--output-dir", help="destination root; each panel gets its own subdirectory")
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT),
                        help="package data root holding manifests/ and audio_sources/")
    parser.add_argument("--limit", type=int, help="process only the first N conditions of each panel")
    parser.add_argument("--condition-id", action="append", default=[],
                        help="process only these condition ids (repeatable; takes precedence over --limit)")
    parser.add_argument("--check-only", action="store_true",
                        help="render and hash-verify in memory without writing anything")
    parser.add_argument("--verify-existing", action="store_true",
                        help="idempotently verify an existing output tree instead of refusing to overwrite")
    args = parser.parse_args(argv)

    panels = list(dict.fromkeys(args.panel or PANELS))
    if not args.check_only and not args.output_dir:
        parser.error("--output-dir is required unless --check-only is given")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")

    data_root = Path(args.data_root).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else None
    pool = SourcePool(data_root)

    reports = []
    for panel in panels:
        manifest = data_root / "manifests" / f"{panel}.jsonl"
        if not manifest.is_file():
            raise SystemExit(f"missing frozen manifest: {manifest}")
        rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
        selected = _select_rows(rows, args.condition_id, args.limit)
        if not selected:
            raise SystemExit(f"{panel}: no conditions selected")
        report = render_panel(
            panel, selected, pool, output_dir,
            check_only=args.check_only, verify_existing=args.verify_existing,
        )
        report["status"] = "verified" if report["views_verified"] == report["expected_views"] else "INCOMPLETE"
        reports.append(report)
        print(
            f"{panel:11s} conditions={report['conditions']:5d} "
            f"views={report['views_verified']:5d}/{report['expected_views']:5d} "
            f"written={report['views_written']:5d} sha256={report['status']}",
            flush=True,
        )
        if report["status"] != "verified":
            raise VerificationError(f"{panel}: verified {report['views_verified']} of {report['expected_views']} views")

    summary = {
        "status": "ok",
        "mode": "check-only" if args.check_only else "render",
        "data_root": str(data_root),
        "output_dir": str(output_dir) if output_dir else None,
        "panels": reports,
        "views_total": sum(r["views_verified"] for r in reports),
    }
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except VerificationError as exc:
        print(f"VERIFICATION FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1)
