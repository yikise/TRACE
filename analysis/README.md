# TRACE paper analysis

Recompute the final paper's Tables 1–4, headline statistics, and optional Figure 1 from frozen model scores. This directory runs entirely on CPU; it does not load model weights or compile LaTeX.

## Run

From the repository root:

```bash
python analysis/run_all.py --check
python analysis/run_all.py
python analysis/run_all.py --figures
```

Install NumPy for statistics; Matplotlib is needed only for the optional figure. See `requirements/analysis.txt` at repository root. Inputs are verified against `config/inputs.json`, staged from `inputs/frozen/` into `generated/`, and processed in dependency order. Outputs are JSON statistics and LaTeX table/macro fragments under `generated/`; execution records are in `logs/`.

`--only` runs the named steps, **not their dependencies**. Use the full pipeline on a fresh checkout.

## Paper-to-code map

| Paper element | Implementation |
| --- | --- |
| Table 1: overlapping counterfactual checks | `scripts/make_checks_table.py` |
| Table 2: eight-model decomposition | `scripts/build_attribution_numbers.py`, formatting in `scripts/make_contrast_table.py` |
| Table 3: category quintiles and target substitution | `scripts/build_category_structure.py`, `scripts/build_subst_numbers.py`, `scripts/make_risk_table.py` |
| Table 4: paired acoustic/training contrasts | `scripts/build_attribution_numbers.py`, `scripts/build_revision_numbers.py`, `scripts/make_contrast_table.py` |
| Re-paired backgrounds | `scripts/build_repair_numbers.py`, `scripts/build_utility_numbers.py` |
| Matched absent queries | `scripts/build_matched_numbers.py` |
| Paired answer mappings | `scripts/build_counterfactual_interface.py` |
| Near-tie and threshold checks | `scripts/build_robust_numbers.py`, `scripts/build_revision_numbers.py` |
| Oracle, shortcut, RMS-sum probes | `scripts/build_probe_numbers.py` |
| Figure 1 | `scripts/make_overview_figure.py` |

## Metric and compatibility

The final paper uses `H = min(m_full, min(m_rivals)) - m_target_removed`.
A ranking failure is `m_full > 0 and H <= 0`. `H > 0` certifies separation of this local view set by a shared threshold, not source identification or causal grounding.

Some inherited scripts additionally compute `G = Δt - max_j |Δj|` and split ordering-consistent answers into legacy `context` / `grounded` states. These are auxiliary compatibility outputs; **G is not TRACE's failure criterion**, and the label `grounded` is not a proof of grounding. References to an ambiguous printed equation inside legacy diagnostic outputs concern an earlier manuscript, not Equation (3) in the final TRACE paper.

The primary panel and paper-facing macros use exactly eight models. Certain historical inputs and diagnostic scripts retain two additional `ke_omni_r_*` models to preserve their original ten-model diagnostic calculations. They are **not** included in Table 2 or the headline denominator. Consult paper-facing `HDesign*` and `MatchPaper*` fields instead of pooling all historical diagnostic rows.

## Frozen inputs and raw evidence

`config/inputs.json` records both source-file digests and digests after path sanitization. Paths embedded in frozen analysis inputs are provenance labels, not filesystem dependencies. No score, label, condition identifier or audio hash was changed when paths were sanitized.

The repository also includes independently auditable per-view scores in `../data/scores/`, manifests in `../data/manifests/`, and source audio in `../data/audio_sources/`. Verify the raw-score headline with:

```bash
python scripts/audit_scores.py
```

The analysis subprocesses consume frozen exports rather than regenerating them automatically from raw scores. This intentionally separates exact historical analysis reproduction from score export. `scripts/verify_release.py` at repository root checks that the overlapping exported margins equal the released raw margins.

Target-substitution analysis uses `subst_aggregates.json` (per-model/tuple/quintile sufficient statistics). The full repository additionally supplies `../data/design/subst_units.json`, the raw substitution scores and design, plus `../scripts/aggregate_subst.py` to recompute these aggregates and check them against the frozen analysis input. This closes the aggregate-only boundary of this subdirectory.

## Expected results

- Main: 27,648 conditions; 18,911 correct; 1,184 failures = 6.3% rounded, with 503 competitor-confused and 681 ungrounded cases.
- Source-tuple bootstrap interval for failure share: 4.9–7.8%; 10,000 draws, main analysis seed 20260907.
- SNR comparison on 5,466 common-correct units: 4.9% to 8.4% (+3.5 percentage points).
- Category Q1/Q5: 22.3% / 0.8%; target-replaced gap +15.7 points.
- Synthetic probes: oracle 0.0%, shortcut 100.0%, RMS-sum proxy 45.0%.
- Matched present/absent questions: 56.6% true-positive rate, 25.0% false-positive rate.

## Reproduction limits

1. This pipeline recomputes statistics from existing scores; it is not a new eight-model inference run. Hardware, model-weight revisions and GPU kernels can affect new scores.
2. Synthetic probe point estimates are recomputed. The historical `probe_boot.json` bootstrap producer was not recovered, so its extra intervals and unused theorem/sweep macros are not claimed as reproduced. The final paper's displayed 0/100/45.0 point estimates are covered.
3. `counterfactual_inputs.json` includes legacy phase-scrambled scores used by auxiliary scripts. That supplementary audio panel is not part of the seven released reconstruction panels; the final paper's deletion/RMS controls are included.
4. Figure PDFs can differ bytewise because of embedded generation timestamps. Visual content and numerical anchors can be checked separately; identical PNG bytes across different Matplotlib versions/platforms are not guaranteed.
5. Raw listener response sheets are not provided. Audio provenance and deterministic reconstruction do not replace independent listening verification.
6. Figure example WAVs and all other derived audio remain subject to the ESC-50 noncommercial terms in `../data/ESC50-LICENSE.txt`.

The original paper PDF, manuscript body, reviews and internal research notes are deliberately not included. Tables are LaTeX fragments, not standalone compilable documents.
