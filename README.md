# TRACE

**Right Answers, Inconsistent Evidence: Source-Level Counterfactual Auditing of Audio-Language Models**

Yi Zhang, Yi Li, Hongwei Du

TRACE audits whether an audio-language model consistently ranks target-present audio above a source-matched target-absent view—even when its original answer is correct.

> Project-authored code, metadata and scores are released under the MIT license: see [LICENSE](LICENSE). ESC-50-derived audio keeps its upstream noncommercial terms and is **not** covered by MIT; see [Licensing](#licensing). No model weights are included.

## Main result

Across eight models and 3,456 mixture conditions per model:

- **18,911** correct mixture answers out of **27,648** evaluations.
- **1,184 / 18,911 = 6.3%** ranking failures (rounded; 95% source-tuple bootstrap CI: 4.9–7.8%).
- **681** ungrounded and **503** competitor-confused answers.
- A target-only score check misses **42.5%** of these failures.
- On 5,466 common-correct units, failures increase from **4.9% to 8.4%** as target SNR falls from +5 to −5 dB.

Let `m_full` be the present-vs-absent margin on the original mixture, `m_target_removed` the margin after deleting the queried source, and `m_rivals` the margins after deleting each competing source:

```text
H = min(m_full, min(m_rivals)) - m_target_removed
correct mixture answer: m_full > 0
ranking failure:        m_full > 0 and H <= 0
```

`H > 0` means one shared local threshold can correctly label this view set. It does **not** prove that the model identifies the source as a listener would. Legacy code identifiers such as QSAEC/Q-HEAR are preserved only for provenance; the final metric is H, not the older absolute-effect criterion G.

## Quick start: reproduce without GPUs

Python 3.10+ is required. Commands below run from the repository root.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements/analysis.txt

# Validate 51 raw-score files and reproduce the exact main counts.
python scripts/audit_scores.py

# Run metric boundary tests.
python -m unittest discover -s tests -v

# Validate checksums, source/analysis consistency, syntax and release hygiene.
python scripts/verify_release.py

# Recompute the target-substitution sufficient statistics from 23,040 paired units.
python scripts/aggregate_subst.py

# Recompute Tables 1–4 and all included paper-facing statistics.
python analysis/run_all.py

# Also rebuild Figure 1 (Matplotlib, no LaTeX installation needed).
python analysis/run_all.py --figures
```

Results are written to `analysis/generated/`; step logs are written to `analysis/logs/`. These are generated working directories and are excluded from Git. The scripts use local frozen inputs and do not access the original experiment server.

## Repository layout

```text
TRACE/
  README.md
  LICENSE                     MIT, for project-authored code and results
  LICENSE-DATA.md             data/audio scope note (ESC-50 terms, not MIT)
  THIRD_PARTY_NOTICES.md
  SHA256SUMS                  checksums for public source/data files
  requirements/               analysis and separate model-runtime requirements
  configs/                    eight-model registry, config hashes, source provenance
  experiments/
    qsaec/run_stress.py        original frozen inference entry point
    pri/diagnostic/            model adapters and view types
  scripts/
    render_audio.py            rebuild or verify the seven audio panels
    audit_scores.py            raw-score validation and headline counts
    aggregate_subst.py         paired-substitution aggregation
    verify_release.py          release and raw-to-analysis checks
  data/
    manifests/                seven frozen manifests with portable references
    audio_sources/            608 small source/fixed WAVs and attribution index
    design/                   background and target substitution designs/paired units
    scores/                   51 compressed per-view score files, eight paper models
    ESC50-LICENSE.txt          upstream license and per-recording attribution
    ESC50-meta.csv
  analysis/
    inputs/frozen/            exact analysis inputs, with host paths sanitized
    scripts/                  final manuscript statistics/table/figure generators
    run_all.py                ordered CPU reproduction pipeline
```

The release deliberately excludes model weights, runtime environments, GPU queue machinery, credentials, internal hostnames, manuscript/review notes, redundant experiment branches and the multi-gigabyte fully rendered audio trees.

## Dataset and evaluation panels

All audio is derived from [ESC-50](https://github.com/karoldvl/ESC-50), not a newly collected independent corpus. The main panel uses 192 source recordings spanning 49 categories, arranged into 64 three-source tuples and 192 queried target roles.

Per target role, there are three SNR levels, two overlap settings and three distractor selections: rival 1 alone, rival 2 alone, or both. Thus `192 × 3 × 2 × 3 = 3,456`, not a fully crossed distractor-count-by-replicate design.

| Panel | Conditions/model | Views/model | Models | Purpose |
| --- | ---: | ---: | ---: | --- |
| `baseline` | 3,456 | 11,520 | 8 | Main source-removal audit |
| `global_rms` | 3,456 | 11,520 | 8 | Loudness-matched removal control |
| `repair` | 2,880 | 11,520 | 3 | Five re-pairings of each of 192 target roles |
| `subst` | 2,880 | 9,600 | 8 | 160 replacement recordings over 56 Q1/Q5 roles |
| `absent` | 512 | 512 | 8 | Four wordings × two mappings on 64 absent-event mixtures |
| `prompt` | 1,536 | 6,144 | 8 | Paired prompt/mapping checks |
| `fixed` | 192 | 768 | 8 | Fixed-mixture query-swap reference |

There are **51,584 view entries across the seven audio panels** and **355,072 released model-view scores**. A view entry is not necessarily a unique waveform; fixed/prompt/absent panels reuse audio.

The three re-pairing models are Qwen2.5-Omni-7B, Qwen3-Omni-30B and Step-Audio 2 Mini. Target substitution covers 42 original tuples and 19 categories, rather than every main-panel tuple/category.

`data/audio_sources/` contains 192 single-stem files, 160 substitution clips and 256 reusable fixed-mixture views. `INDEX.jsonl` records SHA-256 digests and ESC-50 source attribution. Rendered main/stress mixtures are 4 s; source mixing uses 44.1 kHz PCM16, while adapters handle model input resampling. Source gains, offsets and clipping/normalization factors are frozen in the manifests.

See [data/README.md](data/README.md) for schemas and provenance.

## Rebuild audio

```bash
# Smoke test all seven panels in memory.
python scripts/render_audio.py --check-only --limit 4

# Full bit-exact verification without writing generated audio.
python scripts/render_audio.py --check-only

# Render baseline for inference.
python scripts/render_audio.py --panel baseline --output-dir rendered
```

The final command creates `rendered/baseline/stress_manifest.jsonl` and its audio files. Every view must match its frozen SHA-256 digest. Nonempty panel output directories are refused; `--verify-existing` explicitly requests verification/resumption instead of silent replacement. Full rendering requires substantially more storage than the small released source pool.

## Re-run model inference

The adapters are copied from the actual frozen execution version; checkpoints must be obtained separately under their upstream terms. Use **separate environments** for core models, Audio Flamingo 3, and MiMo. Do not install all Transformers requirement files together.

| Paper model | Adapter argument | Runtime |
| --- | --- | --- |
| Qwen2-Audio-7B | `qwen2_audio` | core |
| Qwen2.5-Omni-3B / 7B | `qwen` | core |
| Qwen2.5-Omni-7B + AHA | `qwen` + `--lora-path` | core + PEFT |
| Qwen3-Omni-30B | `qwen3_omni` | core |
| Step-Audio 2 Mini | `step_audio2` | core |
| MiMo-Audio-7B | `mimo_audio` + `--tokenizer-path` | MiMo upstream source |
| Audio Flamingo 3 | `audio_flamingo_3_hf` | AF3 HF |

Example, after installing PyTorch compatible with your CUDA driver and `requirements/core.txt` in a dedicated environment:

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.qsaec.run_stress \
  --manifest rendered/baseline/stress_manifest.jsonl \
  --model /path/to/Qwen2-Audio-7B-Instruct \
  --model-id qwen2_audio_7b --adapter qwen2_audio \
  --device cuda --dtype bfloat16 \
  --output outputs/qwen2_audio_7b.jsonl
```

Use `--shard-index 0 --num-shards 8` for one of eight **condition-level** shards; run indices 0–7 with distinct output paths. The direct inference entry point writes a fresh output file: use a new path to avoid replacing previous scores. A/B labels stay fixed within each query's deletion views. Scores come from answer-token probabilities, not parsed free-form generations.

Historical core/AF3 environments used PyTorch `2.10.0+cu129`, Transformers `4.57.6` / `5.0.0rc1`, respectively. MiMo used Transformers `4.49.0` and upstream source commit `691ce54144a6844cc641fd96046a6ba20776c8b0`; expose that checkout on `PYTHONPATH` and provide its separate tokenizer checkpoint. `requirements/mimo.txt` records the recovered Transformers pin, **not a complete environment lock**; follow the pinned upstream source requirements. PEFT's exact historical version and complete model weight revisions were not recovered. `configs/models.json` retains configuration hashes, which are not hashes of the complete weights.

**No GPU inference was rerun during release preparation.** CPU score audits, statistics, argument parsing and audio reconstruction were tested; installing a fresh GPU environment and rescoring all eight models remains separate validation.

## Verification and known limitations

- Raw release: all **51 files / 355,072 views** validated for complete condition coverage, duplicate keys, finite values, label mapping and available audio hashes.
- Main counts: exact match to **18,911 / 1,184 / 503 / 681**.
- All seven audio panels: **51,584 / 51,584** view hashes reproduced; fixed/prompt/absent use released frozen audio rather than a new query-swap renderer.
- Tables 1–4 and included statistics: CPU pipeline tested independently inside this repository.
- Target substitution: **23,040** paired units reproduce **408** aggregate cells and **56** roles exactly.
- Analysis includes some historical ten-model diagnostic inputs and phase-scrambled scores for compatibility. They are not part of the eight-model headline or the seven-panel waveform reconstruction release. See [analysis/README.md](analysis/README.md).
- Synthetic-probe point estimates are reproduced; the historical probe-bootstrap producer was not recovered. Its unused extra confidence intervals/theorem macros are not claimed as reproduced.
- Original per-clip listener response sheets and numeric confidence values were not recovered as completed records. Existing project attestations are not a substitute for releasing raw listener annotations; the repository does **not** claim to provide them. Blind-review identity mappings are necessarily exposed by source attribution; do not show them to new blinded annotators.
- Checkpoint/config provenance is incomplete at the full-weight revision level, so fresh GPU outputs are not guaranteed bit-identical.

## Licensing

- **Project-authored code and results:** MIT. This covers the Python sources, configuration, manifests, design files, numeric scores, checksums and documentation that the authors wrote. See [LICENSE](LICENSE).
- **ESC-50 and derived audio:** Creative Commons Attribution-NonCommercial 3.0 Unported, with per-recording attribution retained verbatim in `data/ESC50-LICENSE.txt`. Commercial use is restricted. This release uses the full ESC-50 source pool, not just the differently licensed ESC-10 subset. MIT does **not** apply to this audio; see [LICENSE-DATA.md](LICENSE-DATA.md).
- **Models and upstream implementation packages:** not redistributed; their individual licenses and access conditions apply. This repository does not grant rights to their weights.
- No single repository-wide permissive license overrides third-party data or model restrictions.

## Citation

The citation does not assert an accepted venue or publication year. Internal run identifiers are retained as provenance, not publication metadata.

```bibtex
@misc{zhang_trace,
  title  = {Right Answers, Inconsistent Evidence: Source-Level Counterfactual Auditing of Audio-Language Models},
  author = {Zhang, Yi and Li, Yi and Du, Hongwei},
  url    = {https://github.com/yikise/TRACE}
}
```

Please also cite ESC-50 and the upstream models used in your experiments.

## Before public upload

1. Review the annotation/provenance limitations above and decide whether to add completed listener records.
2. Run the validation commands. Inspect the exact Git staging list before committing; `.gitignore` excludes generated audio, environments, checkpoints and logs.
3. This preparation does **not** create a GitHub repository, commit, or push. Publication is left to the authors.
