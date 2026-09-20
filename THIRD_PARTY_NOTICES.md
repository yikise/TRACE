# Third-party notices

## ESC-50

Source: https://github.com/karoldvl/ESC-50

Piczak, Karol J. ESC: Dataset for Environmental Sound Classification. ACM Multimedia, 2015.

The upstream archive identifies the complete dataset as Creative Commons Attribution-NonCommercial 3.0 Unported. ESC-10 has separate CC BY 3.0 terms, but this release is not restricted to ESC-10. Preserve `data/ESC50-LICENSE.txt`, including its per-clip Freesound credits, and `data/ESC50-meta.csv` with the derived audio. `data/audio_sources/INDEX.jsonl` maps distributed clips to source recording names; `data/source_provenance.json` records the archive checksum.

The three Figure 1 example WAV files in `analysis/data/` are also ESC-50-derived and carry the same data restrictions.

## Model software and weights

Qwen2-Audio, Qwen2.5-Omni, Qwen3-Omni, Step-Audio 2, MiMo-Audio, Audio Flamingo 3, and AHA are external projects. TRACE provides evaluation adapters and frozen scores, not those projects' weights. Install upstream software and obtain checkpoints separately; follow each project's terms. Some upstream assets have noncommercial or model-specific restrictions. No repository-level code license can override them.

MiMo's historical implementation commit is recorded in `configs/models.json`. The original environment specifications in `requirements/` are partial reproducibility records, not a redistribution of those packages.
