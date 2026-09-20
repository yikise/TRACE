"""Score a frozen source-factorial QSAEC stress manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterator, Sequence

from experiments.pri.diagnostic.adapters import (
    AudioFlamingo3Adapter,
    AudioFlamingo3HFAdapter,
    MiMoAudioAdapter,
    Qwen25OmniAdapter,
    Qwen2AudioAdapter,
    Qwen3OmniAdapter,
    StepAudio2Adapter,
    call_scoring_adapter,
)
from experiments.pri.diagnostic.models import ViewRecord


def score(adapter: Any, manifest: str | Path, *, shard_index: int, num_shards: int, model_id: str) -> Iterator[dict]:
    manifest_path = Path(manifest).resolve()
    conditions = [json.loads(line) for line in manifest_path.read_text().splitlines() if line]
    conditions = [row for index, row in enumerate(conditions) if index % num_shards == shard_index]
    for condition in conditions:
        labels = (condition["y_present"], condition["y_absent"])
        for view in condition["views"]:
            record = ViewRecord("x_f", str(manifest_path.parent / view["audio"]), question=condition["question"])
            raw = call_scoring_adapter(adapter, record, labels, question=condition["question"])
            optional = {
                key: condition[key]
                for key in (
                    "prompt_template",
                    "label_mapping",
                    "energy_variant",
                    "semantic_answer_present",
                    "absent_category",
                    "selection_seed",
                    "audit_status",
                )
                if key in condition
            }
            view_optional = {
                key: view[key]
                for key in (
                    "audio_sha256",
                    "intervention_semantics",
                    "pre_rms",
                    "post_rms",
                    "pre_peak",
                    "post_peak",
                    "normalization_gain",
                    "shared_safety_gain",
                    "surrogate_seed",
                )
                if key in view
            }
            yield {
                **{key: condition[key] for key in (
                    "condition_id", "source_tuple_id", "question_id", "target_index", "target_category",
                    "snr_db", "overlap", "distractor_count", "distractor_replicate",
                )},
                **optional,
                **view_optional,
                "model_id": model_id,
                "intervention": view["intervention"],
                "removed_index": view["removed_index"],
                "audio": view["audio"],
                "y_present": labels[0],
                "y_absent": labels[1],
                "logp_present": raw[labels[0]],
                "logp_absent": raw[labels[1]],
                "margin": raw[labels[0]] - raw[labels[1]],
            }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument(
        "--adapter",
        choices=("qwen", "qwen2_audio", "qwen3_omni", "mimo", "mimo_audio", "step_audio2", "step_audio", "step", "af3", "audio_flamingo_3", "af3_hf", "audio_flamingo_3_hf"),
        default="qwen",
        help="scoring backend; mimo/mimo_audio use MiMo-Audio; step_audio2/step_audio/step use Step-Audio-2; af3/audio_flamingo_3 use the official llava Audio Flamingo 3 runtime; af3_hf/audio_flamingo_3_hf use the Hugging Face AF3 runtime",
    )
    parser.add_argument("--lora-path")
    parser.add_argument(
        "--tokenizer-path",
        help="MiMo-Audio-Tokenizer checkpoint directory (required for --adapter mimo)",
    )
    parser.add_argument(
        "--af3-repo",
        help="Audio Flamingo 3 repository root containing the official llava package",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args(argv)
    if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
        parser.error("require 0 <= shard-index < num-shards")
    if args.adapter == "qwen":
        adapter = Qwen25OmniAdapter(
            args.model, device=args.device, dtype=args.dtype, lora_path=args.lora_path
        )
    elif args.adapter == "qwen2_audio":
        if args.lora_path:
            parser.error("--lora-path is only supported by the qwen adapter")
        adapter = Qwen2AudioAdapter(args.model, device=args.device, dtype=args.dtype)
    elif args.adapter == "qwen3_omni":
        if args.lora_path:
            parser.error("--lora-path is not supported by the qwen3_omni adapter")
        adapter = Qwen3OmniAdapter(args.model, device=args.device, dtype=args.dtype)
    elif args.adapter in {"mimo", "mimo_audio"}:
        if args.lora_path:
            parser.error("--lora-path is not supported by the MiMo-Audio adapter")
        if not args.tokenizer_path:
            parser.error("--tokenizer-path is required by the MiMo-Audio adapter")
        adapter = MiMoAudioAdapter(
            args.model,
            tokenizer_path=args.tokenizer_path,
            device=args.device,
            dtype=args.dtype,
        )
    elif args.adapter in {"af3", "audio_flamingo_3"}:
        if args.lora_path:
            parser.error("--lora-path is not supported by the Audio Flamingo 3 adapter")
        if args.tokenizer_path:
            parser.error("--tokenizer-path is only supported by the MiMo-Audio adapter")
        adapter = AudioFlamingo3Adapter(
            args.model,
            repo_path=args.af3_repo,
            device=args.device,
            dtype=args.dtype,
        )
    elif args.adapter in {"af3_hf", "audio_flamingo_3_hf"}:
        if args.lora_path:
            parser.error("--lora-path is not supported by the Hugging Face Audio Flamingo 3 adapter")
        if args.tokenizer_path:
            parser.error("--tokenizer-path is only supported by the MiMo-Audio adapter")
        if args.af3_repo:
            parser.error("--af3-repo is only supported by the official llava Audio Flamingo 3 adapter")
        adapter = AudioFlamingo3HFAdapter(
            args.model,
            device=args.device,
            dtype=args.dtype,
        )
    else:
        if args.lora_path:
            parser.error("--lora-path is not supported by the Step-Audio-2 adapter")
        adapter = StepAudio2Adapter(args.model, device=args.device, dtype=args.dtype)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    cells = 0
    conditions = set()
    with output.open("w") as handle:
        for row in score(adapter, args.manifest, shard_index=args.shard_index, num_shards=args.num_shards, model_id=args.model_id):
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            handle.flush()
            cells += 1
            conditions.add(row["condition_id"])
    print(json.dumps({"cells": cells, "conditions": len(conditions), "output": str(output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
