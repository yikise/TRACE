"""Model-scoring adapters used by the frozen diagnostic.

The statistical code only depends on ``ScoringAdapter.score_logprobs``.  This
keeps the diagnostic runnable on CPU with deterministic fixtures and avoids
coupling it to a particular Qwen/Ke-Omni-R trainer implementation.
"""

from __future__ import annotations

import math
import wave
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Optional

from .models import ViewRecord


LabelScores = Mapping[str, float]
ScorerCallable = Callable[[ViewRecord, Sequence[str], Optional[str]], LabelScores]


class ScoringAdapter(ABC):
    """Minimal adapter protocol for answer-only teacher-forced scoring."""

    @abstractmethod
    def score_logprobs(
        self,
        view: ViewRecord,
        labels: Sequence[str],
        *,
        question: str | None = None,
    ) -> LabelScores:
        """Return log probabilities for ``labels`` under one view.

        Implementations must score each answer label from the same question and
        answer-only context.  They must not generate a chain-of-thought prefix.
        """

    # ``score`` is intentionally a convenience alias.  Duck-typed adapters that
    # predate this class can still be used by ``call_scoring_adapter`` below.
    def score(
        self,
        view: ViewRecord,
        labels: Sequence[str],
        *,
        question: str | None = None,
    ) -> LabelScores:
        return self.score_logprobs(view, labels, question=question)


def _coerce_float(value: Any, *, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"log probability for {label!r} is not numeric: {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"log probability for {label!r} must be finite, got {result}")
    return result


def normalize_label_scores(result: Any, labels: Sequence[str]) -> dict[str, float]:
    """Normalize mapping/sequence adapter outputs and validate requested labels."""

    requested = tuple(str(label) for label in labels)
    if len(requested) == 0:
        raise ValueError("at least one answer label is required")

    if isinstance(result, Mapping):
        mapping: Mapping[Any, Any] = result
        normalized: dict[str, float] = {}
        for label in requested:
            if label in mapping:
                normalized[label] = _coerce_float(mapping[label], label=label)
                continue
            # Some tokenizers return integer token keys.  Accept an exact string
            # representation as a convenience without guessing arbitrary ids.
            if str(label) in mapping:
                normalized[label] = _coerce_float(mapping[str(label)], label=label)
                continue
            raise KeyError(f"adapter did not return a log probability for label {label!r}")
        return normalized

    if isinstance(result, Sequence) and not isinstance(result, (str, bytes, bytearray)):
        if len(result) != len(requested):
            raise ValueError(
                f"adapter returned {len(result)} scores for {len(requested)} labels"
            )
        return {
            label: _coerce_float(value, label=label) for label, value in zip(requested, result)
        }
    raise TypeError("adapter output must be a mapping or a sequence of log probabilities")


def call_scoring_adapter(
    adapter: Any,
    view: ViewRecord,
    labels: Sequence[str],
    *,
    question: str | None = None,
) -> dict[str, float]:
    """Call a concrete or duck-typed adapter with a small compatibility shim."""

    method = getattr(adapter, "score_logprobs", None)
    if method is None:
        method = getattr(adapter, "score", None)
    if method is None:
        method = getattr(adapter, "score_view", None)
    if method is None or not callable(method):
        raise TypeError(
            "scoring adapter must define score_logprobs(view, labels), score(...), or score_view(...)"
        )

    # The canonical signature has a keyword-only question.  The fallback keeps
    # simple deterministic adapters with a two-argument method easy to use.
    try:
        result = method(view, labels, question=question)
    except TypeError as exc:
        if "question" not in str(exc) and "keyword" not in str(exc):
            raise
        result = method(view, labels)
    return normalize_label_scores(result, labels)


class DeterministicMockAdapter(ScoringAdapter):
    """A deterministic adapter for tests and CPU smoke runs.

    ``scores`` may be keyed by view name (for example ``{"x_f": {"A": 0.0,
    "B": -2.0}}``), or omitted when each view payload contains a ``logprobs``
    mapping.  A callable scorer is also accepted and receives
    ``(view, labels, question)``.
    """

    def __init__(
        self,
        scores: Mapping[str, Mapping[str, float]] | None = None,
        *,
        scorer: ScorerCallable | Callable[..., LabelScores] | None = None,
    ) -> None:
        if scores is not None and scorer is not None:
            raise ValueError("provide either scores or scorer, not both")
        self._scores = {
            str(view_name): {str(label): float(value) for label, value in values.items()}
            for view_name, values in (scores or {}).items()
        }
        self._scorer = scorer

    def score_logprobs(
        self,
        view: ViewRecord,
        labels: Sequence[str],
        *,
        question: str | None = None,
    ) -> LabelScores:
        if self._scorer is not None:
            try:
                result = self._scorer(view, labels, question)
            except TypeError as exc:
                # Keep the mock friendly to ``lambda view, labels: ...`` tests.
                if "positional" not in str(exc) and "argument" not in str(exc):
                    raise
                result = self._scorer(view, labels)  # type: ignore[misc]
            return normalize_label_scores(result, labels)

        values: Mapping[str, float] | None = self._scores.get(view.name)
        if values is None and isinstance(view.payload, Mapping):
            payload_scores = view.payload.get("logprobs", view.payload.get("logp"))
            if isinstance(payload_scores, Mapping):
                values = payload_scores
        if values is None:
            raise KeyError(
                f"no deterministic scores for view {view.name!r}; provide scores or payload.logprobs"
            )
        return normalize_label_scores(values, labels)


# Short alias for callers that prefer the protocol name.
MockScoringAdapter = DeterministicMockAdapter


def _load_wav(path: str | Path) -> tuple[list[float], int]:
    """Read a PCM WAV with the stdlib for the optional HF adapter."""

    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        sample_width = handle.getsampwidth()
        sample_rate = handle.getframerate()
        frames = handle.readframes(handle.getnframes())
    if sample_width not in (1, 2, 4):
        raise ValueError(f"unsupported WAV sample width {sample_width} bytes: {path}")
    import numpy as np

    dtype = {1: np.uint8, 2: np.int16, 4: np.int32}[sample_width]
    data = np.frombuffer(frames, dtype=dtype).astype(np.float32)
    if sample_width == 1:
        data = (data - 128.0) / 128.0
    else:
        data /= float(2 ** (8 * sample_width - 1))
    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    return data.tolist(), sample_rate


def _resample_audio(waveform: Any, source_rate: int, target_rate: int = 16000) -> Any:
    """Deterministic resampling for Whisper-family audio feature extractors."""

    if source_rate == target_rate:
        return waveform
    import numpy as np

    values = np.asarray(waveform, dtype=np.float32)
    # Polyphase resampling includes the anti-aliasing low-pass needed for
    # non-integer rates such as ESC-50's 44.1 kHz -> 16 kHz conversion.  The
    # former linear interpolation path could alias high-frequency evidence.
    try:
        from scipy.signal import resample_poly
    except ImportError as exc:
        raise RuntimeError(
            "audio resampling requires scipy.signal.resample_poly"
        ) from exc
    divisor = math.gcd(int(source_rate), int(target_rate))
    result = resample_poly(
        values, target_rate // divisor, source_rate // divisor, padtype="constant"
    )
    target_size = int(round(values.size * target_rate / source_rate))
    if result.size > target_size:
        result = result[:target_size]
    elif result.size < target_size:
        result = np.pad(result, (0, target_size - result.size))
    return result.astype(np.float32).tolist()


class HuggingFaceScoringAdapter(ScoringAdapter):
    """Lazy Hugging Face adapter reserved for Qwen2.5-Omni checkpoints.

    The adapter intentionally has no import-time dependency on ``transformers``
    or ``torch``.  It uses a processor/model pair supplied by the caller or loads
    them from ``model_name`` on first score.  Audio payloads may be a path, a
    numeric waveform, or a mapping containing ``audio``/``audio_path`` and an
    optional ``sampling_rate``.  Model-specific processor keyword differences are
    handled conservatively (``audio`` then ``audios``).
    """

    def __init__(
        self,
        model_name: str,
        *,
        processor_name: str | None = None,
        device: str = "auto",
        dtype: str = "auto",
        trust_remote_code: bool = True,
        model: Any | None = None,
        processor: Any | None = None,
        tokenizer: Any | None = None,
    ) -> None:
        self.model_name = model_name
        self.processor_name = processor_name or model_name
        self.device_name = device
        self.dtype_name = dtype
        self.trust_remote_code = trust_remote_code
        self.model = model
        self.processor = processor
        self.tokenizer = tokenizer
        self._torch = None
        self._loaded = model is not None and (processor is not None or tokenizer is not None)

    def _load(self) -> None:
        if self._loaded:
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - exercised only with optional dependency
            raise RuntimeError(
                "HuggingFaceScoringAdapter requires torch and transformers; "
                "install the Qwen2.5-Omni runtime before using --adapter huggingface"
            ) from exc

        self._torch = torch
        if self.processor is None:
            self.processor = AutoProcessor.from_pretrained(
                self.processor_name, trust_remote_code=self.trust_remote_code
            )
        if self.tokenizer is None:
            self.tokenizer = getattr(self.processor, "tokenizer", None)
            if self.tokenizer is None:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.processor_name, trust_remote_code=self.trust_remote_code
                )
        if self.model is None:
            kwargs: dict[str, Any] = {"trust_remote_code": self.trust_remote_code}
            if self.dtype_name != "auto":
                dtype = getattr(torch, self.dtype_name, None)
                if dtype is None:
                    raise ValueError(f"unknown torch dtype {self.dtype_name!r}")
                kwargs["torch_dtype"] = dtype
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **kwargs)
        if self.device_name == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        else:
            device = self.device_name
        self.model.to(device)
        self._device = torch.device(device)
        self.model.eval()
        self._loaded = True

    @staticmethod
    def _payload_text(view: ViewRecord, question: str | None) -> tuple[str, Any, int | None]:
        payload = view.payload
        sampling_rate: int | None = None
        audio: Any = None
        text = question if question is not None else view.question
        if isinstance(payload, Mapping):
            text = payload.get("text", payload.get("prompt", text))
            audio = payload.get("audio", payload.get("audio_path", payload.get("waveform")))
            rate = payload.get("sampling_rate", payload.get("sample_rate"))
            if rate is not None:
                sampling_rate = int(rate)
        elif isinstance(payload, (str, Path)):
            # A path with an audio extension is treated as audio; other strings
            # are already-formed text payloads.
            suffix = Path(payload).suffix.lower()
            if suffix in {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".npy"}:
                audio = payload
            elif not text:
                text = str(payload)
        if text is None:
            text = ""
        return str(text), audio, sampling_rate

    @staticmethod
    def _materialize_audio(audio: Any) -> tuple[Any, int | None]:
        if isinstance(audio, (str, Path)):
            path = Path(audio)
            if path.suffix.lower() == ".wav":
                return _load_wav(path)
            if path.suffix.lower() == ".npy":
                import numpy as np

                return np.load(path).astype("float32").tolist(), None
            # Let external processors decode non-WAV paths when supported.
            return str(path), None
        return audio, None

    def _prepare(self, view: ViewRecord, text: str) -> Mapping[str, Any]:
        self._load()
        processor = self.processor or self.tokenizer
        if processor is None:
            raise RuntimeError("adapter has no tokenizer/processor")
        _, audio, sampling_rate = self._payload_text(view, text)
        kwargs: dict[str, Any] = {"text": text, "return_tensors": "pt"}
        if audio is not None:
            materialized, wav_rate = self._materialize_audio(audio)
            kwargs["audio"] = materialized
            rate = sampling_rate or wav_rate
            if rate is not None:
                kwargs["sampling_rate"] = rate
        try:
            return processor(**kwargs)
        except TypeError:
            if "audio" not in kwargs:
                raise
            # Qwen-family processors have used both names across releases.
            audio_value = kwargs.pop("audio")
            kwargs["audios"] = audio_value
            return processor(**kwargs)

    def _move_to_device(self, inputs: Mapping[str, Any]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in inputs.items():
            result[key] = value.to(self._device) if hasattr(value, "to") else value
        return result

    def _token_ids(self, label: str) -> list[int]:
        tokenizer = self.tokenizer or getattr(self.processor, "tokenizer", None)
        if tokenizer is None:
            raise RuntimeError("Qwen adapter requires a tokenizer")
        encoded = tokenizer(label, add_special_tokens=False)
        ids = encoded.get("input_ids", encoded) if isinstance(encoded, Mapping) else encoded
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        while isinstance(ids, list) and ids and isinstance(ids[0], list):
            ids = ids[0]
        return [int(item) for item in ids]

    def _score_label(self, view: ViewRecord, question: str, label: str) -> float:
        torch = self._torch
        assert torch is not None and self.model is not None
        label_ids = self._token_ids(label)
        if len(label_ids) != 1:
            raise ValueError(
                f"answer label {label!r} tokenizes to {len(label_ids)} tokens; "
                "PRI requires normalized single-token labels"
            )
        prompt_inputs = self._prepare(view, question)
        full_inputs = self._prepare(view, question + label)
        prompt_ids = prompt_inputs.get("input_ids")
        full_ids = full_inputs.get("input_ids")
        if prompt_ids is None or full_ids is None:
            raise RuntimeError("processor output must contain input_ids for answer-only scoring")
        prompt_len = int(prompt_ids.shape[-1])
        full_len = int(full_ids.shape[-1])
        if full_len <= prompt_len:
            raise RuntimeError("label was not appended to the processor input")
        outputs = self.model(**self._move_to_device(full_inputs))
        logits = outputs.logits[0]
        # Logits at position t predict token t+1.
        position = prompt_len - 1
        return float(torch.log_softmax(logits[position], dim=-1)[label_ids[0]].detach().cpu())

    def score_logprobs(
        self,
        view: ViewRecord,
        labels: Sequence[str],
        *,
        question: str | None = None,
    ) -> LabelScores:
        prompt = question if question is not None else view.question
        return {label: self._score_label(view, prompt, label) for label in labels}


class Qwen25OmniAdapter(ScoringAdapter):
    """Answer-only next-token scorer for Qwen2.5-Omni.

    Qwen2.5-Omni is not an ``AutoModelForCausalLM`` checkpoint.  Text logits
    come from ``model.thinker`` and the prompt must contain the processor's
    audio placeholder.  This adapter follows the official multimodal chat path
    while disabling the speech-output talker to save memory.
    """

    def __init__(
        self,
        model_name: str,
        *,
        device: str = "cuda",
        dtype: str = "bfloat16",
        attn_implementation: str | None = None,
        lora_path: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.device_name = device
        self.dtype_name = dtype
        self.attn_implementation = attn_implementation
        self.lora_path = lora_path
        self.model: Any | None = None
        self.processor: Any | None = None
        self.tokenizer: Any | None = None
        self._torch: Any | None = None
        self._thinker_only = False

    def _load(self) -> None:
        if self.model is not None:
            return
        import torch
        from transformers import (
            Qwen2_5OmniConfig,
            Qwen2_5OmniForConditionalGeneration,
            Qwen2_5OmniProcessor,
            Qwen2_5OmniThinkerForConditionalGeneration,
        )

        dtype = getattr(torch, self.dtype_name)
        raw_config = Qwen2_5OmniConfig.get_config_dict(self.model_name)[0]
        self._thinker_only = raw_config.get("model_type") == "qwen2_5_omni_thinker"
        kwargs: dict[str, Any] = {"torch_dtype": dtype}
        if self.attn_implementation:
            kwargs["attn_implementation"] = self.attn_implementation
        if self._thinker_only:
            self.model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
                self.model_name, **kwargs
            )
        else:
            config = Qwen2_5OmniConfig.from_pretrained(self.model_name)
            config.enable_audio_output = False
            kwargs["config"] = config
            self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
                self.model_name, **kwargs
            )
        if self.lora_path:
            from peft import PeftModel

            if self._thinker_only:
                self.model = PeftModel.from_pretrained(self.model, self.lora_path)
            else:
                self.model.thinker = PeftModel.from_pretrained(
                    self.model.thinker, self.lora_path
                )
        self.processor = Qwen2_5OmniProcessor.from_pretrained(self.model_name)
        self.tokenizer = self.processor.tokenizer
        self.model.to(self.device_name)
        self.model.eval()
        self._torch = torch

    def _scoring_model(self) -> Any:
        assert self.model is not None
        return self.model if self._thinker_only else self.model.thinker

    def _label_id(self, label: str) -> int:
        assert self.tokenizer is not None
        ids = self.tokenizer(label, add_special_tokens=False)["input_ids"]
        if len(ids) != 1:
            raise ValueError(
                f"answer label {label!r} tokenizes to {len(ids)} tokens; "
                "PRI requires normalized single-token labels"
            )
        return int(ids[0])

    def _prompt_inputs(self, view: ViewRecord, question: str) -> Mapping[str, Any]:
        assert self.processor is not None
        _, audio_value, sampling_rate = HuggingFaceScoringAdapter._payload_text(view, question)
        if audio_value is None:
            raise ValueError(f"view {view.name} has no audio payload")
        waveform, wav_rate = HuggingFaceScoringAdapter._materialize_audio(audio_value)
        rate = sampling_rate or wav_rate or 16000
        waveform = _resample_audio(waveform, rate, 16000)
        rate = 16000
        conversation = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "You are Qwen, a virtual human developed by the Qwen Team, "
                            "Alibaba Group, capable of perceiving auditory and visual inputs, "
                            "as well as generating text and speech."
                        ),
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": str(audio_value)},
                    {"type": "text", "text": question},
                ],
            },
        ]
        text = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )
        return self.processor(
            text=text,
            audio=[waveform],
            sampling_rate=rate,
            return_tensors="pt",
            padding=True,
        )

    def score_logprobs(
        self,
        view: ViewRecord,
        labels: Sequence[str],
        *,
        question: str | None = None,
    ) -> LabelScores:
        self._load()
        assert self.model is not None and self._torch is not None
        prompt = question if question is not None else view.question
        inputs = self._prompt_inputs(view, prompt)
        moved: dict[str, Any] = {}
        scoring_model = self._scoring_model()
        model_dtype = next(scoring_model.parameters()).dtype
        for key, value in inputs.items():
            if hasattr(value, "to"):
                value = value.to(self.device_name)
                if getattr(value, "is_floating_point", lambda: False)():
                    value = value.to(model_dtype)
            moved[key] = value
        with self._torch.inference_mode():
            outputs = scoring_model(**moved, return_dict=True)
            attention = moved.get("attention_mask")
            if attention is None:
                position = outputs.logits.shape[1] - 1
            else:
                position = int(attention[0].sum().item()) - 1
            log_probs = self._torch.log_softmax(outputs.logits[0, position].float(), dim=-1)
        return {
            str(label): float(log_probs[self._label_id(str(label))].detach().cpu())
            for label in labels
        }


Qwen2_5OmniAdapter = Qwen25OmniAdapter


class Qwen3OmniAdapter(ScoringAdapter):
    """Answer-only next-token scorer for Qwen3-Omni checkpoints.

    Qwen3-Omni exposes a composite thinker/talker model.  The stress panel
    scores text answers only, so loading disables the talker and forwards
    through ``model.thinker``.  This follows the official Transformers
    ``Qwen3OmniMoeProcessor`` multimodal chat path while keeping audio loading
    local and deterministic for the frozen WAV panel.
    """

    _SYSTEM_PROMPT = (
        "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
        "capable of perceiving auditory and visual inputs, as well as generating "
        "text and speech."
    )

    def __init__(
        self,
        model_name: str,
        *,
        device: str = "cuda",
        dtype: str = "bfloat16",
        attn_implementation: str | None = None,
        model: Any | None = None,
        processor: Any | None = None,
        tokenizer: Any | None = None,
    ) -> None:
        self.model_name = model_name
        self.device_name = device
        self.dtype_name = dtype
        self.attn_implementation = attn_implementation
        self.model = model
        self.processor = processor
        self.tokenizer = tokenizer
        self._torch: Any | None = None
        self._device: Any | None = None
        self._model_dtype: Any | None = None

    def _load(self) -> None:
        if self.tokenizer is None and self.processor is not None:
            self.tokenizer = getattr(self.processor, "tokenizer", None)
        if self.model is not None and self.processor is not None and self.tokenizer is not None:
            if self._torch is None:
                import torch

                self._torch = torch
            if self._device is None:
                self._device = self._resolve_device()
            if self._model_dtype is None:
                self._model_dtype = self._resolve_model_dtype()
            return

        try:
            import torch
            from transformers import (
                Qwen3OmniMoeConfig,
                Qwen3OmniMoeForConditionalGeneration,
                Qwen3OmniMoeProcessor,
            )
        except ImportError as exc:  # pragma: no cover - optional runtime dependency
            raise RuntimeError(
                "Qwen3OmniAdapter requires the source Transformers Qwen3-Omni runtime "
                "and torch; install transformers from the Qwen3-Omni-compatible source "
                "and qwen-omni-utils only if URL/remote media is needed"
            ) from exc

        self._torch = torch
        if self.processor is None:
            self.processor = Qwen3OmniMoeProcessor.from_pretrained(self.model_name)
        if self.tokenizer is None:
            self.tokenizer = getattr(self.processor, "tokenizer", None)
        if self.tokenizer is None:
            raise RuntimeError("Qwen3OmniProcessor did not expose a tokenizer")

        if self.model is None:
            config = Qwen3OmniMoeConfig.from_pretrained(self.model_name)
            # The panel never requests speech output.  Avoid constructing the
            # talker/code2wav modules (about 10 GB for the Instruct checkpoint).
            config.enable_audio_output = False
            kwargs: dict[str, Any] = {"config": config}
            if self.attn_implementation:
                kwargs["attn_implementation"] = self.attn_implementation
            if self.device_name == "auto":
                kwargs["device_map"] = "auto"
            if self.dtype_name == "auto":
                # ``dtype`` is the current Transformers API and is also
                # accepted by the Qwen3-Omni reference implementation.
                kwargs["dtype"] = "auto"
            else:
                dtype = getattr(torch, self.dtype_name, None)
                if dtype is None:
                    raise ValueError(f"unknown torch dtype {self.dtype_name!r}")
                kwargs["dtype"] = dtype
            try:
                self.model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
                    self.model_name, **kwargs
                )
            except TypeError as exc:
                # Older source snapshots use ``torch_dtype`` instead of the
                # newer ``dtype`` spelling.  Retry only for that API mismatch.
                if "dtype" not in str(exc):
                    raise
                legacy_kwargs = dict(kwargs)
                legacy_kwargs["torch_dtype"] = legacy_kwargs.pop("dtype")
                self.model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
                    self.model_name, **legacy_kwargs
                )

        # ``config.enable_audio_output=False`` prevents talker construction;
        # calling disable_talker also handles checkpoints/configs that still
        # materialized the module.
        disable_talker = getattr(self.model, "disable_talker", None)
        if callable(disable_talker):
            disable_talker()
        if self.device_name != "auto" and hasattr(self.model, "to"):
            self.model.to(self.device_name)
        if hasattr(self.model, "eval"):
            self.model.eval()
        self._device = self._resolve_device()
        self._model_dtype = self._resolve_model_dtype()

    def _resolve_device(self) -> Any:
        torch = self._torch
        if torch is None:
            return self.device_name
        if self.device_name != "auto":
            return torch.device(self.device_name)
        model = self.model
        model_device = getattr(model, "device", None)
        if model_device is not None and str(model_device) != "meta":
            return torch.device(model_device)
        if model is not None:
            try:
                return next(model.parameters()).device
            except (AttributeError, StopIteration):
                pass
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _resolve_model_dtype(self) -> Any | None:
        model = self._scoring_model()
        try:
            return next(model.parameters()).dtype
        except (AttributeError, StopIteration):
            return None

    def _scoring_model(self) -> Any:
        if self.model is None:
            raise RuntimeError("Qwen3OmniAdapter has not loaded a model")
        thinker = getattr(self.model, "thinker", None)
        if thinker is None:
            raise RuntimeError("Qwen3-Omni checkpoint did not expose model.thinker")
        return thinker

    def _label_id(self, label: str) -> int:
        if self.tokenizer is None:
            raise RuntimeError("Qwen3OmniAdapter requires a tokenizer")
        encoded = self.tokenizer(label, add_special_tokens=False)
        ids = encoded.get("input_ids", encoded) if isinstance(encoded, Mapping) else encoded
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        while isinstance(ids, list) and ids and isinstance(ids[0], list):
            ids = ids[0]
        if not isinstance(ids, Sequence) or len(ids) != 1:
            count = len(ids) if isinstance(ids, Sequence) else "unknown"
            raise ValueError(
                f"answer label {label!r} tokenizes to {count} tokens; "
                "QSAEC requires normalized single-token labels"
            )
        return int(ids[0])

    def _prompt_inputs(self, view: ViewRecord, question: str) -> Mapping[str, Any]:
        if self.processor is None:
            raise RuntimeError("Qwen3OmniAdapter has no processor")
        _, audio_value, sampling_rate = HuggingFaceScoringAdapter._payload_text(view, question)
        if audio_value is None:
            raise ValueError(f"view {view.name} has no audio payload")
        waveform, wav_rate = HuggingFaceScoringAdapter._materialize_audio(audio_value)
        rate = sampling_rate or wav_rate or 16000
        waveform = _resample_audio(waveform, rate, 16000)
        conversation = [
            {
                "role": "system",
                "content": [{"type": "text", "text": self._SYSTEM_PROMPT}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": str(audio_value)},
                    {"type": "text", "text": question},
                ],
            },
        ]
        text = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )
        inputs = self.processor(
            text=text,
            audio=[waveform],
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
        )
        if "input_features" not in inputs:
            raise RuntimeError(
                "Qwen3-Omni processor did not produce input_features; refusing "
                f"text-only scoring (processor keys: {sorted(inputs.keys())})"
            )
        return inputs

    def _move_inputs(self, inputs: Mapping[str, Any]) -> dict[str, Any]:
        if self._device is None:
            raise RuntimeError("Qwen3OmniAdapter has no input device")
        moved: dict[str, Any] = {}
        for key, value in inputs.items():
            if hasattr(value, "to"):
                value = value.to(self._device)
                if self._model_dtype is not None and getattr(
                    value, "is_floating_point", lambda: False
                )():
                    value = value.to(self._model_dtype)
            moved[key] = value
        return moved

    def score_logprobs(
        self,
        view: ViewRecord,
        labels: Sequence[str],
        *,
        question: str | None = None,
    ) -> LabelScores:
        self._load()
        torch = self._torch
        if torch is None:
            raise RuntimeError("Qwen3OmniAdapter could not initialize torch")
        prompt = question if question is not None else view.question
        inputs = self._prompt_inputs(view, prompt)
        moved = self._move_inputs(inputs)
        scoring_model = self._scoring_model()
        with torch.inference_mode():
            outputs = scoring_model(**moved, return_dict=True)
            logits = outputs.logits
            attention = moved.get("attention_mask")
            position = (
                int(attention[0].sum().item()) - 1
                if attention is not None
                else logits.shape[1] - 1
            )
            if position < 0 or position >= logits.shape[1]:
                raise RuntimeError(
                    f"invalid answer position {position} for logits shape {tuple(logits.shape)}"
                )
            log_probs = torch.log_softmax(logits[0, position].float(), dim=-1)
        return {
            str(label): float(log_probs[self._label_id(str(label))].detach().cpu())
            for label in labels
        }


# Descriptive alias for callers that include the checkpoint size in model IDs.
Qwen3Omni30BAdapter = Qwen3OmniAdapter


class Qwen2AudioAdapter(ScoringAdapter):
    """Answer-only scorer for Qwen2-Audio instruction checkpoints."""

    def __init__(self, model_name: str, *, device: str = "cuda", dtype: str = "bfloat16") -> None:
        self.model_name = model_name
        self.device_name = device
        self.dtype_name = dtype
        self.model: Any | None = None
        self.processor: Any | None = None
        self.tokenizer: Any | None = None
        self._torch: Any | None = None

    def _load(self) -> None:
        if self.model is not None:
            return
        import torch
        from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.tokenizer = self.processor.tokenizer
        self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
            self.model_name, torch_dtype=getattr(torch, self.dtype_name)
        ).to(self.device_name)
        self.model.eval()
        self._torch = torch

    def _label_id(self, label: str) -> int:
        ids = self.tokenizer(label, add_special_tokens=False)["input_ids"]
        if len(ids) != 1:
            raise ValueError(f"answer label {label!r} must tokenize to one token")
        return int(ids[0])

    def score_logprobs(self, view: ViewRecord, labels: Sequence[str], *, question: str | None = None) -> LabelScores:
        self._load()
        assert self.model is not None and self.processor is not None and self._torch is not None
        prompt = question if question is not None else view.question
        _, audio_value, sampling_rate = HuggingFaceScoringAdapter._payload_text(view, prompt)
        if audio_value is None:
            raise ValueError(f"view {view.name} has no audio payload")
        waveform, wav_rate = HuggingFaceScoringAdapter._materialize_audio(audio_value)
        rate = sampling_rate or wav_rate or 16000
        waveform = _resample_audio(waveform, rate, 16000)
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio_url": str(audio_value)},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )
        inputs = self.processor(
            text=text,
            audio=[waveform],
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
        )
        # Recent transformers releases silently ignore unknown processor
        # keywords.  Refuse to score when the waveform was not encoded: a
        # text-only fallback would make the entire frozen panel invalid.
        if "input_features" not in inputs and "audio_values" not in inputs:
            raise RuntimeError(
                "Qwen2-Audio processor did not produce audio features; "
                f"processor keys were {sorted(inputs.keys())}"
            )
        moved = {key: value.to(self.device_name) if hasattr(value, "to") else value for key, value in inputs.items()}
        with self._torch.inference_mode():
            outputs = self.model(**moved, return_dict=True)
            attention = moved.get("attention_mask")
            position = int(attention[0].sum().item()) - 1 if attention is not None else outputs.logits.shape[1] - 1
            log_probs = self._torch.log_softmax(outputs.logits[0, position].float(), dim=-1)
        return {label: float(log_probs[self._label_id(label)].cpu()) for label in labels}


class MiMoAudioAdapter(ScoringAdapter):
    """Answer-only scorer for ``XiaomiMiMo/MiMo-Audio-7B-Instruct``.

    MiMo-Audio is a custom audio language model rather than a Transformers
    ``Processor`` checkpoint.  The official runtime first turns a waveform into
    RVQ codes with ``MimoAudio.preprocess_input`` and then builds the
    ``InputSegment`` chat prompt used by ``audio_understanding_sft``.  Its
    custom causal-LM forward returns ``text_logits`` for the final prompt
    group, so the adapter reads A/B next-token log probabilities directly and
    never calls generation.

    ``runtime`` is an intentional injection point for CPU tests.  A real run
    lazily imports the source checkout's ``src.mimo_audio`` package and reports
    an actionable dependency error when that optional runtime is unavailable.
    ``tokenizer_path`` is the separate MiMo-Audio-Tokenizer directory required
    by the official ``MimoAudio`` constructor.
    """

    _DEFAULT_MODEL = "XiaomiMiMo/MiMo-Audio-7B-Instruct"

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        tokenizer_path: str | None = None,
        *,
        device: str = "cuda",
        dtype: str = "bfloat16",
        runtime: Any | None = None,
        mimo_audio: Any | None = None,
    ) -> None:
        if runtime is not None and mimo_audio is not None and runtime is not mimo_audio:
            raise ValueError("provide only one of runtime or mimo_audio")
        self.model_name = model_name
        self.tokenizer_path = tokenizer_path
        self.device_name = device
        self.dtype_name = dtype
        self.runtime = runtime if runtime is not None else mimo_audio
        self._torch: Any | None = None
        self._input_segment: Any | None = None
        self._loaded = self.runtime is not None

    @staticmethod
    def _dependency_error(detail: str) -> RuntimeError:
        return RuntimeError(
            "MiMoAudioAdapter requires the official MiMo-Audio runtime and its "
            "audio dependencies (torch, torchaudio, transformers, scipy). "
            "Clone https://github.com/XiaomiMiMo/MiMo-Audio and put its repository "
            "root on PYTHONPATH (for example, PYTHONPATH=/path/to/MiMo-Audio), "
            "then install the pinned requirements before using --adapter mimo. "
            f"Import detail: {detail}"
        )

    def _load(self) -> None:
        if self._loaded:
            if self._torch is None:
                try:
                    import torch
                except ImportError as exc:  # pragma: no cover - runtime-only path
                    raise self._dependency_error(str(exc)) from exc
                self._torch = torch
            return
        if not self.tokenizer_path:
            raise RuntimeError(
                "MiMoAudioAdapter requires --tokenizer-path pointing to the "
                "MiMo-Audio-Tokenizer checkpoint directory; the 7B language-model "
                "directory does not contain the audio tokenizer."
            )
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - runtime-only path
            raise self._dependency_error(str(exc)) from exc
        self._torch = torch
        try:
            # The official checkout is a namespace package rooted at the repo
            # directory.  Keep both spellings for installed/source layouts.
            try:
                from src.mimo_audio.mimo_audio import MimoAudio
                from src.mimo_audio.process_speechdata import InputSegment
            except ImportError:
                from mimo_audio.mimo_audio import MimoAudio
                from mimo_audio.process_speechdata import InputSegment
        except (ImportError, ModuleNotFoundError) as exc:  # pragma: no cover
            raise self._dependency_error(str(exc)) from exc
        try:
            self.runtime = MimoAudio(
                self.model_name,
                self.tokenizer_path,
                device=self.device_name,
            )
        except Exception as exc:  # pragma: no cover - model/runtime-only path
            raise RuntimeError(
                "MiMoAudioAdapter could not initialize the official MimoAudio "
                f"runtime for model {self.model_name!r} and tokenizer "
                f"{self.tokenizer_path!r}: {exc}"
            ) from exc
        self._input_segment = InputSegment
        self._loaded = True

    @staticmethod
    def _audio_payload(view: ViewRecord) -> tuple[Any, int | None]:
        payload = view.payload
        if isinstance(payload, Mapping):
            audio = payload.get("audio", payload.get("audio_path", payload.get("waveform")))
            rate = payload.get("sampling_rate", payload.get("sample_rate"))
            return audio, int(rate) if rate is not None else None
        if isinstance(payload, (str, Path)):
            return payload, None
        # An already-loaded numeric waveform is accepted for deterministic
        # fixtures; the official runtime accepts it after tensor conversion.
        return payload, None

    @staticmethod
    def _numel(value: Any) -> int:
        if value is None:
            return 0
        numel = getattr(value, "numel", None)
        if callable(numel):
            try:
                return int(numel())
            except (TypeError, ValueError):
                pass
        size = getattr(value, "size", None)
        if isinstance(size, int):
            return size
        try:
            return len(value)
        except TypeError:
            return 1

    def _prepare_audio(self, audio: Any, sampling_rate: int | None) -> Any:
        if audio is None:
            raise ValueError(
                "MiMoAudioAdapter requires an audio/audio_path/waveform payload; "
                "text-only views cannot be scored by the MiMo-Audio adapter"
            )
        if isinstance(audio, (str, Path)):
            path = Path(audio)
            if not path.exists():
                raise FileNotFoundError(f"MiMo-Audio input audio does not exist: {path}")
            # Recent torchaudio releases route file decoding through
            # TorchCodec, whose binary build may not match the cluster CUDA
            # runtime. Decode the PCM WAV with the standard library and then
            # follow the same waveform/resampling path as an in-memory input.
            values, rate = _load_wav(path)
            return self._prepare_audio(values, rate)
        if self._numel(audio) == 0:
            raise ValueError("MiMo-Audio input waveform is empty")
        # MimoAudio.preprocess_input accepts a torch.Tensor as a waveform.  A
        # mapping's sampling rate is resampled before conversion so the
        # official 24 kHz tokenizer sees the declared rate rather than silently
        # treating arbitrary-rate samples as 24 kHz.
        try:
            import numpy as np

            values = np.asarray(audio, dtype=np.float32)
            # Match the official ``MimoAudio.preprocess_input`` convention:
            # multi-channel tensors are averaged to mono before resampling.
            if values.ndim > 1:
                values = values.mean(axis=0)
            values = values.reshape(-1)
            if values.size == 0 or not np.isfinite(values).all():
                raise ValueError("MiMo-Audio input waveform must contain finite samples")
            tokenizer = getattr(self.runtime, "mimo_audio_tokenizer", None)
            config = getattr(tokenizer, "config", None)
            expected_rate = getattr(config, "sampling_rate", None)
            if sampling_rate is not None and expected_rate is not None and sampling_rate != expected_rate:
                values = np.asarray(
                    _resample_audio(values.tolist(), sampling_rate, int(expected_rate)),
                    dtype=np.float32,
                )
            if self._torch is not None:
                return self._torch.as_tensor(values, dtype=self._torch.float32)
            return values.tolist()
        except ImportError as exc:
            raise RuntimeError(
                "MiMo-Audio numeric waveform preprocessing requires numpy; use a WAV "
                "path or install the official runtime requirements"
            ) from exc

    def _label_id(self, label: str) -> int:
        tokenizer = getattr(self.runtime, "tokenizer", None)
        if tokenizer is None:
            raise RuntimeError("MiMoAudioAdapter runtime did not expose a tokenizer")
        encoded = tokenizer(label, add_special_tokens=False)
        ids = encoded.get("input_ids", encoded) if isinstance(encoded, Mapping) else encoded
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        while isinstance(ids, list) and ids and isinstance(ids[0], list):
            ids = ids[0]
        if not isinstance(ids, Sequence) or len(ids) != 1:
            count = len(ids) if isinstance(ids, Sequence) else "unknown"
            raise ValueError(
                f"answer label {label!r} tokenizes to {count} tokens; "
                "QSAEC requires normalized single-token labels"
            )
        return int(ids[0])

    def _prompt_inputs(self, view: ViewRecord, question: str) -> Any:
        audio, sampling_rate = self._audio_payload(view)
        audio = self._prepare_audio(audio, sampling_rate)
        preprocess = getattr(self.runtime, "preprocess_input", None)
        if not callable(preprocess):
            raise RuntimeError("MiMoAudio runtime does not expose official preprocess_input")
        audio_tokens = preprocess(audio)
        if self._numel(audio_tokens) <= 0:
            raise RuntimeError(
                "MiMo-Audio preprocessing produced no audio features/tokens; refusing "
                "text-only scoring"
            )

        # Build the exact official audio-understanding chat prompt from the
        # preprocessed RVQ codes.  This avoids encoding each view twice.  The
        # fallback invokes the official helper directly for injected runtimes
        # that do not expose InputSegment (the CPU fake path).
        if self._input_segment is not None and callable(getattr(self.runtime, "get_input_ids", None)):
            segment = self._input_segment
            kwargs = {
                "speech_zeroemb_idx": self.runtime.speech_zeroemb_idx,
                "text_zeroemb_idx": self.runtime.empty_token,
            }
            prompt = [
                segment(text="<|im_start|>user\n", **kwargs),
                segment(audio=audio_tokens, **kwargs),
                segment(text=question, **kwargs),
                segment(text="<|im_end|>\n", **kwargs),
                segment(text="<|im_start|>assistant\n", **kwargs),
                segment(text="<think>\n\n</think>\n", **kwargs),
            ]
            inputs = self.runtime.get_input_ids(prompt)
        else:
            official_prompt = getattr(self.runtime, "get_audio_understanding_sft_prompt", None)
            if not callable(official_prompt):
                raise RuntimeError(
                    "MiMo-Audio runtime exposes neither InputSegment/get_input_ids nor "
                    "the official get_audio_understanding_sft_prompt helper"
                )
            inputs = official_prompt(audio, question, thinking=False)

        shape = getattr(inputs, "shape", None)
        if shape is None or len(shape) != 2 or int(shape[-1]) <= 0:
            raise RuntimeError(
                "MiMo-Audio chat preprocessing returned an empty/invalid input tensor; "
                f"got shape {shape!r}"
            )
        return inputs

    def _forward_logits(self, inputs: Any) -> Any:
        torch = self._torch
        if torch is None:
            raise RuntimeError("MiMoAudioAdapter could not initialize torch")
        model = getattr(self.runtime, "model", None)
        if model is None:
            raise RuntimeError("MiMoAudio runtime did not expose its causal-LM model")
        if not hasattr(inputs, "ndim"):
            inputs = torch.as_tensor(inputs)
        if inputs.ndim == 2:
            inputs = inputs.unsqueeze(0)
        if inputs.ndim != 3:
            raise RuntimeError(
                "MiMo-Audio model input must have shape [batch, audio_channels+1, time], "
                f"got {tuple(inputs.shape)}"
            )
        device_name = getattr(self.runtime, "device", self.device_name)
        try:
            device = torch.device(device_name)
        except (TypeError, RuntimeError):
            device = torch.device(self.device_name)
        inputs = inputs.to(device)
        group_size = int(getattr(self.runtime, "group_size", 1))
        time_steps = int(inputs.shape[-1])
        if group_size <= 0 or time_steps % group_size:
            raise RuntimeError(
                f"MiMo-Audio input time dimension {time_steps} is not divisible by "
                f"group_size={group_size}"
            )
        groups = time_steps // group_size
        attention_mask = torch.ones((inputs.shape[0], groups), dtype=torch.bool, device=device)
        position_ids = torch.arange(groups, dtype=torch.long, device=device).unsqueeze(0)
        with torch.inference_mode():
            try:
                outputs = model(
                    input_ids=inputs,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    return_dict=True,
                )
            except TypeError as exc:
                if "return_dict" not in str(exc):
                    raise
                outputs = model(
                    input_ids=inputs,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                )
        logits = getattr(outputs, "text_logits", None)
        if logits is None:
            logits = getattr(outputs, "logits", None)
        if logits is None:
            raise RuntimeError("MiMo-Audio forward output did not contain text_logits")
        if getattr(logits, "ndim", 0) == 3:
            logits = logits[:, -1, :]
        if getattr(logits, "ndim", 0) != 2:
            raise RuntimeError(f"MiMo-Audio text_logits has invalid shape {getattr(logits, 'shape', None)!r}")
        return logits[0].float()

    def score_logprobs(
        self,
        view: ViewRecord,
        labels: Sequence[str],
        *,
        question: str | None = None,
    ) -> LabelScores:
        self._load()
        prompt = question if question is not None else view.question
        if not isinstance(prompt, str) or not prompt:
            raise ValueError("MiMoAudioAdapter requires a non-empty audio-understanding question")
        normalized = [str(label) for label in labels]
        if not normalized:
            raise ValueError("at least one answer label is required")
        token_ids = {label: self._label_id(label) for label in normalized}
        logits = self._forward_logits(self._prompt_inputs(view, prompt))
        torch = self._torch
        assert torch is not None
        max_id = max(token_ids.values())
        if max_id < 0 or max_id >= int(logits.shape[-1]):
            raise RuntimeError(
                f"MiMo-Audio tokenizer label id {max_id} is outside model vocabulary "
                f"size {int(logits.shape[-1])}"
            )
        log_probs = torch.log_softmax(logits, dim=-1)
        return {
            label: float(log_probs[token_id].detach().cpu())
            for label, token_id in token_ids.items()
        }


# Keep spelling variants convenient for callers and checkpoint manifests.
MimoAudioAdapter = MiMoAudioAdapter
MiMoAdapter = MiMoAudioAdapter


class StepAudio2Adapter(ScoringAdapter):
    """Answer-only scorer for ``stepfun-ai/Step-Audio-2-mini``.

    Step-Audio 2 uses custom remote code rather than a Transformers audio
    processor.  Its official wrapper builds ``<|BOT|>human``/``<|BOT|>assistant``
    turns, replaces an ``<audio_start>`` marker with log-mel features, and calls
    ``StepAudio2ForCausalLM(input_ids, wavs, wav_lens, attention_mask)``.  This
    adapter follows that path directly and deliberately refuses text-only
    fallback when an audio payload is absent or cannot be encoded.

    ``model``, ``tokenizer`` and ``audio_encoder`` are injectable for CPU tests.
    ``audio_encoder`` should expose ``(waveform, sample_rate) -> mel`` or, when
    omitted, the adapter lazily constructs the same 16-kHz/128-mel transform as
    the official ``stepaudio2.utils.log_mel_spectrogram`` helper.  The transform
    is kept local because that helper lives in the inference repository rather
    than the Hugging Face custom-model module.
    """

    _AUDIO_START_ID = 151688
    _AUDIO_END_ID = 151689
    _AUDIO_PATCH = "<audio_patch>"
    _BOT = "<|BOT|>"
    _EOT = "<|EOT|>"

    def __init__(
        self,
        model_name: str,
        *,
        device: str = "cuda",
        dtype: str = "bfloat16",
        trust_remote_code: bool = True,
        model: Any | None = None,
        tokenizer: Any | None = None,
        audio_encoder: Callable[..., Any] | None = None,
        audio_token_num: Callable[[int], int] | None = None,
    ) -> None:
        self.model_name = model_name
        self.device_name = device
        self.dtype_name = dtype
        self.trust_remote_code = trust_remote_code
        self.model = model
        self.tokenizer = tokenizer
        self.audio_encoder = audio_encoder
        self.audio_token_num = audio_token_num
        self._torch: Any | None = None
        self._loaded = model is not None and tokenizer is not None
        self._audio_utils_loaded = False
        self._compute_token_num: Callable[[int], int] | None = None

    def _load(self) -> None:
        if self._loaded:
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - optional runtime path
            raise RuntimeError(
                "StepAudio2Adapter requires torch and transformers with the "
                "Step-Audio-2 custom modeling code; install the official runtime "
                "(transformers==4.49.0 plus librosa/torchaudio)."
            ) from exc
        self._torch = torch
        self.tokenizer = self.tokenizer or AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=self.trust_remote_code,
            padding_side="right",
        )
        kwargs: dict[str, Any] = {"trust_remote_code": self.trust_remote_code}
        if self.dtype_name != "auto":
            dtype = getattr(torch, self.dtype_name, None)
            if dtype is None:
                raise ValueError(f"unknown torch dtype {self.dtype_name!r}")
            # Official examples use torch_dtype; newer Transformers accepts dtype.
            kwargs["torch_dtype"] = dtype
        self.model = self.model or AutoModelForCausalLM.from_pretrained(self.model_name, **kwargs)
        if self.device_name != "auto" and hasattr(self.model, "to"):
            self.model.to(self.device_name)
        if hasattr(self.model, "eval"):
            self.model.eval()
        self._loaded = True

    def _load_audio_utils(self) -> None:
        if self._audio_utils_loaded:
            return
        if self.audio_encoder is not None and self.audio_token_num is not None:
            self._audio_utils_loaded = True
            self._compute_token_num = self.audio_token_num
            return
        if self.audio_encoder is None:
            try:
                import librosa
                import torchaudio
                import torch
                import torch.nn.functional as F

                def _official_encoder(waveform: Any, sample_rate: int = 16000) -> Any:
                    if isinstance(waveform, (str, Path)):
                        values, rate = _load_wav(waveform)
                        waveform = torch.tensor(values, dtype=torch.float32)
                        sample_rate = rate
                    elif not torch.is_tensor(waveform):
                        waveform = torch.tensor(waveform, dtype=torch.float32)
                    if waveform.ndim > 1:
                        waveform = waveform.mean(dim=0)
                    if sample_rate != 16000:
                        waveform = torchaudio.transforms.Resample(sample_rate, 16000)(
                            waveform.unsqueeze(0)
                        )[0]
                    waveform = F.pad(waveform, (0, 479))
                    window = torch.hann_window(400, device=waveform.device)
                    stft = torch.stft(
                        waveform, 400, 160, window=window, return_complex=True
                    )
                    magnitudes = stft[..., :-1].abs() ** 2
                    filters = torch.from_numpy(
                        librosa.filters.mel(sr=16000, n_fft=400, n_mels=128)
                    ).to(waveform)
                    mel = filters @ magnitudes
                    log_spec = torch.clamp(mel, min=1e-10).log10()
                    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
                    return (log_spec + 4.0) / 4.0

                self.audio_encoder = _official_encoder
            except ImportError as exc:  # pragma: no cover - optional runtime path
                raise RuntimeError(
                    "StepAudio2Adapter cannot encode audio: install the official "
                    "Step-Audio-2 dependencies (torchaudio, librosa), or inject "
                    "audio_encoder for a controlled test."
                ) from exc
        if self.audio_token_num is None:
            def _official_token_num(feature_len: int) -> int:
                # Official compute_token_num for the 128-mel encoder/adapter.
                max_feature_len = feature_len - 2
                encoder_output_dim = (max_feature_len + 1) // 2 // 2
                return (encoder_output_dim + 2 - 3) // 2 + 1

            self._compute_token_num = _official_token_num
        else:
            self._compute_token_num = self.audio_token_num
        self._audio_utils_loaded = True

    @staticmethod
    def _audio_payload(view: ViewRecord) -> tuple[Any, int]:
        payload = view.payload
        sampling_rate = 16000
        audio: Any = None
        if isinstance(payload, Mapping):
            audio = payload.get("audio", payload.get("audio_path", payload.get("waveform")))
            sampling_rate = int(payload.get("sampling_rate", payload.get("sample_rate", 16000)))
        elif isinstance(payload, (str, Path)):
            audio = payload
        if audio is None:
            raise ValueError(
                f"view {view.name} has no audio payload; Step-Audio-2 scoring cannot use text-only input"
            )
        return audio, sampling_rate

    def _label_id(self, label: str) -> int:
        if self.tokenizer is None:
            raise RuntimeError("StepAudio2Adapter requires a tokenizer")
        encoded = self.tokenizer(label, add_special_tokens=False)
        ids = encoded.get("input_ids", encoded) if isinstance(encoded, Mapping) else encoded
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        while isinstance(ids, list) and ids and isinstance(ids[0], list):
            ids = ids[0]
        if not isinstance(ids, Sequence) or len(ids) != 1:
            count = len(ids) if isinstance(ids, Sequence) else "unknown"
            raise ValueError(
                f"answer label {label!r} tokenizes to {count} tokens; "
                "QSAEC requires normalized single-token labels"
            )
        return int(ids[0])

    def _official_messages(self, question: str, audio: Any) -> tuple[str, Any, int]:
        """Return official StepAudio2 prompt text and one encoded mel clip."""
        self._load_audio_utils()
        assert self.audio_encoder is not None and self._compute_token_num is not None
        payload, sample_rate = audio
        # The official wrapper chunks clips at 25 s.  QSAEC's frozen clips are
        # short, so one encoded clip follows the same marker/wav_lens protocol.
        if isinstance(payload, (str, Path)) and str(payload).lower().endswith(".wav"):
            waveform, wav_rate = _load_wav(payload)
            payload, sample_rate = waveform, wav_rate
        try:
            mel = self.audio_encoder(payload, sample_rate)
        except TypeError as exc:
            # Tiny injected fakes often accept only the waveform.  The official
            # helper accepts the optional sample rate, so retry only on a
            # signature mismatch and preserve all real encoder errors.
            if "positional" not in str(exc) and "argument" not in str(exc):
                raise
            mel = self.audio_encoder(payload)
        if not hasattr(mel, "shape") or len(mel.shape) != 2:
            raise RuntimeError("Step-Audio-2 audio encoder must return a (128, frames) mel tensor")
        token_count = int(self._compute_token_num(int(mel.shape[-1])))
        marker = "<audio_start>" + self._AUDIO_PATCH * token_count + "<audio_end>"
        # Match StepAudio2.apply_chat_template exactly for a final assistant turn.
        text = (
            f"{self._BOT}system\nYou are a helpful assistant.{self._EOT}"
            f"{self._BOT}human\n{marker}{question}{self._EOT}"
            f"{self._BOT}assistant\n"
        )
        return text, mel, int(getattr(mel, "shape", [0, 0])[-1] - 2)

    def _encode_prompt(self, view: ViewRecord, question: str) -> Mapping[str, Any]:
        if self.tokenizer is None:
            raise RuntimeError("StepAudio2Adapter has no tokenizer")
        audio, sample_rate = self._audio_payload(view)
        text, mel, mel_len = self._official_messages(question, (audio, sample_rate))
        encoded = self.tokenizer(text=text, return_tensors="pt", padding=True)
        if not isinstance(encoded, Mapping) or "input_ids" not in encoded:
            raise RuntimeError("Step-Audio-2 tokenizer did not return input_ids")
        torch = self._torch
        if torch is None:
            import torch as _torch

            torch = _torch
            self._torch = torch
        if not torch.is_tensor(mel):
            mel = torch.tensor(mel)
        # Official padding_mels subtracts the two padded frames from wav_lens.
        encoded = dict(encoded)
        encoded["wavs"] = mel.unsqueeze(0)
        encoded["wav_lens"] = torch.tensor([mel_len], dtype=torch.int32)
        if "attention_mask" not in encoded:
            encoded["attention_mask"] = torch.ones_like(encoded["input_ids"])
        return encoded

    def score_logprobs(
        self,
        view: ViewRecord,
        labels: Sequence[str],
        *,
        question: str | None = None,
    ) -> LabelScores:
        self._load()
        self._load_audio_utils()
        assert self.model is not None and self.tokenizer is not None
        torch = self._torch
        if torch is None:
            import torch as _torch

            torch = _torch
            self._torch = torch
        prompt = question if question is not None else view.question
        inputs = self._encode_prompt(view, prompt)
        moved: dict[str, Any] = {}
        device = self.device_name
        if device == "auto":
            device = str(getattr(self.model, "device", "cpu"))
        for key, value in inputs.items():
            moved[key] = value.to(device) if hasattr(value, "to") else value
        with torch.inference_mode():
            outputs = self.model(**moved, return_dict=True)
            logits = getattr(outputs, "logits", None)
            if logits is None:
                raise RuntimeError("Step-Audio-2 model output did not expose logits")
            attention = moved.get("attention_mask")
            position = int(attention[0].sum().item()) - 1 if attention is not None else logits.shape[1] - 1
            if position < 0 or position >= logits.shape[1]:
                raise RuntimeError(f"invalid answer position {position} for logits shape {tuple(logits.shape)}")
            log_probs = torch.log_softmax(logits[0, position].float(), dim=-1)
        return {
            str(label): float(log_probs[self._label_id(str(label))].detach().cpu())
            for label in labels
        }


# Descriptive aliases used by launch scripts and model registries.
StepAudio2MiniAdapter = StepAudio2Adapter
StepAudioAdapter = StepAudio2Adapter


class AudioFlamingo3Adapter(ScoringAdapter):
    """Answer-only scorer for NVIDIA Audio Flamingo 3 checkpoints.

    AF3 is loaded from the official ``llava`` checkout rather than through a
    Transformers ``Processor``.  Audio is expanded by the repository's
    ``extract_media``/``process_sounds`` helpers, and the resulting media plus
    masks are passed to ``LlavaLlamaModel`` exactly as in the official runtime.
    The adapter intentionally accepts file-backed audio only: a missing or
    failed audio feature extraction is an error, never a text-only fallback.

    ``runtime`` is an injection point for CPU/unit tests.  It may be a mapping
    or object exposing ``load``, ``Sound``, ``extract_media``,
    ``process_sounds``, ``process_sound_masks``, and ``tokenize_conversation``.
    A real run lazily imports those symbols from ``repo_path`` (or the
    installed ``llava`` package) on first score.
    """

    def __init__(
        self,
        model_name: str,
        *,
        repo_path: str | Path | None = None,
        device: str = "cuda",
        dtype: str = "float16",
        model: Any | None = None,
        runtime: Any | None = None,
    ) -> None:
        self.model_name = model_name
        self.repo_path = str(repo_path) if repo_path is not None else None
        self.device_name = device
        self.dtype_name = dtype
        self.model = model
        self.runtime = runtime
        self._torch: Any | None = None
        self._load_fn: Callable[..., Any] | None = None
        self._Sound: Any | None = None
        self._extract_media: Callable[..., Any] | None = None
        self._process_sounds: Callable[..., Any] | None = None
        self._process_sound_masks: Callable[..., Any] | None = None
        self._tokenize_conversation: Callable[..., Any] | None = None
        # Label ids are invariant for the lifetime of a loaded checkpoint.  In
        # QSAEC every cell asks for the same A/B tokens, so avoid invoking the
        # tokenizer once per cell while keeping the public string-label API.
        self._label_id_cache: dict[str, int] = {}
        self._loaded = False

    @staticmethod
    def _runtime_get(runtime: Any, name: str) -> Any:
        if isinstance(runtime, Mapping):
            return runtime.get(name)
        return getattr(runtime, name, None)

    def _bind_runtime(self, runtime: Any) -> None:
        self._load_fn = self._runtime_get(runtime, "load")
        self._Sound = self._runtime_get(runtime, "Sound")
        self._extract_media = self._runtime_get(runtime, "extract_media")
        self._process_sounds = self._runtime_get(runtime, "process_sounds")
        self._process_sound_masks = self._runtime_get(runtime, "process_sound_masks")
        self._tokenize_conversation = self._runtime_get(runtime, "tokenize_conversation")

    @staticmethod
    def _dependency_error(detail: str) -> RuntimeError:
        return RuntimeError(
            "AudioFlamingo3Adapter requires the official NVIDIA Audio Flamingo 3 "
            "runtime (the repository's llava package) plus torch, transformers, "
            "soundfile, and kaldiio. Clone NVIDIA/audio-flamingo, pass its root "
            "with --af3-repo (or install it), then install the pinned requirements. "
            f"Import detail: {detail}"
        )

    def _load(self) -> None:
        if self._loaded:
            return

        try:
            import torch
        except ImportError as exc:  # pragma: no cover - optional runtime path
            raise self._dependency_error(str(exc)) from exc
        self._torch = torch

        if self.runtime is not None:
            self._bind_runtime(self.runtime)
        else:
            if self.repo_path:
                import sys

                path = str(Path(self.repo_path).expanduser().resolve())
                if path not in sys.path:
                    sys.path.insert(0, path)
            try:
                from llava import load
                from llava.media import Sound
                from llava.mm_utils import process_sound_masks, process_sounds
                from llava.utils.media import extract_media
                from llava.utils.tokenizer import tokenize_conversation
            except (ImportError, ModuleNotFoundError) as exc:  # pragma: no cover
                raise self._dependency_error(str(exc)) from exc
            self._load_fn = load
            self._Sound = Sound
            self._extract_media = extract_media
            self._process_sounds = process_sounds
            self._process_sound_masks = process_sound_masks
            self._tokenize_conversation = tokenize_conversation

        required = {
            "Sound": self._Sound,
            "extract_media": self._extract_media,
            "process_sounds": self._process_sounds,
            "process_sound_masks": self._process_sound_masks,
            "tokenize_conversation": self._tokenize_conversation,
        }
        if self.model is None:
            required["load"] = self._load_fn
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise RuntimeError(
                "AudioFlamingo3Adapter runtime is missing official helpers: "
                + ", ".join(missing)
            )

        if self.model is None:
            assert self._load_fn is not None
            load_kwargs: dict[str, Any]
            if self.device_name in {"auto", "cuda"}:
                load_kwargs = {"device_map": "auto"}
            else:
                # The official builder accepts a device map of the form
                # {"": "cuda:0"} and keeps all components on that device.
                load_kwargs = {"device_map": {"": self.device_name}}
            try:
                self.model = self._load_fn(self.model_name, **load_kwargs)
            except TypeError as exc:
                # Small injected runtimes may only accept a path.  Do not hide
                # real model-loading failures from the official runtime.
                if self.runtime is None or "device_map" not in str(exc):
                    raise
                self.model = self._load_fn(self.model_name)
        if self.model is None:
            raise RuntimeError("AudioFlamingo3Adapter failed to initialize its model")
        if hasattr(self.model, "eval"):
            self.model.eval()
        self._loaded = True

    @staticmethod
    def _audio_path(view: ViewRecord) -> str:
        payload = view.payload
        if isinstance(payload, Mapping):
            audio = payload.get("audio", payload.get("audio_path"))
        else:
            audio = payload
        if audio is None:
            raise ValueError(
                f"view {view.name} has no audio payload; AudioFlamingo3Adapter "
                "does not score text-only inputs"
            )
        if hasattr(audio, "path") and not isinstance(audio, (str, Path)):
            audio = getattr(audio, "path")
        if not isinstance(audio, (str, Path)):
            raise ValueError(
                "AudioFlamingo3Adapter requires an audio/audio_path WAV or other "
                "file path; numeric waveforms are not accepted by the official Sound API"
            )
        path = Path(audio).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"AudioFlamingo3Adapter input audio does not exist: {path}")
        return str(path)

    @staticmethod
    def _to_half_list(value: Any, torch: Any) -> list[Any]:
        """Flatten official batch/mask tensors into one entry per audio chunk."""

        if isinstance(value, torch.Tensor):
            return [item.half() for item in torch.unbind(value, dim=0)]
        if isinstance(value, (list, tuple)):
            result: list[Any] = []
            for item in value:
                if isinstance(item, torch.Tensor):
                    result.extend([chunk.half() for chunk in torch.unbind(item, dim=0)])
                else:
                    result.append(item)
            return result
        return []

    @staticmethod
    def _has_nonzero_mask(masks: Sequence[Any], torch: Any) -> bool:
        for mask in masks:
            if mask is None:
                continue
            try:
                if bool(torch.as_tensor(mask).sum().item() > 0):
                    return True
            except (TypeError, ValueError, RuntimeError):
                continue
        return False

    def _model_device(self) -> Any:
        torch = self._torch
        if torch is None:
            raise RuntimeError("AudioFlamingo3Adapter has not initialized torch")
        model = self.model
        model_device = getattr(model, "device", None)
        if model_device is not None and str(model_device) != "meta":
            return torch.device(model_device)
        if model is not None:
            try:
                return next(model.parameters()).device
            except (AttributeError, StopIteration):
                pass
        if self.device_name != "auto":
            return torch.device(self.device_name)
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _prepare(self, view: ViewRecord, question: str) -> tuple[Any, dict[str, Any], dict[str, Any]]:
        self._load()
        torch = self._torch
        assert torch is not None
        assert self.model is not None
        assert self._Sound is not None
        assert self._extract_media is not None
        assert self._process_sounds is not None
        assert self._process_sound_masks is not None
        assert self._tokenize_conversation is not None

        audio_path = self._audio_path(view)
        conversation: list[dict[str, Any]] = [
            {"from": "human", "value": [self._Sound(audio_path), question]}
        ]
        media, media_meta = self._extract_media(conversation, getattr(self.model, "config", None))
        sounds = media.get("sound") if isinstance(media, Mapping) else None
        if not sounds or all(sound is None for sound in sounds):
            raise RuntimeError(
                f"AudioFlamingo3Adapter extracted no usable audio for view {view.name!r}"
            )

        batch, sound_mask = self._process_sounds(sounds, inference=True)
        if batch is None or int(batch.shape[0]) <= 0:
            raise RuntimeError(
                "AudioFlamingo3Adapter process_sounds produced no encoded audio; "
                "refusing text-only scoring"
            )
        if sound_mask is not None and hasattr(sound_mask, "any") and not bool(sound_mask.any().item()):
            raise RuntimeError("AudioFlamingo3Adapter audio mask contains no usable chunks")

        sfm_raw = self._process_sound_masks(
            media_meta.get("sound_feature_masks", [None] * int(batch.shape[0]))
        )
        sem_raw = self._process_sound_masks(
            media_meta.get("sound_embed_masks", [None] * int(batch.shape[0]))
        )
        sfm = self._to_half_list(sfm_raw, torch)
        sem = self._to_half_list(sem_raw, torch)
        if not self._has_nonzero_mask(sem, torch) or not self._has_nonzero_mask(sfm, torch):
            raise RuntimeError(
                "AudioFlamingo3Adapter produced empty audio feature/embed masks; "
                "refusing text-only scoring"
            )
        if len(sem) != int(batch.shape[0]) or len(sfm) != int(batch.shape[0]):
            raise RuntimeError(
                "AudioFlamingo3Adapter media and mask chunk counts do not align: "
                f"batch={int(batch.shape[0])}, feature_masks={len(sfm)}, embed_masks={len(sem)}"
            )

        media = dict(media)
        media["sound"] = [item.half() for item in torch.unbind(batch, dim=0)]
        media_meta = dict(media_meta)
        media_meta["sound_feature_masks"] = sfm
        media_meta["sound_embed_masks"] = sem
        input_ids = self._tokenize_conversation(
            conversation, self.model.tokenizer, add_generation_prompt=True
        )
        if not hasattr(input_ids, "to"):
            input_ids = torch.as_tensor(input_ids)
        input_ids = input_ids.to(self._model_device()).unsqueeze(0)
        token_ids = getattr(self.model.tokenizer, "media_token_ids", {})
        sound_token_id = token_ids.get("sound") if isinstance(token_ids, Mapping) else None
        if sound_token_id is None or int((input_ids == int(sound_token_id)).sum().item()) <= 0:
            raise RuntimeError(
                "AudioFlamingo3 tokenizer input contains no <sound> token; refusing "
                "to score an unencoded/text-only prompt"
            )
        return input_ids, media, media_meta

    def _forward(self, input_ids: Any, media: Mapping[str, Any], media_meta: Mapping[str, Any]) -> Any:
        assert self.model is not None and self._torch is not None
        from collections import defaultdict

        kwargs = {
            "input_ids": input_ids,
            "media": media,
            "media_config": defaultdict(dict),
            "media_meta": media_meta,
            "return_dict": True,
            "use_cache": False,
        }
        with self._torch.inference_mode():
            try:
                outputs = self.model(**kwargs)
            except TypeError as exc:
                if "use_cache" not in str(exc) and "return_dict" not in str(exc):
                    raise
                kwargs.pop("use_cache", None)
                outputs = self.model(**kwargs)
        logits = getattr(outputs, "logits", None)
        if logits is None or getattr(logits, "ndim", 0) != 3:
            raise RuntimeError(
                "AudioFlamingo3 forward did not return [batch, sequence, vocab] logits; "
                f"got {getattr(logits, 'shape', None)!r}"
            )
        return logits

    def _forward_batch(
        self,
        input_ids: Any,
        attention_mask: Any,
        media: Mapping[str, Any],
        media_meta: Mapping[str, Any],
    ) -> Any:
        """Run one official AF3 multimodal forward for a padded request batch.

        AF3's ``_embed`` path accepts a flattened list of audio chunks and a
        padded text batch.  The media order is the same as the ``<sound>``
        token order, so one forward is equivalent to independent forwards but
        amortizes language-model and kernel launch overhead.
        """

        assert self.model is not None and self._torch is not None
        from collections import defaultdict

        kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "media": media,
            "media_config": defaultdict(dict),
            "media_meta": media_meta,
            "return_dict": True,
            "use_cache": False,
        }
        with self._torch.inference_mode():
            try:
                outputs = self.model(**kwargs)
            except TypeError as exc:
                if "use_cache" not in str(exc) and "return_dict" not in str(exc):
                    raise
                kwargs.pop("use_cache", None)
                outputs = self.model(**kwargs)
        logits = getattr(outputs, "logits", None)
        if logits is None or getattr(logits, "ndim", 0) != 3:
            raise RuntimeError(
                "AudioFlamingo3 batched forward did not return [batch, sequence, vocab] logits; "
                f"got {getattr(logits, 'shape', None)!r}"
            )
        if int(logits.shape[0]) != int(input_ids.shape[0]):
            raise RuntimeError(
                "AudioFlamingo3 batched forward changed the request batch size: "
                f"input={int(input_ids.shape[0])}, logits={int(logits.shape[0])}"
            )
        return logits

    def _label_id(self, label: str) -> int:
        assert self.model is not None
        label = str(label)
        if label in self._label_id_cache:
            return self._label_id_cache[label]
        encoded = self.model.tokenizer(label, add_special_tokens=False)
        ids = encoded.get("input_ids", encoded) if isinstance(encoded, Mapping) else encoded
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        while isinstance(ids, list) and ids and isinstance(ids[0], list):
            ids = ids[0]
        if not isinstance(ids, Sequence) or len(ids) != 1:
            count = len(ids) if isinstance(ids, Sequence) else "unknown"
            raise ValueError(
                f"answer label {label!r} tokenizes to {count} tokens; "
                "QSAEC requires normalized single-token labels"
            )
        token_id = int(ids[0])
        self._label_id_cache[label] = token_id
        return token_id

    def _prepare_batch(
        self,
        requests: Sequence[tuple[ViewRecord, Sequence[str], str | None]],
    ) -> tuple[Any, Any, dict[str, Any], dict[str, Any], list[int]]:
        """Prepare a batch while preserving the official AF3 media protocol.

        ``extract_media`` remains per conversation because it owns file
        decoding, windowing, and feature/embed masks.  The resulting raw
        sounds are then handed to *one* official ``process_sounds`` call and
        masks are flattened in that same order.  No text-only fallback is
        possible: every request must contain a usable sound and a sound media
        token before the batch is forwarded.
        """

        self._load()
        torch = self._torch
        assert torch is not None
        assert self.model is not None
        assert self._Sound is not None
        assert self._extract_media is not None
        assert self._process_sounds is not None
        assert self._process_sound_masks is not None
        assert self._tokenize_conversation is not None

        if not requests:
            raise ValueError("AudioFlamingo3Adapter score_logprobs_batch requires at least one request")

        conversations: list[list[dict[str, Any]]] = []
        raw_sounds: list[Any] = []
        raw_feature_masks: list[Any] = []
        raw_embed_masks: list[Any] = []
        tokenized: list[Any] = []
        row_chunk_counts: list[int] = []

        for view, _labels, question in requests:
            prompt = question if question is not None else view.question
            if not isinstance(prompt, str) or not prompt:
                raise ValueError("AudioFlamingo3Adapter requires a non-empty question")
            audio_path = self._audio_path(view)
            conversation: list[dict[str, Any]] = [
                {"from": "human", "value": [self._Sound(audio_path), prompt]}
            ]
            media, media_meta = self._extract_media(
                conversation, getattr(self.model, "config", None)
            )
            sounds = media.get("sound") if isinstance(media, Mapping) else None
            if not sounds or all(sound is None for sound in sounds):
                raise RuntimeError(
                    f"AudioFlamingo3Adapter extracted no usable audio for view {view.name!r}"
                )
            if any(sound is None for sound in sounds):
                raise RuntimeError(
                    "AudioFlamingo3Adapter batch does not support missing sound entries; "
                    "refusing a partially encoded/text-only request"
                )

            conversations.append(conversation)
            raw_sounds.extend(sounds)
            # ``process_sounds`` splits each sound entry into one item per
            # 30-second window.  The tokenizer inserts one <sound> token per
            # such window, so retain this row boundary for final-logit lookup.
            row_chunk_counts.append(sum(len(sound) for sound in sounds))
            raw_feature_masks.extend(
                media_meta.get("sound_feature_masks", [None] * len(sounds))
            )
            raw_embed_masks.extend(
                media_meta.get("sound_embed_masks", [None] * len(sounds))
            )

        batch, sound_mask = self._process_sounds(raw_sounds, inference=True)
        if batch is None or int(batch.shape[0]) <= 0:
            raise RuntimeError(
                "AudioFlamingo3 process_sounds produced no encoded audio; refusing text-only scoring"
            )
        if sound_mask is not None and hasattr(sound_mask, "any") and not bool(sound_mask.any().item()):
            raise RuntimeError("AudioFlamingo3 audio mask contains no usable chunks")

        sfm = self._to_half_list(self._process_sound_masks(raw_feature_masks), torch)
        sem = self._to_half_list(self._process_sound_masks(raw_embed_masks), torch)
        if not self._has_nonzero_mask(sem, torch) or not self._has_nonzero_mask(sfm, torch):
            raise RuntimeError(
                "AudioFlamingo3 produced empty audio feature/embed masks; refusing text-only scoring"
            )
        if len(sem) != int(batch.shape[0]) or len(sfm) != int(batch.shape[0]):
            raise RuntimeError(
                "AudioFlamingo3 media and mask chunk counts do not align: "
                f"batch={int(batch.shape[0])}, feature_masks={len(sfm)}, embed_masks={len(sem)}"
            )

        media = {"sound": [item.half() for item in torch.unbind(batch, dim=0)]}
        media_meta = {"sound_feature_masks": sfm, "sound_embed_masks": sem}

        sound_token_id = None
        token_ids = getattr(self.model.tokenizer, "media_token_ids", {})
        if isinstance(token_ids, Mapping):
            sound_token_id = token_ids.get("sound")
        if sound_token_id is None:
            raise RuntimeError("AudioFlamingo3 tokenizer has no sound media token id")

        # Tokenize through the official helper, then pad explicitly.  This
        # retains the exact chat template while allowing variable audio-window
        # counts and text lengths in one model call.
        for conversation in conversations:
            input_ids = self._tokenize_conversation(
                conversation, self.model.tokenizer, add_generation_prompt=True
            )
            if not hasattr(input_ids, "to"):
                input_ids = torch.as_tensor(input_ids)
            input_ids = input_ids.to(self._model_device()).reshape(-1)
            if int((input_ids == int(sound_token_id)).sum().item()) <= 0:
                raise RuntimeError(
                    "AudioFlamingo3 tokenizer input contains no <sound> token; refusing "
                    "to score an unencoded/text-only prompt"
                )
            tokenized.append(input_ids)

        tokenizer = self.model.tokenizer
        pad_id = getattr(tokenizer, "pad_token_id", None)
        if pad_id is None:
            pad_id = getattr(tokenizer, "eos_token_id", None)
        if pad_id is None:
            # Tiny injected tokenizers used by tests may expose neither.  A
            # zero pad is safe because attention_mask excludes it from scoring.
            pad_id = 0
        max_length = max(int(ids.numel()) for ids in tokenized)
        batch_ids = torch.full(
            (len(tokenized), max_length), int(pad_id), dtype=tokenized[0].dtype,
            device=tokenized[0].device,
        )
        attention_mask = torch.zeros(
            (len(tokenized), max_length), dtype=torch.long, device=tokenized[0].device
        )
        left_padding = getattr(tokenizer, "padding_side", "right") == "left"
        for row, ids in enumerate(tokenized):
            length = int(ids.numel())
            start = max_length - length if left_padding else 0
            batch_ids[row, start : start + length] = ids
            attention_mask[row, start : start + length] = 1
        return batch_ids, attention_mask, media, media_meta, row_chunk_counts

    def _batch_last_positions(
        self,
        input_ids: Any,
        attention_mask: Any,
        logits: Any,
        media_meta: Mapping[str, Any],
        row_chunk_counts: Sequence[int],
    ) -> list[int]:
        """Map each request to its final text position after AF3 audio insertion.

        AF3 replaces every ``<sound>`` token by a rounded number of audio
        embeddings and then pads the expanded sequences.  The input attention
        mask therefore cannot be used directly as a logits index; reconstruct
        the same lengths from the official embed masks and padding rule.
        """

        torch = self._torch
        assert torch is not None
        token_ids = getattr(self.model.tokenizer, "media_token_ids", {}) if self.model else {}
        sound_token_id = token_ids.get("sound") if isinstance(token_ids, Mapping) else None
        if sound_token_id is None:
            raise RuntimeError("AudioFlamingo3 tokenizer has no sound media token id")
        masks = list(media_meta.get("sound_embed_masks", []))
        if len(row_chunk_counts) != int(input_ids.shape[0]):
            raise RuntimeError(
                "AudioFlamingo3 batch row/media boundary mismatch: "
                f"rows={int(input_ids.shape[0])}, boundaries={len(row_chunk_counts)}"
            )
        expanded_lengths: list[int] = []
        offset = 0
        for row, chunk_count in enumerate(row_chunk_counts):
            if chunk_count <= 0 or offset + chunk_count > len(masks):
                raise RuntimeError(
                    "AudioFlamingo3 batch row has invalid audio chunk boundary: "
                    f"row={row}, chunks={chunk_count}, offset={offset}, masks={len(masks)}"
                )
            active_tokens = int(attention_mask[row].sum().item())
            placeholder_count = int((input_ids[row] == int(sound_token_id)).sum().item())
            if placeholder_count != chunk_count:
                raise RuntimeError(
                    "AudioFlamingo3 audio/token mismatch in batch row: "
                    f"row={row}, audio_chunks={chunk_count}, sound_tokens={placeholder_count}"
                )
            audio_tokens = 0
            for mask in masks[offset : offset + chunk_count]:
                count = int(torch.as_tensor(mask).sum().item())
                audio_tokens += round(count / 10.0) * 10
            if audio_tokens <= 0:
                raise RuntimeError(
                    f"AudioFlamingo3 batch row {row} has no usable audio embedding tokens"
                )
            expanded_lengths.append(active_tokens + audio_tokens - chunk_count)
            offset += chunk_count
        if offset != len(masks):
            raise RuntimeError(
                "AudioFlamingo3 batch has unused audio masks: "
                f"consumed={offset}, masks={len(masks)}"
            )

        batch_size = int(input_ids.shape[0])
        left_edge_zero = bool(torch.any(attention_mask[:, 0] == 0).item())
        right_edge_zero = bool(torch.any(attention_mask[:, -1] == 0).item())
        if batch_size <= 1:
            left_padding = True
        elif left_edge_zero and not right_edge_zero:
            left_padding = True
        elif not left_edge_zero and right_edge_zero:
            left_padding = False
        elif not left_edge_zero and not right_edge_zero:
            left_padding = getattr(self.model.tokenizer, "padding_side", "left") == "left"
        else:
            raise RuntimeError("AudioFlamingo3 batch has both left and right padding")

        max_length = int(logits.shape[1])
        if left_padding:
            return [max_length - 1] * batch_size
        return [length - 1 for length in expanded_lengths]

    def score_logprobs_batch(
        self,
        requests: Sequence[tuple[ViewRecord, Sequence[str], str | None]],
    ) -> list[LabelScores]:
        """Score multiple AF3 requests with one multimodal forward.

        Each request is ``(view, labels, question_or_none)``.  Labels may
        differ per request (QSAEC swaps A/B orientation), but all labels for a
        request are read from the same final prompt position, exactly as in
        :meth:`score_logprobs`.
        """

        self._load()
        normalized: list[tuple[ViewRecord, list[str], str | None]] = []
        token_ids_per_request: list[dict[str, int]] = []
        for item in requests:
            if len(item) != 3:
                raise ValueError(
                    "AudioFlamingo3Adapter batch requests must be (view, labels, question_or_none)"
                )
            view, labels, question = item
            labels_list = [str(label) for label in labels]
            if not labels_list:
                raise ValueError("at least one answer label is required")
            normalized.append((view, labels_list, question))
            token_ids_per_request.append({label: self._label_id(label) for label in labels_list})

        input_ids, attention_mask, media, media_meta, row_chunk_counts = self._prepare_batch(normalized)
        logits = self._forward_batch(input_ids, attention_mask, media, media_meta)
        if int(logits.shape[1]) <= 0:
            raise RuntimeError("AudioFlamingo3 batched forward returned an empty sequence")

        torch = self._torch
        assert torch is not None
        results: list[LabelScores] = []
        vocab_size = int(logits.shape[-1])
        positions = self._batch_last_positions(
            input_ids, attention_mask, logits, media_meta, row_chunk_counts
        )
        for row, token_ids in enumerate(token_ids_per_request):
            position = positions[row]
            if position < 0 or position >= int(logits.shape[1]):
                raise RuntimeError(
                    f"AudioFlamingo3 batch final position {position} is outside logits shape "
                    f"{tuple(logits.shape)}"
                )
            next_token_log_probs = torch.log_softmax(logits[row, position].float(), dim=-1)
            scores: dict[str, float] = {}
            for label, token_id in token_ids.items():
                if token_id < 0 or token_id >= vocab_size:
                    raise RuntimeError(
                        f"AudioFlamingo3 tokenizer label id {token_id} is outside vocabulary "
                        f"size {vocab_size}"
                    )
                scores[label] = float(next_token_log_probs[token_id].detach().cpu())
            results.append(scores)
        return results

    def score_logprobs(
        self,
        view: ViewRecord,
        labels: Sequence[str],
        *,
        question: str | None = None,
    ) -> LabelScores:
        self._load()
        prompt = question if question is not None else view.question
        if not isinstance(prompt, str) or not prompt:
            raise ValueError("AudioFlamingo3Adapter requires a non-empty question")
        normalized = [str(label) for label in labels]
        if not normalized:
            raise ValueError("at least one answer label is required")
        token_ids = {label: self._label_id(label) for label in normalized}

        prompt_ids, prompt_media, prompt_meta = self._prepare(view, prompt)
        prompt_logits = self._forward(prompt_ids, prompt_media, prompt_meta)
        if int(prompt_logits.shape[1]) <= 0:
            raise RuntimeError("AudioFlamingo3 prompt forward returned an empty sequence")

        torch = self._torch
        assert torch is not None
        next_token_log_probs = torch.log_softmax(prompt_logits[0, -1].float(), dim=-1)
        vocab_size = int(prompt_logits.shape[-1])
        scores: dict[str, float] = {}
        for label, token_id in token_ids.items():
            if token_id < 0 or token_id >= vocab_size:
                raise RuntimeError(
                    f"AudioFlamingo3 tokenizer label id {token_id} is outside vocabulary "
                    f"size {vocab_size}"
                )
            scores[label] = float(next_token_log_probs[token_id].detach().cpu())
        return scores


class AudioFlamingo3HFAdapter(ScoringAdapter):
    """Answer-only scorer for the Hugging Face Audio Flamingo 3 checkpoint.

    This path deliberately matches the already-frozen AF3 prompt experiment:
    ``AutoProcessor.apply_chat_template`` encodes one text-plus-audio user turn,
    and ``AudioFlamingo3ForConditionalGeneration`` supplies the next-token
    logits used to score the normalized A/B labels.  Model and processor
    injection keep the adapter testable without importing the optional AF3
    runtime or allocating a GPU.
    """

    def __init__(
        self,
        model_name: str,
        *,
        device: str = "cuda",
        dtype: str = "bfloat16",
        model: Any | None = None,
        processor: Any | None = None,
    ) -> None:
        if (model is None) != (processor is None):
            raise ValueError("inject both model and processor, or neither")
        self.model_name = model_name
        self.device_name = device
        self.dtype_name = dtype
        self.model = model
        self.processor = processor
        self._torch: Any | None = None
        self._label_id_cache: dict[str, int] = {}
        self._loaded = model is not None

    @staticmethod
    def _dependency_error(detail: str) -> RuntimeError:
        return RuntimeError(
            "AudioFlamingo3HFAdapter requires a transformers build exposing "
            "AudioFlamingo3ForConditionalGeneration and AutoProcessor, plus "
            "torch and the checkpoint's audio dependencies. "
            f"Import detail: {detail}"
        )

    def _load(self) -> None:
        if self._loaded and self._torch is not None:
            return
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - optional runtime path
            raise self._dependency_error(str(exc)) from exc
        self._torch = torch
        if self._loaded:
            return
        try:
            from transformers import AudioFlamingo3ForConditionalGeneration, AutoProcessor
        except (ImportError, ModuleNotFoundError) as exc:  # pragma: no cover
            raise self._dependency_error(str(exc)) from exc

        dtype = getattr(torch, self.dtype_name, None)
        if dtype is None:
            raise ValueError(f"unknown torch dtype {self.dtype_name!r}")
        device_map: Any
        if self.device_name in {"auto", "cuda"}:
            device_map = "auto"
        else:
            device_map = {"": self.device_name}
        self.processor = AutoProcessor.from_pretrained(
            self.model_name, local_files_only=True
        )
        self.model = AudioFlamingo3ForConditionalGeneration.from_pretrained(
            self.model_name,
            dtype=dtype,
            device_map=device_map,
            local_files_only=True,
            attn_implementation="sdpa",
        )
        if hasattr(self.model, "eval"):
            self.model.eval()
        self._loaded = True

    @staticmethod
    def _audio_path(view: ViewRecord) -> str:
        payload = view.payload
        if isinstance(payload, Mapping):
            audio = payload.get("audio", payload.get("audio_path"))
        else:
            audio = payload
        if audio is None:
            raise ValueError(
                f"view {view.name} has no audio payload; AudioFlamingo3HFAdapter "
                "does not score text-only inputs"
            )
        if not isinstance(audio, (str, Path)):
            raise ValueError(
                "AudioFlamingo3HFAdapter requires an audio/audio_path file path"
            )
        path = Path(audio).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"AudioFlamingo3HFAdapter input audio does not exist: {path}")
        return str(path)

    def _model_device(self) -> Any:
        assert self._torch is not None and self.model is not None
        device = getattr(self.model, "device", None)
        if device is not None and str(device) != "meta":
            return device
        try:
            return next(self.model.parameters()).device
        except (AttributeError, StopIteration):
            return self._torch.device("cpu" if self.device_name == "auto" else self.device_name)

    def _model_dtype(self) -> Any:
        assert self._torch is not None and self.model is not None
        try:
            return next(self.model.parameters()).dtype
        except (AttributeError, StopIteration):
            dtype = getattr(self._torch, self.dtype_name, None)
            if dtype is None:
                raise ValueError(f"unknown torch dtype {self.dtype_name!r}")
            return dtype

    def _label_id(self, label: str) -> int:
        assert self.processor is not None
        label = str(label)
        if label in self._label_id_cache:
            return self._label_id_cache[label]
        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is None:
            raise RuntimeError("AudioFlamingo3HFAdapter processor did not expose a tokenizer")
        encoded = tokenizer(label, add_special_tokens=False)
        ids = encoded.get("input_ids", encoded) if isinstance(encoded, Mapping) else encoded
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        while isinstance(ids, list) and ids and isinstance(ids[0], list):
            ids = ids[0]
        if not isinstance(ids, Sequence) or len(ids) != 1:
            count = len(ids) if isinstance(ids, Sequence) else "unknown"
            raise ValueError(
                f"answer label {label!r} tokenizes to {count} tokens; "
                "QSAEC requires normalized single-token labels"
            )
        token_id = int(ids[0])
        self._label_id_cache[label] = token_id
        return token_id

    def score_logprobs(
        self,
        view: ViewRecord,
        labels: Sequence[str],
        *,
        question: str | None = None,
    ) -> LabelScores:
        self._load()
        assert self._torch is not None and self.model is not None and self.processor is not None
        prompt = question if question is not None else view.question
        if not isinstance(prompt, str) or not prompt:
            raise ValueError("AudioFlamingo3HFAdapter requires a non-empty question")
        normalized = [str(label) for label in labels]
        if not normalized:
            raise ValueError("at least one answer label is required")
        token_ids = {label: self._label_id(label) for label in normalized}
        audio_path = self._audio_path(view)
        conversation = [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "audio", "path": audio_path},
            ],
        }]
        inputs = self.processor.apply_chat_template(
            conversation,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        if hasattr(inputs, "to"):
            inputs = inputs.to(self._model_device())
        if not isinstance(inputs, Mapping):
            raise RuntimeError("AudioFlamingo3HFAdapter processor did not return a mapping")
        audio_keys = {
            key for key in inputs
            if key in {"input_features", "input_values", "audio_values", "audio_features"}
            or "audio" in key
        }
        if not audio_keys:
            raise RuntimeError(
                "AudioFlamingo3HFAdapter processor produced no audio tensor; "
                f"processor keys were {sorted(inputs.keys())}"
            )
        device = self._model_device()
        model_dtype = self._model_dtype()
        moved: dict[str, Any] = {}
        for key, value in inputs.items():
            if hasattr(value, "to"):
                value = value.to(device)
                if self._torch.is_tensor(value) and self._torch.is_floating_point(value):
                    value = value.to(dtype=model_dtype)
            moved[key] = value
        with self._torch.inference_mode():
            outputs = self.model(**moved, return_dict=True)
            logits = getattr(outputs, "logits", None)
            if logits is None or getattr(logits, "ndim", 0) != 3:
                raise RuntimeError(
                    "AudioFlamingo3HFAdapter forward did not return "
                    f"[batch, sequence, vocab] logits; got {getattr(logits, 'shape', None)!r}"
                )
            attention = moved.get("attention_mask")
            position = (
                int(attention[0].sum().item()) - 1
                if attention is not None
                else int(logits.shape[1]) - 1
            )
            if position < 0 or position >= int(logits.shape[1]):
                raise RuntimeError(
                    f"invalid AF3 answer position {position} for logits shape {tuple(logits.shape)}"
                )
            log_probs = self._torch.log_softmax(logits[0, position].float(), dim=-1)
        vocab_size = int(logits.shape[-1])
        scores: dict[str, float] = {}
        for label, token_id in token_ids.items():
            if token_id < 0 or token_id >= vocab_size:
                raise RuntimeError(
                    f"AudioFlamingo3HFAdapter label id {token_id} is outside vocabulary "
                    f"size {vocab_size}"
                )
            scores[label] = float(log_probs[token_id].detach().cpu())
        return scores


# Spelling variants used by launch scripts and model registries.
AF3Adapter = AudioFlamingo3Adapter
AudioFlamingoAdapter = AudioFlamingo3Adapter
