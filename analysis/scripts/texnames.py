r"""Shared helper: model identifiers and digit-free TeX macro names.

TeX control sequences consist of letters only, so a macro named
`\MatchKeOmniR3bTPR` is parsed as `\MatchKeOmniR` followed by the literal text
`3bTPR`. Depending on what follows, that either raises a confusing error far from
the real cause or, worse, silently prints the wrong thing. Every macro name
emitted by the build scripts must therefore be spelled with letters only, which
is the convention the main panel already uses (`AudioFlamingoThree`,
`StepAudioTwoMini`).

`macro_name` is the single place that enforces this: it maps a model id to its
canonical letter-only stem and asserts that no digit survives. Use it for any
new per-model macro.
"""
from __future__ import annotations

import re

# canonical letter-only stems, matching scripts/build_revision_numbers.py
SAFE_MODEL = {
    'audio_flamingo_3': 'AudioFlamingoThree',
    'ke_omni_r_3b': 'KeOmniRThreeB',
    'ke_omni_r_7b': 'KeOmniRSevenB',
    'mimo_audio_7b': 'MimoAudioSevenB',
    'qwen2_audio_7b': 'QwenTwoAudio',
    'qwen25_omni_3b': 'QwenTwoFiveOmniThreeB',
    'qwen25_omni_7b': 'QwenTwoFiveOmniSevenB',
    'qwen3_omni_30b': 'QwenThreeOmni',
    'qwen_audio_aha_7b': 'QwenAudioAha',
    'step_audio2_mini': 'StepAudioTwoMini',
}

PAPER_MODELS = ['audio_flamingo_3', 'mimo_audio_7b', 'qwen25_omni_3b',
                'qwen25_omni_7b', 'qwen2_audio_7b', 'qwen3_omni_30b',
                'qwen_audio_aha_7b', 'step_audio2_mini']
HELD_OUT_MODELS = ['ke_omni_r_3b', 'ke_omni_r_7b']
ALL_MODELS = PAPER_MODELS + HELD_OUT_MODELS

_DIGITS = {'0': 'Zero', '1': 'One', '2': 'Two', '3': 'Three', '4': 'Four',
           '5': 'Five', '6': 'Six', '7': 'Seven', '8': 'Eight', '9': 'Nine'}


def model_stem(model_id: str) -> str:
    """Letter-only CamelCase stem for a model id."""
    if model_id in SAFE_MODEL:
        return SAFE_MODEL[model_id]
    stem = ''.join(w.capitalize() for w in re.split(r'[_\s]+', model_id) if w)
    for d, word in _DIGITS.items():
        stem = stem.replace(d, word)
    assert not any(c.isdigit() for c in stem), stem
    return stem


def assert_letter_only(name: str, context: str = '') -> str:
    """Every macro name handed to write_macros passes through here."""
    if any(c.isdigit() for c in name):
        raise ValueError(
            f'macro name {name!r} contains a digit; TeX reads only letters in a '
            f'control sequence, so this would be parsed as a shorter macro name'
            + (f' ({context})' if context else ''))
    return name


def write_macros(path, macros: dict, header: str) -> None:
    """Write a generated .tex macro file, rejecting digit-bearing names."""
    for k in macros:
        assert_letter_only(k, context=str(path))
    body = '\n'.join(f'\\newcommand{{\\{k}}}{{{v}}}' for k, v in macros.items())
    with open(path, 'w') as f:
        f.write(f'% {header}\n{body}\n')


def pct(x) -> str:
    """Percent with a LaTeX-escaped sign, or n/a for a missing value."""
    return 'n/a' if x is None else f'{100 * x:.1f}\\%'
