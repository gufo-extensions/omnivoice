# OmniVoice quality

There is no listening study yet. This page records numerical checks against
the reference implementation, intelligibility through an ASR model and
Gufo's functional TTS suite. Measured November 9, 2026, with extension 0.1.0
on Gufo `5f9612c3`, on k2-fsa/OmniVoice's Hugging Face files at `c5fdb5cc`.

## Against the reference implementation

The reference is k2-fsa/OmniVoice's Python package at `08be0b4` with
transformers 5.22.2 and torch 2.15 in float32 on the CPU. The scripts in
[tools/reference](../../../tools/reference) write the fixtures in
`tests/fixtures/`; the tests compare against them.

| Stage | Check | Result |
| --- | --- | --- |
| Tokenizer | 31 multilingual texts, IDs as Hugging Face's tokenizer | 31/31 identical |
| Prompt preprocessing | Language names (19), voice-design items (13), duration estimates (70), text cleanup (42), punctuation (9), chunking (7), prompt token layouts (6) | all identical |
| Waveform helpers | pydub silence removal (PCM16 checksums), torchaudio resampling, fades, cross-fades, reference preparation, post-processing: 45 cases | all within tolerance; silence removal exact |
| Unmasking schedule | 32-step schedules of three generations | identical |
| Generator, one CFG pass | Argmax of the guided target logits, 2 x 8 x 40 frame-codebooks | conditional 311/320, unconditional 316/320; logits within 0.03 |
| Generator, scoring | Guided predictions of 320 frame-codebooks | 314/320; confidences within 0.02 |
| Audio tokenizer, decode | 65 frames to 62,400 samples | samples within 1.0e-6 |
| Audio tokenizer, encode | 3 s synthetic signal to 8 x 75 codes | 600/600 identical |

The generator runs its GEMMs with float16 operands and float32 accumulation,
like the reference's GPU inference (`dtype=torch.float16`), with a float32
residual stream and norms. For scale: the reference itself agrees with its
float32 logits' argmax on 87–89% of these frame-codebooks in float16 and on
48% in bfloat16. Audio codes are high-entropy, so many argmaxes are near
ties.

Deterministic generations (position and class temperature 0) therefore
follow the float32 reference until a near tie and then diverge: three
generations of 46 to 65 frames first differ at steps 11, 20 and 22 of 32.
The kernels decide where: during development, with other attention kernels,
one of them matched all 392 codes. Seeded sampling can't match the reference
anyway, because its Gumbel noise comes from another generator.

## Intelligibility

Seed 42 and the default decoding; recognized by Qwen3-ASR 1.7B through
`gufo transcribe`. WER measures word errors, not voice similarity or
prosody.

| Text | Variant | Word errors |
| --- | --- | ---: |
| The 37-word paragraph of [benchmarks](BENCHMARKS.md) | Automatic voice | 0/37 |
| | Voice design (`female, low pitch, british accent`) | 0/37 |
| | Clone of Gufo's functional-test voice | 0/37 |
| A 19-word German weather report | Automatic voice | 0/19 |
| | Voice design (`male, middle-aged`) | 0/19 |

A cloned German voice from a private recording was also recognized without
errors; the recording isn't part of this repository.

## Functional API suite

Gufo's `tests/functional/model_services.py tts` with Qwen3-ASR 1.7B as the
evaluator:

- `--variant base` (the default voice is a reference preset): all eleven
  checks pass — readiness, voice discovery, synthesis and intelligibility
  (WER 0), seeded replay, PCM and SSE equal to WAV, concurrent requests,
  ignored unknown fields, rejections and recovery, stream cancellation and
  the server observing it.
- `--variant design` fails at its first request: the suite describes the voice
  in free text ("A clear English voice speaking calmly and naturally."),
  which OmniVoice's attribute vocabulary doesn't contain. The reference
  implementation rejects such instructions as well.
- `--variant custom` doesn't apply: OmniVoice has no named speakers.

## Reproduce

```sh
# Fixtures, from a Python environment with the omnivoice package:
python tools/reference/tokenizer_golden.py MODEL_DIR tests/fixtures/tokenizer_golden.json
python tools/reference/prompt_golden.py MODEL_DIR tests/fixtures/prompt_golden.json
python tools/reference/audio_golden.py tests/fixtures/audio_golden.json
python tools/reference/generator_golden.py MODEL_DIR tests/fixtures/generator_golden.json

# In a Gufo checkout built with this extension:
GUFO_OMNIVOICE_MODEL=MODEL_DIR ctest --test-dir build/gpu-test -R '^omnivoice\.'
python3 tests/functional/model_services.py tts --binary build/gpu-test/gufo \
  --model MODEL_DIR --output report --variant base \
  --asr-binary build/gpu-test/gufo --asr-model QWEN3_ASR_DIR
```
