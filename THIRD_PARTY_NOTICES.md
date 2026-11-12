# Third-party notices

This extension contains code adapted from the projects below. Their license
texts are in [licenses/](licenses/) and are installed with the extension.

| Project | Used for | License | Revision | Source |
| --- | --- | --- | --- | --- |
| Gufo | Safetensors reader (`safetensors.cpp`), parts of the tokenizer (`tokenizer.cpp`) and UTF-8 helpers (`unicode.cpp`), adapted from Qwen3-TTS; the hipBLASLt plan cache (`gemm.cpp`), adapted from Qwen3-ASR | `MIT` | `5f9612c3b19ab6676b842095bc88bb9b32da5cf0` | [gufo-org/gufo](https://github.com/gufo-org/gufo) |
| OmniVoice | Prompt preprocessing (`text.cpp`, `prompt.cpp`), reference and output audio processing (`audio.cpp`), the unmasking schedule and position choice (`decoding.cpp`), generation with chunking (`synthesizer.cpp`) and the language table (`language_table.inc`, generated from `omnivoice/utils/lang_map.py`) | `Apache-2.0` | `08be0b4` | [k2-fsa/OmniVoice](https://github.com/k2-fsa/OmniVoice) |
| Hugging Face transformers | The audio tokenizer's encoder and decoder (`codec*.cpp`, `kernels/codec_kernels.hip`, `kernels/encoder_kernels.hip`) port its HiggsAudioV2Tokenizer, DAC and HuBERT modules; the generator (`generator.cpp`) computes its Qwen3 layers as transformers does | `Apache-2.0` | `5.22.2` | [huggingface/transformers](https://github.com/huggingface/transformers) |
| pydub | Silence detection and removal reproduced in `audio.cpp` | `MIT` | `0.25.1` | [jiaaro/pydub](https://github.com/jiaaro/pydub) |
| torchaudio | Sinc resampler reproduced in `audio.cpp` | `BSD-2-Clause` | `2.11.1` | [pytorch/audio](https://github.com/pytorch/audio) |

| License text | Projects |
| --- | --- |
| [licenses/gufo.txt](licenses/gufo.txt) | Gufo |
| [licenses/OmniVoice.txt](licenses/OmniVoice.txt) | OmniVoice |
| [licenses/transformers.txt](licenses/transformers.txt) | Hugging Face transformers |
| [licenses/pydub.txt](licenses/pydub.txt) | pydub |
| [licenses/torchaudio.txt](licenses/torchaudio.txt) | torchaudio |

The ported files keep their sources' copyright notices:

- OmniVoice: Copyright 2026 Xiaomi Corp. (authors: Han Zhu).
- transformers' Higgs Audio v2 tokenizer: Copyright 2025 Boson AI and The
  HuggingFace Team.
- transformers' DAC: Copyright 2024 Descript and The HuggingFace Inc. team.
- transformers' HuBERT: Copyright 2021 The Fairseq Authors and the
  HuggingFace Inc. team.
- transformers' Qwen3: Copyright 2025 The Qwen team, Alibaba Group and the
  HuggingFace Inc. team.

The test fixtures in `tests/fixtures/` hold token IDs, logit statistics,
codes and waveform samples that `tools/reference/` computed with OmniVoice's
reference implementation and its released checkpoint. They contain no
weights.
