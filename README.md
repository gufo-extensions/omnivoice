# OmniVoice Gufo extension

A community [Gufo](https://github.com/gufo-org/gufo) extension for
[OmniVoice](https://github.com/k2-fsa/OmniVoice), k2-fsa's massively
multilingual zero-shot text-to-speech model, on AMD Strix Halo (`gfx1151`).

> **Status:** first release, 0.1.0. Speech synthesis runs through
> `gufo serve tts` with an automatic voice, voice design and voice cloning.
> See [quality](docs/models/omnivoice/QUALITY.md),
> [benchmarks](docs/models/omnivoice/BENCHMARKS.md) and the known limits
> below.

## Model

| Provider | Model | Files |
| --- | --- | --- |
| `omnivoice` | OmniVoice (Qwen3-0.6B backbone, Higgs Audio v2 tokenizer) | The Hugging Face directory [`k2-fsa/OmniVoice`](https://huggingface.co/k2-fsa/OmniVoice) |

OmniVoice fills in the audio codes of a whole utterance over 32 unmasking
steps with classifier-free guidance. Its backbone has Qwen3-0.6B's shape but
attends in both directions over text and audio positions. The Higgs Audio v2
tokenizer turns eight codebooks at 25 frames per second into 24 kHz audio and
encodes reference voices for cloning.

The provider loads that directory: `config.json`, `model.safetensors`,
`tokenizer.json` and `audio_tokenizer/` with float32 weights, converted to
float16 for the generator on load. GGUF packages of OmniVoice (audio.cpp's
Q8_0 file and other community conversions, each in its own layout) are not
supported.

The weights aren't part of this repository. OmniVoice's weights are published
under CC-BY-NC-4.0 and its audio tokenizer under the Boson Higgs Audio 2
Community License; read both before you download them.

## Use

```sh
MODEL=/models/OmniVoice   # the Hugging Face directory
gufo serve tts --model "$MODEL" \
  --voice anna=/voices/anna.wav --voice-text anna=/voices/anna.txt --voice-lang anna=de
```

- `GET /v1/audio/voices` lists `auto` (the model picks a voice), `voice-design`
  and every preset. Requests without `voice` use `auto`.
- **Voice design:** `instructions` takes OmniVoice's attribute items,
  separated by commas: gender (`male`, `female`), age (`child`, `teenager`,
  `young adult`, `middle-aged`, `elderly`), pitch (`very low pitch` …
  `very high pitch`), `whisper`, an English accent (`british accent`,
  `indian accent`, …) or, in Chinese, a dialect (`四川话`, …). Other text is
  rejected, as the reference implementation rejects it.
- **Cloning:** a preset (`--voice NAME=WAV` with its transcript) or
  `reference_audio` (base64 WAV) with `reference_text`. OmniVoice needs the
  transcript, so a preset needs `--voice-text` (or a `.txt` beside its WAV)
  and `voice_clone_mode: speaker_embedding_only` is rejected. A reference is
  encoded once and cached.
- `language` takes a name or code (`German`, `de`, … 646 languages). Gufo's
  server sends `english` when a request names none, so give presets their
  language with `--voice-lang`.
- `seed` selects the unmasking noise; the same seed reproduces the audio
  exactly. `greedy: true` turns the noise off. `max_new_tokens` caps the frames
  (25 per second). Other sampling fields don't apply to OmniVoice.
- Texts estimated longer than 30 s are generated in chunks of about 15 s at
  punctuation and cross-faded, as the reference does.
- Streams (`response_format: pcm`, SSE) send the utterance's 0.1 s of leading
  silence at once and the rest when the utterance is generated. Gufo's
  WebSocket endpoint can split the text by sentence or clause
  (`split_granularity`) and then streams segment by segment; that path hasn't
  been tested with OmniVoice yet.

## Known limits

- Only the Hugging Face directory loads; GGUF packages, quantized ones
  included, don't.
- Short sentences are about as fast as on audio.cpp: with few rows per pass
  the generator's GEMMs leave most of the GPU idle. Longer texts and cloning
  are 1.7–2.1 times faster (see benchmarks).
- One request runs at a time; Gufo queues the others.
- The decoding parameters are OmniVoice's defaults (32 steps, guidance 2.0,
  position temperature 5.0, greedy classes); Gufo's API has no fields for
  them, the speaking rate or a fixed duration.
- Free-text voice descriptions aren't understood, so Gufo's functional TTS
  suite fails its voice-design variant (see quality).
- No automatic transcription of references and no text normalization of
  numbers (both optional in the reference).
- The sampled output can't match the reference's for the same seed: the noise
  comes from a different generator.
- `gufo speak` isn't supported; use `gufo serve tts`.
- Gufo's server reports invalid requests to the runner (such as unknown
  voice-design items) as HTTP 500 and names models `owned_by:
  operator-supplied-qwen`.
- Tested on Strix Halo (`gfx1151`) only. The server needs about 2.5 GiB of GPU
  memory.

## Responsible use

Voice cloning produces realistic speech in another person's voice. Clone
only voices you are allowed to use, and don't use the extension for
impersonation, fraud or disinformation; OmniVoice's developers prohibit
unauthorized voice cloning and such uses of the model
([disclaimer](https://github.com/k2-fsa/OmniVoice/blob/08be0b4ccbac3e13e374e86fbfead4b4cac343e2/README.md#disclaimer)).
The extension adds no watermark or other marking to the audio; laws on
synthetic media, such as labeling duties, may apply where you publish it.

## Build

Compile the extension into a Gufo source build at the revision pinned in
`flake.nix`:

```sh
cmake --preset release -DGUFO_EXTENSION_DIRS=/absolute/path/to/omnivoice
cmake --build --preset release --parallel 4
build/release/gufo diagnose --section extensions
```

With Nix, `nix build` builds the Gufo revision pinned in `flake.nix` with this
extension.

## Tests

```sh
cmake --preset gpu-test -B build/gpu-test -DGUFO_EXTENSION_DIRS=/absolute/path/to/omnivoice
cmake --build build/gpu-test --parallel 4
GUFO_OMNIVOICE_MODEL=/models/OmniVoice ctest --test-dir build/gpu-test -R '^omnivoice\.'
```

CPU tests compare the configuration, tokenizer, prompt building, waveform
helpers and decoding schedule with fixtures from the reference implementation
(`tools/reference/`); GPU tests compare the generator and the audio tokenizer
with the reference model and cover synthesis end to end. Tests labeled
`external-model` skip unless `GUFO_OMNIVOICE_MODEL` names the model directory.

## Supported Gufo revisions

| Extension | Gufo | Extension API |
| --- | --- | --- |
| 0.1.1 | `0e2de43e` | 0.2.25 |
| 0.1.0 | `5f9612c3` | 0.2.16 |

## Release history

- 0.1.1 (2026-11-10): builds against Gufo `0e2de43e`; no functional
  change.
- 0.1.0 — first release.

## License

MIT, see [LICENSE](LICENSE). Code adapted from Gufo, OmniVoice, Hugging Face
transformers, pydub and torchaudio is listed in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
