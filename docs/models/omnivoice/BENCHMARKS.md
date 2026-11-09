# OmniVoice benchmarks

Measured November 9, 2026, on Strix Halo (`gfx1151`, 128 GB) with extension
0.1.0 in a release build of Gufo `5f9612c3` (ROCm 10.1). The reference is the
audio.cpp server v0.9.4 (`f7ac446f`) on Vulkan with its OmniVoice Q8_0 GGUF,
the configuration it serves OmniVoice with by default. Both servers ran alone
under a machine reservation; power profile `performance`, package power
85 W median (113 W peak for Gufo, 108 W for audio.cpp).

Gufo loads the official float32 weights and runs the generator with float16
operands and the audio tokenizer in float32; audio.cpp runs Q8_0 linear
layers. Same requests on both: a short sentence ("The little brown dog ran
across the garden.") and the 37-word paragraph of Gufo's TTS benchmarks,
seed 42, English, OmniVoice's default decoding (32 steps, guidance 2.0):

> When the first light reaches the harbor, speak gently, with quiet confidence
> and a warm, resonant tone. The boats are still resting beside the old wooden
> pier, and the town is slowly waking to another bright morning.

## Loading

One cold-file-cache launch per server until it answers its readiness route
(`/ready`, audio.cpp `/health`), and its memory after the requests below:

| Server | Ready | First request (short sentence) | Resident memory | GPU memory |
| --- | ---: | ---: | ---: | ---: |
| Gufo | 1.92 s | 0.99 s | 393 MiB | 2,593 MiB |
| audio.cpp | 4.84 s | 0.85 s | 363 MiB | 4,835 MiB |

audio.cpp's configuration registers OmniVoice twice, as an offline and a
streaming model, and loads both at startup; its GPU memory includes both.
Gufo's first request of a new size includes
timing hipBLASLt's candidate kernels for its row bucket (0.1–0.6 s); later
requests of similar length reuse the choice.

## Single request

One warmup, then the median of three buffered WAV requests. RTF = request
time / generated audio duration, lower is better. First and complete PCM are
one SSE request each.

| Variant | Audio | Gufo WAV | audio.cpp WAV | Gain | Gufo RTF | audio.cpp RTF | First PCM | Complete PCM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Short sentence, automatic voice | 2.80 s | 0.829 s | 0.848 s | 1.02× | 0.296 | 0.303 | 0.001 s | 0.831 s |
| Paragraph, automatic voice | 13.24 s | 1.858 s | 3.183 s | 1.71× | 0.140 | 0.240 | 0.002 s | 1.866 s |
| Paragraph, voice design | 13.24 s | 1.876 s | 3.107 s | 1.66× | 0.142 | 0.235 | 0.001 s | 1.884 s |
| Paragraph, clone of a 9 s reference | 16.26 s | 3.402 s | 6.853 s | 2.10× per audio second | 0.209 | 0.438 | 0.003 s | 3.408 s |

- Voice design uses `female, low pitch, british accent`; cloning uses Gufo's
  functional-test voice (`clear-english-voice.wav`, 9 s) with its transcript,
  as a preset on Gufo and as a voice file on audio.cpp.
- The clone's audio is 16.26 s on Gufo and 15.66 s on audio.cpp: the
  reference preparation leaves different frame counts, and the duration is
  estimated from them. Gufo's preparation matches the reference
  implementation (see quality); the gain compares RTF.
- Every Gufo request repeated its seeded audio exactly, and streamed PCM
  equaled the WAV's samples. audio.cpp repeated its WAVs as well.
- Short requests are no faster than audio.cpp's: with about 150 rows per
  pass the linear layers' GEMMs are small, and hipBLASLt runs them at a
  fraction of the GPU's throughput (about 8 TFLOPS). Long and cloned
  requests have enough rows to use it well.
- Gufo streams send the utterance's 0.1 s of leading silence at once, so the
  first PCM arrives within 3 ms; the rest follows when the utterance is
  generated. audio.cpp's streaming model emits one event per text chunk; this
  paragraph is one chunk, so its first audio comes with the last.

Not measured: Gufo's PCM and WebSocket transports separately from SSE,
concurrency (Gufo runs one request at a time and queues the others) and a
kernel profile.

## Reproduce

Run from a Gufo checkout with a release build that includes this extension:

```sh
OMNIVOICE_MODEL=/models/OmniVoice GUFO=build/release/gufo \
  python3 /path/to/omnivoice/tools/bench/tts_bench.py gufo gufo.json
AUDIOCPP_SERVER=/path/to/audiocpp_server \
AUDIOCPP_MODEL=/models/OmniVoice-GGUF/omnivoice-q8_0.gguf \
  python3 /path/to/omnivoice/tools/bench/tts_bench.py audiocpp audiocpp.json
```

`REFERENCE` names the clone's WAV without its extension (default: the
checkout's `tests/functional/fixtures/audio/clear-english-voice`). The
results above are in [artifacts](artifacts/).
