#ifndef GUFO_MODELS_OMNIVOICE_AUDIO_HPP_
#define GUFO_MODELS_OMNIVOICE_AUDIO_HPP_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <vector>

// Waveform helpers of OmniVoice's reference pipeline: the reference voice's
// preparation and the generated audio's post-processing. Silence handling
// follows pydub, which the reference uses, including its PCM16 round trip.
namespace gufo::models::omnivoice {

/// Mean of the channels of interleaved samples.
[[nodiscard]] std::vector<float> Downmix(std::span<const float> interleaved,
                                         std::uint32_t channels);

/// torchaudio.functional.resample with its defaults: Hann-windowed sinc,
/// six zero crossings, rolloff 0.99.
[[nodiscard]] std::vector<float> Resample(std::span<const float> samples,
                                          std::uint32_t from_rate,
                                          std::uint32_t to_rate);

/// Root mean square.
[[nodiscard]] double Rms(std::span<const float> samples);

/// omnivoice.utils.audio.remove_silence: quantizes to PCM16, drops silences
/// (below -50 dBFS) longer than `middle_ms` down to `middle_ms`, keeps at
/// most `leading_ms` / `trailing_ms` at the edges.
[[nodiscard]] std::vector<float> RemoveSilence(std::span<const float> samples,
                                               std::uint32_t rate,
                                               int middle_ms, int leading_ms,
                                               int trailing_ms);

/// omnivoice.utils.audio.fade_and_pad_audio.
[[nodiscard]] std::vector<float> FadeAndPad(std::span<const float> samples,
                                            std::uint32_t rate,
                                            double pad_seconds = 0.1,
                                            double fade_seconds = 0.1);

/// omnivoice.utils.audio.cross_fade_chunks: fades each boundary out and in
/// over a third of `gap_seconds` and puts a third of silence between.
[[nodiscard]] std::vector<float> CrossFadeChunks(
    const std::vector<std::vector<float>>& chunks, std::uint32_t rate,
    double gap_seconds = 0.3);

/// A reference voice ready for the audio tokenizer.
struct ReferenceAudio {
  /// Mono samples at the codec rate, a whole number of codec frames.
  std::vector<float> samples;
  /// RMS before level correction; the generated audio is scaled to it.
  double rms = 0.0;
};

/// `create_voice_clone_prompt` with prompt preprocessing: downmix, resample
/// to `rate`, raise quiet voices to RMS 0.1, remove silences and cut the
/// tail to whole frames of `frame_samples`.
[[nodiscard]] std::optional<ReferenceAudio> PrepareReference(
    std::span<const float> interleaved, std::uint32_t channels,
    std::uint32_t source_rate, std::uint32_t rate, std::size_t frame_samples,
    std::string* error = nullptr);

/// `_post_process_audio`: removes long silences, matches a quiet
/// reference's level or peak-normalizes to 0.5 without one, then fades and
/// pads both ends.
[[nodiscard]] std::vector<float> PostProcess(
    std::span<const float> samples, std::uint32_t rate,
    std::optional<double> reference_rms);

}  // namespace gufo::models::omnivoice

#endif  // GUFO_MODELS_OMNIVOICE_AUDIO_HPP_
