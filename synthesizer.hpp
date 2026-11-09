#ifndef GUFO_MODELS_OMNIVOICE_SYNTHESIZER_HPP_
#define GUFO_MODELS_OMNIVOICE_SYNTHESIZER_HPP_

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "config.hpp"
#include "decoding.hpp"

namespace gufo::models::omnivoice {

/// A cloned voice: its codes and transcript as OmniVoice's voice clone
/// prompt holds them.
struct ReferenceVoice {
  /// [codebook][frame].
  std::vector<std::int32_t> codes;
  std::size_t frames = 0;
  /// The transcript, ending in punctuation.
  std::string text;
  /// RMS of the recording before level correction.
  double rms = 0.0;
};

struct SynthesisOptions {
  std::string text;
  /// A language name or ID; empty or unknown selects no language.
  std::string language;
  /// Voice-design items; empty for none.
  std::string instruct;
  /// Clone this voice; null for voice design or an automatic voice.
  const ReferenceVoice* reference = nullptr;
  /// Upper bound of generated frames per chunk; 0 for none.
  std::size_t max_frames = 0;
  std::uint64_t seed = 0;
  DecodingConfig decoding;
  bool denoise = true;
};

/// OmniVoice's generate(): text and voice to 24 kHz mono audio, with the
/// reference's chunking of long texts and its post-processing. Not
/// thread-safe.
class Synthesizer {
public:
  [[nodiscard]] static std::unique_ptr<Synthesizer> Load(
      const std::filesystem::path& model_root, std::string* error = nullptr);
  ~Synthesizer();
  Synthesizer(const Synthesizer&) = delete;
  Synthesizer& operator=(const Synthesizer&) = delete;

  [[nodiscard]] const Config& config() const;
  [[nodiscard]] std::uint32_t SampleRate() const;

  /// create_voice_clone_prompt: prepares and encodes a recording
  /// (interleaved, any rate) and punctuates its transcript.
  [[nodiscard]] bool CloneVoice(std::span<const float> interleaved,
                                std::uint32_t channels, std::uint32_t rate,
                                const std::string& transcript,
                                ReferenceVoice* voice,
                                std::string* error = nullptr);

  /// Generates, decodes and post-processes `options.text`. `codes`, when
  /// given, receives every chunk's codes frame by frame ([frame][codebook]).
  [[nodiscard]] bool Synthesize(const SynthesisOptions& options,
                                const std::function<bool()>& cancelled,
                                std::vector<float>* audio,
                                std::vector<std::int32_t>* codes = nullptr,
                                std::string* error = nullptr);

private:
  struct State;
  explicit Synthesizer(std::unique_ptr<State> state);
  std::unique_ptr<State> state_;
};

}  // namespace gufo::models::omnivoice

#endif  // GUFO_MODELS_OMNIVOICE_SYNTHESIZER_HPP_
