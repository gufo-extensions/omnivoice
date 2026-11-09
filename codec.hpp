#ifndef GUFO_MODELS_OMNIVOICE_CODEC_HPP_
#define GUFO_MODELS_OMNIVOICE_CODEC_HPP_

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "config.hpp"

namespace gufo::models::omnivoice {

/// The Higgs Audio v2 tokenizer on the GPU in float32: RVQ codes to 24 kHz
/// audio through the DAC decoder, and audio to codes through HuBERT, the
/// semantic and DAC encoders and the RVQ. Not thread-safe.
class Codec {
public:
  [[nodiscard]] static std::unique_ptr<Codec> Load(
      const Config& config, const std::filesystem::path& model_root,
      std::string* error = nullptr);
  ~Codec();
  Codec(const Codec&) = delete;
  Codec& operator=(const Codec&) = delete;

  /// `codes` is [codebook][frame]; yields frames * frame_samples samples.
  [[nodiscard]] bool Decode(const std::vector<std::int32_t>& codes,
                            std::size_t frames, std::vector<float>* audio,
                            std::string* error = nullptr);

  /// `samples` is 24 kHz mono, a whole number of frames; `codes` receives
  /// [codebook][frame].
  [[nodiscard]] bool Encode(const std::vector<float>& samples,
                            std::vector<std::int32_t>* codes,
                            std::size_t* frames, std::string* error = nullptr);

private:
  struct State;
  explicit Codec(std::unique_ptr<State> state);
  std::unique_ptr<State> state_;
};

}  // namespace gufo::models::omnivoice

#endif  // GUFO_MODELS_OMNIVOICE_CODEC_HPP_
