#ifndef GUFO_MODELS_OMNIVOICE_GENERATOR_HPP_
#define GUFO_MODELS_OMNIVOICE_GENERATOR_HPP_

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "config.hpp"
#include "decoding.hpp"

namespace gufo::models::omnivoice {

/// The bidirectional Qwen3 backbone with its audio embeddings and heads on
/// the GPU, and OmniVoice's iterative unmasking with classifier-free
/// guidance. Not thread-safe.
class Generator {
public:
  [[nodiscard]] static std::unique_ptr<Generator> Load(
      const Config& config, const std::filesystem::path& model_root,
      std::string* error = nullptr);
  ~Generator();
  Generator(const Generator&) = delete;
  Generator& operator=(const Generator&) = delete;

  /// Runs the unmasking loop from `input`'s masked target. `cancelled` is
  /// polled before every step.
  [[nodiscard]] bool Generate(const ConditionalInput& input,
                              const DecodingConfig& decoding,
                              std::uint64_t seed,
                              const std::function<bool()>& cancelled,
                              GenerationResult* result,
                              std::string* error = nullptr);

  /// One forward pass of the CFG batch; `logits` receives the target
  /// frames' rows, conditional then unconditional, [2 * frames][codebooks *
  /// vocab].
  [[nodiscard]] bool Forward(const ConditionalInput& input,
                             std::vector<float>* logits,
                             std::string* error = nullptr);

  /// Forward plus the scoring step: per [codebook][frame] the guided
  /// argmax and its log-probability.
  [[nodiscard]] bool Score(const ConditionalInput& input, float guidance,
                           std::vector<std::int32_t>* prediction,
                           std::vector<float>* confidence,
                           std::string* error = nullptr);

private:
  struct State;
  explicit Generator(std::unique_ptr<State> state);
  std::unique_ptr<State> state_;
};

}  // namespace gufo::models::omnivoice

#endif  // GUFO_MODELS_OMNIVOICE_GENERATOR_HPP_
