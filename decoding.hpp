#ifndef GUFO_MODELS_OMNIVOICE_DECODING_HPP_
#define GUFO_MODELS_OMNIVOICE_DECODING_HPP_

#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

// Host side of OmniVoice's iterative unmasking (_generate_iterative): the
// schedule, the conditional input layout and the per-step position choice.
namespace gufo::models::omnivoice {

/// OmniVoiceGenerationConfig's decoding parameters.
struct DecodingConfig {
  std::size_t steps = 32;
  float guidance = 2.0F;
  float t_shift = 0.1F;
  float layer_penalty = 5.0F;
  /// Gumbel noise on the position scores; 0 selects positions greedily.
  float position_temperature = 5.0F;
};

/// Codes to unmask at each step: `ceil(total * (t[s+1] - t[s]))` of the
/// shifted linear schedule in float32, the remainder at the last step.
[[nodiscard]] std::vector<std::size_t> UnmaskSchedule(std::size_t total,
                                                      std::size_t steps,
                                                      float t_shift);

/// The conditional sequence: [text tokens | reference frames | target
/// frames], `codebooks` rows of `length` IDs (text positions repeat their
/// token in every row; target frames start as the mask).
struct ConditionalInput {
  std::size_t codebooks = 0;
  std::size_t length = 0;
  /// First audio position: the number of text tokens.
  std::size_t audio_start = 0;
  std::size_t target_frames = 0;
  /// [codebook][position].
  std::vector<std::int32_t> ids;

  /// `reference_codes` is [codebook][frame].
  [[nodiscard]] static ConditionalInput Build(
      const std::vector<std::uint32_t>& text_ids,
      const std::vector<std::int32_t>& reference_codes,
      std::size_t reference_frames, std::size_t target_frames,
      std::size_t codebooks, std::uint32_t mask_id);
};

struct GenerationResult {
  /// [codebook][frame].
  std::vector<std::int32_t> codes;
  /// The step that unmasked each code, [codebook][frame].
  std::vector<std::int32_t> unmasked_at;
};

/// Chooses which masked codes a step fills: scores are the guided
/// confidences minus `layer_penalty` per codebook index, divided by the
/// position temperature plus Gumbel noise when it is positive; the `count`
/// best masked positions take their predictions.
class PositionSelector {
public:
  PositionSelector(const DecodingConfig& config, std::uint64_t seed);

  /// `codes`, `prediction` and `confidence` are [codebook][frame]; returns
  /// the flat indices it filled.
  std::vector<std::size_t> Fill(std::size_t count, std::size_t codebooks,
                                std::size_t frames,
                                const std::vector<std::int32_t>& prediction,
                                const std::vector<float>& confidence,
                                std::uint32_t mask_id,
                                std::vector<std::int32_t>* codes);

private:
  float layer_penalty_;
  float temperature_;
  std::mt19937_64 random_;
};

}  // namespace gufo::models::omnivoice

#endif  // GUFO_MODELS_OMNIVOICE_DECODING_HPP_
