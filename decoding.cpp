// Ported from k2-fsa/OmniVoice's _generate_iterative and _get_time_steps
// (Apache-2.0, see licenses/OmniVoice.txt).
#include "decoding.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

namespace gufo::models::omnivoice {

std::vector<std::size_t> UnmaskSchedule(std::size_t total, std::size_t steps,
                                        float t_shift) {
  // torch.linspace(0, 1, steps + 1) in float32: the first half counts up
  // from 0, the second half down from 1.
  std::vector<float> times(steps + 1);
  const float step = 1.0F / static_cast<float>(steps);
  const std::size_t halfway = (steps + 1) / 2;
  for (std::size_t index = 0; index <= steps; ++index) {
    float t = index < halfway ? step * static_cast<float>(index)
                              : 1.0F - step * static_cast<float>(steps - index);
    t = t_shift * t / (1.0F + (t_shift - 1.0F) * t);
    times[index] = t;
  }
  std::vector<std::size_t> schedule(steps, 0);
  std::size_t remaining = total;
  for (std::size_t index = 0; index < steps; ++index) {
    std::size_t count = remaining;
    if (index + 1 < steps) {
      const double delta = static_cast<double>(times[index + 1]) -
                           static_cast<double>(times[index]);
      count = std::min(static_cast<std::size_t>(
                           std::ceil(static_cast<double>(total) * delta)),
                       remaining);
    }
    schedule[index] = count;
    remaining -= count;
  }
  return schedule;
}

ConditionalInput ConditionalInput::Build(
    const std::vector<std::uint32_t>& text_ids,
    const std::vector<std::int32_t>& reference_codes,
    std::size_t reference_frames, std::size_t target_frames,
    std::size_t codebooks, std::uint32_t mask_id) {
  ConditionalInput input;
  input.codebooks = codebooks;
  input.audio_start = text_ids.size();
  input.target_frames = target_frames;
  input.length = text_ids.size() + reference_frames + target_frames;
  input.ids.assign(codebooks * input.length,
                   static_cast<std::int32_t>(mask_id));
  for (std::size_t c = 0; c < codebooks; ++c) {
    std::int32_t* row = &input.ids[c * input.length];
    for (std::size_t t = 0; t < text_ids.size(); ++t)
      row[t] = static_cast<std::int32_t>(text_ids[t]);
    for (std::size_t f = 0; f < reference_frames; ++f)
      row[text_ids.size() + f] = reference_codes[c * reference_frames + f];
  }
  return input;
}

PositionSelector::PositionSelector(const DecodingConfig& config,
                                   std::uint64_t seed)
    : layer_penalty_(config.layer_penalty),
      temperature_(config.position_temperature),
      random_(seed) {}

std::vector<std::size_t> PositionSelector::Fill(
    std::size_t count, std::size_t codebooks, std::size_t frames,
    const std::vector<std::int32_t>& prediction,
    const std::vector<float>& confidence, std::uint32_t mask_id,
    std::vector<std::int32_t>* codes) {
  const std::size_t total = codebooks * frames;
  std::vector<float> scores(total);
  for (std::size_t c = 0; c < codebooks; ++c) {
    const float penalty = static_cast<float>(c) * layer_penalty_;
    for (std::size_t f = 0; f < frames; ++f) {
      const std::size_t index = c * frames + f;
      float score = confidence[index] - penalty;
      if (temperature_ > 0.0F) {
        // _gumbel_sample on the scores, with uniform noise in [0, 1).
        const float uniform = static_cast<float>(random_() >> 40) * 0x1.0p-24F;
        const float gumbel = -std::log(-std::log(uniform + 1e-10F) + 1e-10F);
        score = score / temperature_ + gumbel;
      }
      if ((*codes)[index] != static_cast<std::int32_t>(mask_id))
        score = -std::numeric_limits<float>::infinity();
      scores[index] = score;
    }
  }
  count = std::min(count, total);
  std::vector<std::size_t> order(total);
  std::iota(order.begin(), order.end(), std::size_t{0});
  std::partial_sort(
      order.begin(), order.begin() + static_cast<std::ptrdiff_t>(count),
      order.end(), [&scores](std::size_t left, std::size_t right) {
        if (scores[left] != scores[right])
          return scores[left] > scores[right];
        return left < right;
      });
  order.resize(count);
  for (const std::size_t index : order)
    (*codes)[index] = prediction[index];
  return order;
}

}  // namespace gufo::models::omnivoice
