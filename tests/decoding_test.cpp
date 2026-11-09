#include "decoding.hpp"

#include <cstdint>
#include <fstream>
#include <gufo/ext/json.hpp>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

namespace json = gufo::json;
namespace ov = gufo::models::omnivoice;

namespace {

int failures = 0;

void Require(bool condition, const std::string& message) {
  if (!condition) {
    std::cout << "FAIL " << message << '\n';
    ++failures;
  }
}

std::vector<std::int32_t> Flatten(const json::Value& rows) {
  std::vector<std::int32_t> values;
  for (const json::Value& row : rows.items())
    for (const json::Value& value : row.items())
      values.push_back(static_cast<std::int32_t>(value.as_double()));
  return values;
}

}  // namespace

int main() {
  std::ifstream stream(GUFO_OMNIVOICE_GENERATOR_GOLDEN);
  const json::Value golden =
      json::parse(std::string((std::istreambuf_iterator<char>(stream)),
                              std::istreambuf_iterator<char>()));
  const json::Value& config = *golden.find("config");
  const auto steps = config.find("num_step")->as_size();
  const auto t_shift = static_cast<float>(config.find("t_shift")->as_double());
  for (const json::Value& entry : golden.find("generate")->items()) {
    const std::size_t frames = entry.find("target")->as_size();
    std::vector<std::size_t> expected;
    for (const json::Value& count : entry.find("schedule")->items())
      expected.push_back(count.as_size());
    Require(ov::UnmaskSchedule(8 * frames, steps, t_shift) == expected,
            "schedule for " + std::to_string(frames) + " frames");
    // The conditional input from its text IDs and reference codes.
    const std::vector<std::int32_t> ids = Flatten(*entry.find("input_ids"));
    const std::size_t length = ids.size() / 8;
    const std::size_t audio_start = entry.find("audio_start")->as_size();
    std::vector<std::uint32_t> text(
        ids.begin(), ids.begin() + static_cast<std::ptrdiff_t>(audio_start));
    std::vector<std::int32_t> reference;
    std::size_t reference_frames = 0;
    if (!entry.find("reference_codes")->is_null()) {
      reference = Flatten(*entry.find("reference_codes"));
      reference_frames = reference.size() / 8;
    }
    const auto input = ov::ConditionalInput::Build(
        text, reference, reference_frames, frames, 8, 1024);
    Require(input.length == length && input.ids == ids,
            "conditional input for " + entry.find("text")->str());
  }
  // Greedy position choice: the best masked scores after the layer penalty.
  ov::DecodingConfig decoding;
  decoding.position_temperature = 0.0F;
  ov::PositionSelector selector(decoding, 0);
  std::vector<std::int32_t> codes = {1024, 7, 1024, 1024, 1024, 1024};
  const std::vector<std::int32_t> prediction = {10, 11, 12, 13, 14, 15};
  // Two codebooks of three frames; codebook 1 pays a penalty of 5.
  const std::vector<float> confidence = {-0.5F, -0.1F, -2.0F,
                                         -0.2F, -0.3F, -9.0F};
  const auto chosen =
      selector.Fill(2, 2, 3, prediction, confidence, 1024, &codes);
  Require(chosen == std::vector<std::size_t>({0, 2}) &&
              codes == std::vector<std::int32_t>({10, 7, 12, 1024, 1024, 1024}),
          "greedy position choice");
  std::cout << (failures == 0 ? "ok   " : "FAIL ") << "decoding, " << failures
            << " failures\n";
  return failures == 0 ? 0 : 1;
}
