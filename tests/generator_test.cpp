// Compares the GPU generator with the float32 reference run of
// tools/reference/generator_golden.py.
#include "generator.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <gufo/ext/json.hpp>
#include <iostream>
#include <iterator>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

#include "config.hpp"

namespace json = gufo::json;
namespace ov = gufo::models::omnivoice;

namespace {

constexpr int kSkip = 77;
int failures = 0;

void Require(bool condition, const std::string& message) {
  if (!condition) {
    std::cout << "FAIL " << message << '\n';
    ++failures;
  }
}

std::vector<double> Flatten(const json::Value& rows) {
  std::vector<double> values;
  for (const json::Value& row : rows.items())
    for (const json::Value& value : row.items())
      values.push_back(value.as_double());
  return values;
}

ov::ConditionalInput InputFrom(const json::Value& entry) {
  ov::ConditionalInput input;
  input.codebooks = 8;
  for (const double id : Flatten(*entry.find("input_ids")))
    input.ids.push_back(static_cast<std::int32_t>(id));
  input.length = input.ids.size() / 8;
  input.audio_start = entry.find("audio_start")->as_size();
  input.target_frames = entry.find("target")->as_size();
  return input;
}

/// Per [codebook][frame] argmax, max and log-sum-exp of logits rows
/// [first, first + frames).
void Stats(const std::vector<float>& logits, std::size_t first,
           std::size_t frames, std::vector<double>* argmax,
           std::vector<double>* maximum, std::vector<double>* lse) {
  constexpr std::size_t kVocab = 1025;
  for (std::size_t c = 0; c < 8; ++c) {
    for (std::size_t f = 0; f < frames; ++f) {
      const float* row = &logits[(first + f) * 8 * kVocab + c * kVocab];
      const float* best = std::max_element(row, row + kVocab);
      double sum = 0.0;
      for (std::size_t v = 0; v < kVocab; ++v)
        sum += std::exp(static_cast<double>(row[v]) - *best);
      argmax->push_back(static_cast<double>(best - row));
      maximum->push_back(*best);
      lse->push_back(*best + std::log(sum));
    }
  }
}

void CompareLogits(std::string_view name, const std::vector<float>& logits,
                   std::size_t first, std::size_t frames,
                   const json::Value& expected) {
  std::vector<double> argmax, maximum, lse;
  Stats(logits, first, frames, &argmax, &maximum, &lse);
  const auto want_argmax = Flatten(*expected.find("argmax"));
  const auto want_max = Flatten(*expected.find("max"));
  const auto want_lse = Flatten(*expected.find("logsumexp"));
  std::size_t agree = 0;
  double max_error = 0.0, lse_error = 0.0;
  for (std::size_t i = 0; i < argmax.size(); ++i) {
    agree += argmax[i] == want_argmax[i] ? 1 : 0;
    max_error = std::max(max_error, std::abs(maximum[i] - want_max[i]));
    lse_error = std::max(lse_error, std::abs(lse[i] - want_lse[i]));
  }
  const double rate =
      static_cast<double>(agree) / static_cast<double>(argmax.size());
  std::cout << "  " << name << ": argmax " << agree << "/" << argmax.size()
            << ", max |d max| " << max_error << ", max |d logsumexp| "
            << lse_error << '\n';
  Require(rate >= 0.95, std::string(name) + " argmax agreement");
  Require(lse_error < 0.1, std::string(name) + " log-sum-exp");
}

bool TestForward(ov::Generator& generator, const json::Value& golden) {
  const json::Value& entry = *golden.find("forward");
  const ov::ConditionalInput input = InputFrom(entry);
  std::string error;
  std::vector<float> logits;
  if (!generator.Forward(input, &logits, &error)) {
    Require(false, error);
    return true;
  }
  CompareLogits("conditional", logits, 0, input.target_frames,
                *entry.find("cond"));
  CompareLogits("unconditional", logits, input.target_frames,
                input.target_frames, *entry.find("uncond"));
  std::vector<std::int32_t> prediction;
  std::vector<float> confidence;
  if (!generator.Score(input, 2.0F, &prediction, &confidence, &error)) {
    Require(false, error);
    return true;
  }
  const auto want_prediction = Flatten(*entry.find("pred"));
  const auto want_confidence = Flatten(*entry.find("confidence"));
  std::size_t agree = 0;
  double confidence_error = 0.0;
  for (std::size_t i = 0; i < prediction.size(); ++i) {
    agree +=
        prediction[i] == static_cast<std::int32_t>(want_prediction[i]) ? 1 : 0;
    confidence_error = std::max(confidence_error,
                                std::abs(confidence[i] - want_confidence[i]));
  }
  std::cout << "  scoring: prediction " << agree << "/" << prediction.size()
            << ", max |d confidence| " << confidence_error << '\n';
  Require(agree * 100 >= prediction.size() * 95, "scoring agreement");
  return true;
}

void TestGenerate(ov::Generator& generator, const json::Value& golden) {
  ov::DecodingConfig decoding;
  decoding.position_temperature = 0.0F;
  for (const json::Value& entry : golden.find("generate")->items()) {
    const ov::ConditionalInput input = InputFrom(entry);
    ov::GenerationResult result;
    std::string error;
    const auto start = std::chrono::steady_clock::now();
    if (!generator.Generate(input, decoding, 0, {}, &result, &error)) {
      Require(false, error);
      continue;
    }
    const double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
            .count();
    const auto want_codes = Flatten(*entry.find("tokens"));
    const auto want_steps = Flatten(*entry.find("unmasked_at"));
    std::size_t agree = 0;
    int first_divergence = std::numeric_limits<int>::max();
    for (std::size_t i = 0; i < want_codes.size(); ++i) {
      agree +=
          result.codes[i] == static_cast<std::int32_t>(want_codes[i]) ? 1 : 0;
      if (result.codes[i] != static_cast<std::int32_t>(want_codes[i]) ||
          result.unmasked_at[i] != static_cast<std::int32_t>(want_steps[i]))
        first_divergence =
            std::min(first_divergence,
                     std::min(result.unmasked_at[i],
                              static_cast<std::int32_t>(want_steps[i])));
    }
    std::cout << "  generate \"" << entry.find("text")->str() << "\": codes "
              << agree << "/" << want_codes.size() << ", first divergent step "
              << (first_divergence == std::numeric_limits<int>::max()
                      ? std::string("none")
                      : std::to_string(first_divergence))
              << ", " << seconds << " s\n";
    Require(
        std::all_of(result.codes.begin(), result.codes.end(),
                    [](std::int32_t code) { return code >= 0 && code < 1024; }),
        "every code unmasked");
  }
}

}  // namespace

int main(int argc, char** argv) {
  const std::string_view mode = argc > 1 ? argv[1] : "all";
  const char* root = std::getenv("GUFO_OMNIVOICE_MODEL");
  if (root == nullptr || *root == '\0') {
    std::cout << "skip: GUFO_OMNIVOICE_MODEL is not set\n";
    return kSkip;
  }
  std::string error;
  const auto config = ov::LoadConfig(root, &error);
  if (!config.has_value()) {
    std::cout << "FAIL config: " << error << '\n';
    return 1;
  }
  const auto load_start = std::chrono::steady_clock::now();
  auto generator = ov::Generator::Load(*config, root, &error);
  if (generator == nullptr) {
    std::cout << "FAIL load: " << error << '\n';
    return 1;
  }
  std::cout << "  loaded in "
            << std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                             load_start)
                   .count()
            << " s\n";
  std::ifstream stream(GUFO_OMNIVOICE_GENERATOR_GOLDEN);
  const json::Value golden =
      json::parse(std::string((std::istreambuf_iterator<char>(stream)),
                              std::istreambuf_iterator<char>()));
  if (mode == "all" || mode == "forward")
    TestForward(*generator, golden);
  if (mode == "all" || mode == "generate")
    TestGenerate(*generator, golden);
  std::cout << (failures == 0 ? "ok   " : "FAIL ") << "generator, " << failures
            << " failures\n";
  return failures == 0 ? 0 : 1;
}
