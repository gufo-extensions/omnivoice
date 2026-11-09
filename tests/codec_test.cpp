// Compares the GPU audio tokenizer with the float32 reference run of
// tools/reference/generator_golden.py.
#include "codec.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <gufo/ext/json.hpp>
#include <iostream>
#include <iterator>
#include <string>
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

void TestDecode(ov::Codec& codec, const json::Value& golden) {
  const json::Value& entry = *golden.find("decode");
  const json::Value& generation =
      golden.find("generate")->items()[entry.find("tokens_from")->as_size()];
  std::vector<std::int32_t> codes;
  for (const json::Value& row : generation.find("tokens")->items())
    for (const json::Value& code : row.items())
      codes.push_back(static_cast<std::int32_t>(code.as_double()));
  const std::size_t frames = codes.size() / 8;
  std::vector<float> audio;
  std::string error;
  const auto start = std::chrono::steady_clock::now();
  if (!codec.Decode(codes, frames, &audio, &error)) {
    Require(false, error);
    return;
  }
  const double seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
          .count();
  Require(audio.size() == entry.find("length")->as_size(), "decoded length");
  double squares = 0.0;
  for (const float sample : audio)
    squares += static_cast<double>(sample) * sample;
  const double rms = std::sqrt(squares / static_cast<double>(audio.size()));
  const std::size_t stride = golden.find("stride")->as_size();
  const auto& samples = entry.find("samples")->items();
  double worst = 0.0;
  for (std::size_t i = 0; i < samples.size() && i * stride < audio.size(); ++i)
    worst =
        std::max(worst, std::abs(audio[i * stride] - samples[i].as_double()));
  const double expected_rms = entry.find("rms")->as_double();
  std::cout << "  decode " << frames << " frames: rms " << rms << " (reference "
            << expected_rms << "), max |d sample| " << worst << ", " << seconds
            << " s\n";
  Require(std::abs(rms - expected_rms) < 1e-4, "decoded RMS");
  Require(worst < 1e-3, "decoded samples");
}

/// tools/reference/generator_golden.py's encoder test signal.
std::vector<float> Signal(double seconds, std::int64_t seed) {
  constexpr double kPi = 3.14159265358979323846;
  std::int64_t state = seed;
  std::vector<float> out(static_cast<std::size_t>(seconds * 24000));
  for (std::size_t n = 0; n < out.size(); ++n) {
    state = (state * 1103515245 + 12345) & 0x7FFFFFFF;
    const auto x = static_cast<double>(n);
    const double envelope = 0.5 + 0.5 * std::sin(2 * kPi * 3 * x / 24000);
    const double tone = std::sin(2 * kPi * 180 * x / 24000) +
                        0.4 * std::sin(2 * kPi * 720 * x / 24000);
    const double noise =
        static_cast<double>((state >> 8) % 2001 - 1000) / 1000.0;
    out[n] = static_cast<float>(0.2 * envelope * tone + 0.02 * noise);
  }
  return out;
}

void TestEncode(ov::Codec& codec, const json::Value& golden) {
  const json::Value& entry = *golden.find("encode");
  const json::Value& signal = *entry.find("signal");
  const auto samples =
      Signal(signal.find("seconds")->as_double(),
             static_cast<std::int64_t>(signal.find("seed")->as_size()));
  std::vector<std::int32_t> codes;
  std::size_t frames = 0;
  std::string error;
  const auto start = std::chrono::steady_clock::now();
  if (!codec.Encode(samples, &codes, &frames, &error)) {
    Require(false, error);
    return;
  }
  const double seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
          .count();
  const auto& expected = entry.find("codes")->items();
  Require(frames == expected.front().items().size(), "encoded frame count");
  std::cout << "  encode " << frames << " frames in " << seconds << " s:";
  std::size_t total_agree = 0;
  for (std::size_t c = 0; c < expected.size(); ++c) {
    std::size_t agree = 0;
    const auto& row = expected[c].items();
    for (std::size_t f = 0; f < frames && f < row.size(); ++f)
      agree +=
          codes[c * frames + f] == static_cast<std::int32_t>(row[f].as_double())
              ? 1
              : 0;
    std::cout << ' ' << agree;
    total_agree += agree;
  }
  std::cout << " of " << frames << " per codebook\n";
  Require(total_agree * 100 >= frames * expected.size() * 90, "encoded codes");
}

}  // namespace

int main() {
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
  auto codec = ov::Codec::Load(*config, root, &error);
  if (codec == nullptr) {
    std::cout << "FAIL load: " << error << '\n';
    return 1;
  }
  std::ifstream stream(GUFO_OMNIVOICE_GENERATOR_GOLDEN);
  const json::Value golden =
      json::parse(std::string((std::istreambuf_iterator<char>(stream)),
                              std::istreambuf_iterator<char>()));
  TestDecode(*codec, golden);
  TestEncode(*codec, golden);
  std::cout << (failures == 0 ? "ok   " : "FAIL ") << "codec, " << failures
            << " failures\n";
  return failures == 0 ? 0 : 1;
}
