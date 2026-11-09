#include "audio.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <gufo/ext/json.hpp>
#include <iostream>
#include <iterator>
#include <string>
#include <string_view>
#include <vector>

namespace json = gufo::json;
namespace ov = gufo::models::omnivoice;

namespace {

int failures = 0;
int checks = 0;
std::size_t stride = 1;

void Fail(const std::string& what) {
  std::cout << "FAIL " << what << '\n';
  ++failures;
}

/// tools/reference/audio_golden.py's integer signal generator.
std::vector<float> Generate(const json::Value& spec, std::int64_t rate,
                            std::int64_t seed) {
  std::int64_t state = seed;
  std::vector<float> samples;
  for (const json::Value& segment : spec.items()) {
    const auto& parts = segment.items();
    const auto ms = static_cast<std::int64_t>(parts[0].as_size());
    const std::string& kind = parts[1].str();
    const auto amplitude = static_cast<std::int64_t>(parts[2].as_size());
    const std::int64_t noise_amplitude = std::min<std::int64_t>(amplitude, 64);
    for (std::int64_t n = 0; n < ms * rate / 1000; ++n) {
      state = (state * 1103515245 + 12345) & 0x7FFFFFFF;
      std::int64_t value = 0;
      if (kind == "tone") {
        constexpr std::int64_t kPeriod = 55;
        const std::int64_t phase = n % kPeriod;
        const std::int64_t noise =
            (state >> 8) % (2 * noise_amplitude + 1) - noise_amplitude;
        value =
            (4 * amplitude * std::abs(2 * phase - kPeriod)) / (2 * kPeriod) -
            amplitude + noise;
      } else {
        value = (state >> 8) % (2 * amplitude + 1) - amplitude;
      }
      value = std::clamp<std::int64_t>(value, -32768, 32767);
      samples.push_back(static_cast<float>(value) / 32768.0F);
    }
  }
  return samples;
}

std::vector<float> GenerateSignal(const json::Value& signal) {
  std::vector<float> samples =
      Generate(*signal.find("spec"),
               static_cast<std::int64_t>(signal.find("rate")->as_size()),
               static_cast<std::int64_t>(signal.find("seed")->as_size()));
  if (const json::Value* extra = signal.find("extra"))
    samples.insert(samples.end(), extra->as_size(), 0.25F);
  return samples;
}

/// Exact PCM16 output: length, sums and strided samples.
void ComparePcm(const std::string& what, const std::vector<float>& got,
                const json::Value& expected) {
  ++checks;
  if (got.size() != expected.find("length")->as_size()) {
    Fail(what + ": length " + std::to_string(got.size()) + ", expected " +
         std::to_string(expected.find("length")->as_size()));
    return;
  }
  std::int64_t sum = 0, squares = 0;
  std::vector<std::int64_t> pcm;
  for (const float sample : got) {
    const auto value =
        static_cast<std::int64_t>(std::nearbyint(sample * 32768.0));
    sum += value;
    squares += value * value;
    pcm.push_back(value);
  }
  const auto expected_sum =
      static_cast<std::int64_t>(expected.find("sum")->as_double());
  const auto expected_squares =
      static_cast<std::int64_t>(expected.find("sum_squares")->as_double());
  if (sum != expected_sum || squares != expected_squares)
    Fail(what + ": checksum " + std::to_string(sum) + "/" +
         std::to_string(squares) + ", expected " +
         std::to_string(expected_sum) + "/" + std::to_string(expected_squares));
}

/// Float output: length and strided samples within `tolerance`.
void CompareSamples(const std::string& what, const std::vector<float>& got,
                    const json::Value& expected, double tolerance) {
  ++checks;
  if (got.size() != expected.find("length")->as_size()) {
    Fail(what + ": length " + std::to_string(got.size()) + ", expected " +
         std::to_string(expected.find("length")->as_size()));
    return;
  }
  double worst = 0.0;
  const auto& samples = expected.find("samples")->items();
  for (std::size_t index = 0; index < samples.size(); ++index)
    worst = std::max(
        worst, std::abs(got[index * stride] - samples[index].as_double()));
  if (worst > tolerance)
    Fail(what + ": max difference " + std::to_string(worst));
}

}  // namespace

int main() {
  std::ifstream stream(GUFO_OMNIVOICE_AUDIO_GOLDEN);
  const json::Value golden =
      json::parse(std::string((std::istreambuf_iterator<char>(stream)),
                              std::istreambuf_iterator<char>()));
  stride = golden.find("stride")->as_size();
  std::size_t index = 0;
  for (const json::Value& entry : golden.find("silence")->items()) {
    const std::vector<float> input = GenerateSignal(*entry.find("signal"));
    const auto& params = entry.find("params")->items();
    const auto got =
        ov::RemoveSilence(input, 24000, static_cast<int>(params[0].as_size()),
                          static_cast<int>(params[1].as_size()),
                          static_cast<int>(params[2].as_size()));
    ComparePcm("silence " + std::to_string(index++), got, entry);
  }
  index = 0;
  for (const json::Value& entry : golden.find("fade")->items())
    CompareSamples("fade " + std::to_string(index++),
                   ov::FadeAndPad(GenerateSignal(*entry.find("signal")), 24000),
                   entry, 1e-7);
  index = 0;
  for (const json::Value& entry : golden.find("postprocess")->items()) {
    const json::Value* rms = entry.find("rms");
    const std::optional<double> level =
        rms->is_null() ? std::nullopt : std::optional<double>(rms->as_double());
    CompareSamples(
        "postprocess " + std::to_string(index++),
        ov::PostProcess(GenerateSignal(*entry.find("signal")), 24000, level),
        entry, 1e-6);
  }
  for (const json::Value& entry : golden.find("crossfade")->items()) {
    std::vector<std::vector<float>> chunks;
    for (const json::Value& signal : entry.find("signals")->items())
      chunks.push_back(GenerateSignal(signal));
    CompareSamples("crossfade", ov::CrossFadeChunks(chunks, 24000), entry,
                   1e-7);
  }
  for (const json::Value& entry : golden.find("resample")->items()) {
    const json::Value& signal = *entry.find("signal");
    const auto source =
        static_cast<std::uint32_t>(signal.find("rate")->as_size());
    const auto target =
        static_cast<std::uint32_t>(entry.find("target")->as_size());
    CompareSamples(
        "resample " + std::to_string(source) + " -> " + std::to_string(target),
        ov::Resample(GenerateSignal(signal), source, target), entry, 2e-6);
  }
  index = 0;
  for (const json::Value& entry : golden.find("reference")->items()) {
    const auto channels =
        static_cast<std::uint32_t>(entry.find("channels")->as_size());
    const auto rate = static_cast<std::int64_t>(entry.find("rate")->as_size());
    const auto& seeds = entry.find("seeds")->items();
    const auto left = Generate(*entry.find("spec"), rate,
                               static_cast<std::int64_t>(seeds[0].as_size()));
    const auto right = Generate(*entry.find("spec"), rate,
                                static_cast<std::int64_t>(seeds[1].as_size()));
    std::vector<float> interleaved;
    for (std::size_t frame = 0; frame < left.size(); ++frame) {
      interleaved.push_back(left[frame]);
      if (channels == 2)
        interleaved.push_back(right[frame]);
    }
    std::string error;
    const auto reference = ov::PrepareReference(
        interleaved, channels, static_cast<std::uint32_t>(rate), 24000, 960,
        &error);
    const std::string what = "reference " + std::to_string(index++);
    ++checks;
    if (!reference.has_value()) {
      Fail(what + ": " + error);
      continue;
    }
    if (std::abs(reference->rms - entry.find("rms")->as_double()) > 1e-6)
      Fail(what + ": rms " + std::to_string(reference->rms));
    // Resampling rounds differently in float32; PCM16 values may differ by
    // one step where they straddle a rounding boundary.
    if (reference->samples.size() != entry.find("length")->as_size()) {
      Fail(what + ": length " + std::to_string(reference->samples.size()));
      continue;
    }
    double worst = 0.0;
    const auto& samples = entry.find("samples")->items();
    for (std::size_t sample = 0; sample < samples.size(); ++sample)
      worst = std::max(worst,
                       std::abs(reference->samples[sample * stride] * 32768.0 -
                                samples[sample].as_double()));
    if (worst > 1.0)
      Fail(what + ": max PCM16 difference " + std::to_string(worst));
  }
  std::cout << (failures == 0 ? "ok   " : "FAIL ") << checks << " audio cases, "
            << failures << " failures\n";
  return failures == 0 ? 0 : 1;
}
