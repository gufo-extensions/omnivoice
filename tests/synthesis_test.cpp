// End-to-end synthesis on the released checkpoint: seeded determinism,
// voice design, cloning and the chunked path for long texts.
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "synthesizer.hpp"

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

double Seconds(const std::vector<float>& audio, const ov::Synthesizer& s) {
  return static_cast<double>(audio.size()) / s.SampleRate();
}

double Rms(const std::vector<float>& audio) {
  double sum = 0.0;
  for (const float sample : audio)
    sum += static_cast<double>(sample) * sample;
  return audio.empty() ? 0.0
                       : std::sqrt(sum / static_cast<double>(audio.size()));
}

bool Run(ov::Synthesizer& synthesizer, const ov::SynthesisOptions& options,
         std::vector<float>* audio, std::vector<std::int32_t>* codes,
         const std::string& label) {
  std::string error;
  const auto start = std::chrono::steady_clock::now();
  if (!synthesizer.Synthesize(options, {}, audio, codes, &error)) {
    Require(false, label + ": " + error);
    return false;
  }
  const double elapsed =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
          .count();
  std::cout << "  " << label << ": " << Seconds(*audio, synthesizer)
            << " s of audio in " << elapsed << " s, rms " << Rms(*audio)
            << '\n';
  return true;
}

}  // namespace

int main() {
  const char* root = std::getenv("GUFO_OMNIVOICE_MODEL");
  if (root == nullptr || *root == '\0') {
    std::cout << "skip: GUFO_OMNIVOICE_MODEL is not set\n";
    return kSkip;
  }
  std::string error;
  auto synthesizer = ov::Synthesizer::Load(root, &error);
  if (synthesizer == nullptr) {
    std::cout << "FAIL load: " << error << '\n';
    return 1;
  }
  ov::SynthesisOptions options;
  options.text = "The little brown dog ran across the garden.";
  options.language = "English";
  options.seed = 42;
  std::vector<float> first, second, third;
  std::vector<std::int32_t> codes;
  if (Run(*synthesizer, options, &first, &codes, "seed 42") &&
      Run(*synthesizer, options, &second, nullptr, "seed 42 again")) {
    Require(first == second, "the same seed reproduces the audio");
    Require(Seconds(first, *synthesizer) > 1.0 &&
                Seconds(first, *synthesizer) < 6.0,
            "a plausible duration");
    Require(Rms(first) > 0.01, "audible output");
    Require(codes.size() % 8 == 0 && !codes.empty(), "frame-major codes");
  }
  options.seed = 43;
  if (Run(*synthesizer, options, &third, nullptr, "seed 43"))
    Require(third != first, "another seed changes the audio");
  // Voice design and its validation.
  options.instruct = "female, low pitch, british accent";
  std::vector<float> designed;
  Run(*synthesizer, options, &designed, nullptr, "voice design");
  options.instruct = "very loud";
  Require(!synthesizer->Synthesize(options, {}, &designed, nullptr, &error) &&
              error.find("unsupported voice design items") != std::string::npos,
          "unknown voice-design items are rejected");
  options.instruct.clear();
  // Cloning the first result.
  ov::ReferenceVoice voice;
  if (synthesizer->CloneVoice(first, 1, synthesizer->SampleRate(),
                              "The little brown dog ran across the garden",
                              &voice, &error)) {
    Require(voice.frames > 20 && voice.text.back() == '.',
            "an encoded reference with a punctuated transcript");
    options.reference = &voice;
    options.text = "Nice to meet you.";
    std::vector<float> cloned;
    Run(*synthesizer, options, &cloned, nullptr, "cloned voice");
    options.reference = nullptr;
  } else {
    Require(false, "clone: " + error);
  }
  // A text estimated above 30 s is generated in chunks.
  std::string story;
  for (int sentence = 0; sentence < 9; ++sentence)
    story +=
        "The quick brown fox jumps over the lazy dog near the quiet river "
        "bank, and then it runs back home before the evening rain. ";
  options.text = story;
  std::vector<float> long_audio;
  if (Run(*synthesizer, options, &long_audio, &codes, "chunked story"))
    Require(Seconds(long_audio, *synthesizer) > 30.0,
            "the long text is spoken in full");
  // Cancellation stops the generation.
  options.text = "This request is cancelled before it starts.";
  Require(!synthesizer->Synthesize(
              options, [] { return true; }, &long_audio, nullptr, &error) &&
              error == "cancelled",
          "cancellation");
  std::cout << (failures == 0 ? "ok   " : "FAIL ") << "synthesis, " << failures
            << " failures\n";
  return failures == 0 ? 0 : 1;
}
