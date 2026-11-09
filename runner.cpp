#include "runner.hpp"

#include <algorithm>
#include <cstring>
#include <deque>
#include <memory>
#include <utility>
#include <vector>

#include "synthesizer.hpp"

namespace gufo::models::omnivoice {
namespace {

constexpr const char* kAutoVoice = "auto";
constexpr const char* kDesignVoice = "voice-design";
// Prepared reference voices kept for repeated requests (presets reuse theirs).
constexpr std::size_t kCachedVoices = 8;
// Streamed audio arrives in pieces of this many seconds.
constexpr double kStreamPieceSeconds = 0.5;

std::uint64_t Fingerprint(const ext::TtsAudioBuffer& audio,
                          const std::string& text) {
  std::uint64_t hash = 1469598103934665603ULL;
  const auto mix = [&hash](const void* data, std::size_t bytes) {
    const auto* p = static_cast<const unsigned char*>(data);
    for (std::size_t i = 0; i < bytes; ++i) {
      hash ^= p[i];
      hash *= 1099511628211ULL;
    }
  };
  mix(&audio.sample_rate, sizeof(audio.sample_rate));
  mix(&audio.channels, sizeof(audio.channels));
  mix(audio.samples.data(), audio.samples.size() * sizeof(float));
  mix(text.data(), text.size());
  return hash;
}

void SetError(std::string* error, std::string message) {
  if (error != nullptr)
    *error = std::move(message);
}

class OmniVoiceRunner final : public ext::TtsRunner {
public:
  explicit OmniVoiceRunner(std::unique_ptr<Synthesizer> synthesizer)
      : synthesizer_(std::move(synthesizer)) {}

  bool Synthesize(const ext::SynthesisRequest& request,
                  const ext::CancellationCheck& is_cancelled,
                  ext::SynthesisResult* result, std::string* error) override {
    if (request.speaker_embedding_only) {
      SetError(error,
               "OmniVoice clones a voice from its recording and transcript; "
               "send reference_text instead of speaker_embedding_only");
      return false;
    }
    SynthesisOptions options;
    options.text = request.text;
    options.language = request.language;
    options.instruct = request.instruct;
    options.max_frames = request.max_new_tokens;
    options.seed = request.sampling.seed;
    if (!request.sampling.sample)
      options.decoding.position_temperature = 0.0F;
    if (!request.reference_audio.samples.empty()) {
      options.reference = Reference(request, error);
      if (options.reference == nullptr)
        return false;
    }
    const std::uint32_t rate = SampleRate();
    // The post-processed audio always opens with 0.1 s of padding. Streams
    // send it before generating, so audio starts at once and a client that
    // leaves is noticed while the model still works.
    const auto padding = static_cast<std::size_t>(0.1 * rate);
    const bool streaming = static_cast<bool>(request.on_audio);
    bool stopped = false;
    const auto cancelled = [&] {
      return stopped || (is_cancelled && is_cancelled());
    };
    if (streaming) {
      const std::vector<float> silence(padding, 0.0F);
      if (!request.on_audio(silence) || cancelled()) {
        SetError(error, "cancelled");
        return false;
      }
    }
    std::vector<float> audio;
    std::vector<std::int32_t> codes;
    if (!synthesizer_->Synthesize(options, cancelled, &audio, &codes, error))
      return false;
    // An utterance that is silence throughout still keeps its padding.
    if (audio.size() < padding)
      audio.assign(padding, 0.0F);
    result->sample_rate = rate;
    result->code_groups =
        static_cast<std::uint32_t>(synthesizer_->config().audio_codebooks);
    result->codes = std::move(codes);
    result->sample_count = audio.size();
    if (!streaming) {
      result->samples = std::move(audio);
      return true;
    }
    const auto piece = static_cast<std::size_t>(kStreamPieceSeconds * rate);
    for (std::size_t offset = padding; offset < audio.size(); offset += piece) {
      const std::size_t count = std::min(piece, audio.size() - offset);
      if (cancelled() ||
          !request.on_audio(
              std::span<const float>(audio).subspan(offset, count))) {
        SetError(error, "cancelled");
        return false;
      }
    }
    return true;
  }

  std::vector<std::string> Voices() const override {
    return {kAutoVoice, kDesignVoice};
  }

  ext::TtsVariant Variant() const override { return ext::TtsVariant::kBase; }

  std::uint32_t SampleRate() const override {
    return synthesizer_->SampleRate();
  }

  ext::TtsConditioningCapabilities Conditioning() const override {
    // One checkpoint serves an automatic voice, voice design from
    // instructions and cloning from a reference or a preset.
    ext::TtsConditioningCapabilities capabilities;
    capabilities.unconditioned = true;
    capabilities.require_voice = false;
    capabilities.default_voice = kAutoVoice;
    capabilities.reference_audio = ext::TtsConditioningMode::kOptional;
    capabilities.reference_presets = true;
    capabilities.instructions = ext::TtsConditioningMode::kOptional;
    return capabilities;
  }

  std::string ModelId() const override { return "omnivoice"; }

private:
  struct CachedVoice {
    std::uint64_t key = 0;
    std::shared_ptr<ReferenceVoice> voice;
  };

  /// The request's reference voice, encoded once per recording and
  /// transcript.
  const ReferenceVoice* Reference(const ext::SynthesisRequest& request,
                                  std::string* error) {
    const std::uint64_t key =
        Fingerprint(request.reference_audio, request.reference_text);
    for (auto it = cache_.begin(); it != cache_.end(); ++it) {
      if (it->key == key) {
        CachedVoice hit = *it;
        cache_.erase(it);
        cache_.push_front(hit);
        return cache_.front().voice.get();
      }
    }
    auto voice = std::make_shared<ReferenceVoice>();
    const ext::TtsAudioBuffer& audio = request.reference_audio;
    if (!synthesizer_->CloneVoice(audio.samples, audio.channels,
                                  audio.sample_rate, request.reference_text,
                                  voice.get(), error))
      return nullptr;
    cache_.push_front({key, std::move(voice)});
    if (cache_.size() > kCachedVoices)
      cache_.pop_back();
    return cache_.front().voice.get();
  }

  std::unique_ptr<Synthesizer> synthesizer_;
  std::deque<CachedVoice> cache_;
};

}  // namespace

std::unique_ptr<ext::TtsRunner> CreateOmniVoiceRunner(
    const ext::TtsRunnerOptions& options, std::string* error) {
  auto synthesizer = Synthesizer::Load(options.model_root, error);
  if (synthesizer == nullptr)
    return nullptr;
  return std::make_unique<OmniVoiceRunner>(std::move(synthesizer));
}

}  // namespace gufo::models::omnivoice
