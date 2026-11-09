// OmniVoice.generate() for one text: ported from k2-fsa/OmniVoice's
// omnivoice/models/omnivoice.py (_preprocess_all, _generate_chunked,
// _decode_and_post_process; Apache-2.0, see licenses/OmniVoice.txt).
#include "synthesizer.hpp"

#include <algorithm>
#include <utility>

#include "audio.hpp"
#include "codec.hpp"
#include "generator.hpp"
#include "prompt.hpp"
#include "text.hpp"
#include "tokenizer.hpp"
#include "unicode.hpp"

namespace gufo::models::omnivoice {
namespace {

// OmniVoiceGenerationConfig: texts estimated longer than 30 s are split
// into chunks of about 15 s.
constexpr double kChunkThresholdSeconds = 30.0;
constexpr double kChunkSeconds = 15.0;

void SetError(std::string* error, std::string message) {
  if (error != nullptr)
    *error = std::move(message);
}

std::size_t CodepointCount(const std::string& text) {
  std::vector<Codepoint> points;
  return DecodeUtf8(text, &points) ? points.size() : text.size();
}

}  // namespace

struct Synthesizer::State {
  Config config;
  std::unique_ptr<Tokenizer> tokenizer;
  std::unique_ptr<Generator> generator;
  std::unique_ptr<Codec> codec;

  /// One generation of `text` against an optional reference.
  bool GenerateChunk(const std::string& text,
                     const std::optional<std::string>& language,
                     const std::optional<std::string>& instruct,
                     const std::optional<std::string>& reference_text,
                     const std::vector<std::int32_t>& reference_codes,
                     std::size_t reference_frames,
                     const SynthesisOptions& options,
                     const std::function<bool()>& cancelled,
                     GenerationResult* result, std::size_t* frames,
                     std::string* error) {
    *frames = EstimateTargetFrames(text, reference_text, reference_frames);
    if (options.max_frames > 0)
      *frames = std::min(*frames, options.max_frames);
    PromptText prompt;
    prompt.text = text;
    prompt.reference_text = reference_text;
    prompt.language = language;
    prompt.instruct = instruct;
    prompt.denoise = options.denoise;
    prompt.has_reference = reference_frames > 0;
    std::vector<std::uint32_t> ids;
    if (!EncodePromptText(*tokenizer, prompt, &ids, error))
      return false;
    const ConditionalInput input =
        ConditionalInput::Build(ids, reference_codes, reference_frames, *frames,
                                config.audio_codebooks, config.audio_mask_id);
    return generator->Generate(input, options.decoding, options.seed, cancelled,
                               result, error);
  }
};

Synthesizer::Synthesizer(std::unique_ptr<State> state)
    : state_(std::move(state)) {}
Synthesizer::~Synthesizer() = default;

std::unique_ptr<Synthesizer> Synthesizer::Load(
    const std::filesystem::path& model_root, std::string* error) {
  auto state = std::make_unique<State>();
  const auto config = LoadConfig(model_root.string(), error);
  if (!config.has_value() || !IsSupported(*config, error))
    return nullptr;
  state->config = *config;
  state->tokenizer = Tokenizer::Load(model_root / "tokenizer.json", error);
  if (state->tokenizer == nullptr)
    return nullptr;
  if (state->tokenizer->vocabulary_size() > config->backbone.vocab_size) {
    SetError(error,
             "OmniVoice tokenizer is larger than the backbone's vocabulary");
    return nullptr;
  }
  state->generator = Generator::Load(*config, model_root, error);
  if (state->generator == nullptr)
    return nullptr;
  state->codec = Codec::Load(*config, model_root, error);
  if (state->codec == nullptr)
    return nullptr;
  return std::unique_ptr<Synthesizer>(new Synthesizer(std::move(state)));
}

const Config& Synthesizer::config() const {
  return state_->config;
}

std::uint32_t Synthesizer::SampleRate() const {
  return state_->config.codec.sample_rate;
}

bool Synthesizer::CloneVoice(std::span<const float> interleaved,
                             std::uint32_t channels, std::uint32_t rate,
                             const std::string& transcript,
                             ReferenceVoice* voice, std::string* error) {
  State& s = *state_;
  if (StripSpace(transcript).empty()) {
    SetError(error, "OmniVoice: voice cloning needs the reference transcript");
    return false;
  }
  const auto prepared =
      PrepareReference(interleaved, channels, rate, SampleRate(),
                       s.config.codec.frame_samples, error);
  if (!prepared.has_value())
    return false;
  if (!s.codec->Encode(prepared->samples, &voice->codes, &voice->frames, error))
    return false;
  voice->text = AddPunctuation(transcript);
  voice->rms = prepared->rms;
  return true;
}

bool Synthesizer::Synthesize(const SynthesisOptions& options,
                             const std::function<bool()>& cancelled,
                             std::vector<float>* audio,
                             std::vector<std::int32_t>* codes,
                             std::string* error) {
  State& s = *state_;
  const std::size_t codebooks = s.config.audio_codebooks;
  // The reference estimates and chunks the raw text; only the prompt strips
  // it.
  const std::string& text = options.text;
  if (StripSpace(text).empty()) {
    SetError(error, "OmniVoice: the text is empty");
    return false;
  }
  const std::optional<std::string> language = ResolveLanguage(options.language);
  std::string instruct_error;
  const std::optional<std::string> instruct = ResolveInstruct(
      options.instruct, ContainsCjkIdeograph(text), &instruct_error);
  if (!instruct_error.empty()) {
    SetError(error, "OmniVoice: " + instruct_error);
    return false;
  }
  const ReferenceVoice* reference = options.reference;
  std::optional<std::string> reference_text;
  std::vector<std::int32_t> reference_codes;
  std::size_t reference_frames = 0;
  if (reference != nullptr) {
    reference_text = reference->text;
    reference_codes = reference->codes;
    reference_frames = reference->frames;
  }
  // Short texts are one generation; longer ones are chunked at punctuation.
  const std::size_t estimate =
      EstimateTargetFrames(text, reference_text, reference_frames);
  const auto frame_rate = static_cast<double>(s.config.codec.FrameRate());
  std::vector<std::string> chunks = {text};
  if (static_cast<double>(estimate) > kChunkThresholdSeconds * frame_rate) {
    const double frames_per_character =
        static_cast<double>(estimate) /
        static_cast<double>(CodepointCount(text));
    const auto chunk_length = static_cast<std::size_t>(
        kChunkSeconds * frame_rate / frames_per_character);
    chunks = ChunkTextPunctuation(text, chunk_length, 3);
  }
  std::vector<std::vector<float>> pieces;
  if (codes != nullptr)
    codes->clear();
  for (std::size_t index = 0; index < chunks.size(); ++index) {
    GenerationResult result;
    std::size_t frames = 0;
    if (!s.GenerateChunk(chunks[index], language, instruct, reference_text,
                         reference_codes, reference_frames, options, cancelled,
                         &result, &frames, error))
      return false;
    std::vector<float> piece;
    if (!s.codec->Decode(result.codes, frames, &piece, error))
      return false;
    pieces.push_back(std::move(piece));
    if (codes != nullptr) {
      for (std::size_t f = 0; f < frames; ++f)
        for (std::size_t c = 0; c < codebooks; ++c)
          codes->push_back(result.codes[c * frames + f]);
    }
    // Without a reference, the first chunk's voice carries the rest.
    if (reference == nullptr && index == 0 && chunks.size() > 1) {
      reference_text = chunks[0];
      reference_codes = result.codes;
      reference_frames = frames;
    }
  }
  const std::optional<double> level =
      reference != nullptr ? std::optional<double>(reference->rms)
                           : std::nullopt;
  *audio =
      PostProcess(CrossFadeChunks(pieces, SampleRate()), SampleRate(), level);
  return true;
}

}  // namespace gufo::models::omnivoice
