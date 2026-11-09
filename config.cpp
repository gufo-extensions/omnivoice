#include "config.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <gufo/ext/json.hpp>
#include <iterator>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace gufo::models::omnivoice {
namespace {

namespace json = gufo::json;

class ConfigError : public std::runtime_error {
public:
  using std::runtime_error::runtime_error;
};

/// Typed access to one JSON object; failures name the file and the key.
class Fields {
public:
  Fields(const json::Value& object, std::string where)
      : object_(object), where_(std::move(where)) {
    if (!object_.is_object())
      throw ConfigError(where_ + " is not a JSON object");
  }

  [[nodiscard]] const json::Value& Get(const std::string& key) const {
    const json::Value* value = object_.find(key);
    if (value == nullptr)
      throw ConfigError(where_ + " lacks \"" + key + "\"");
    return *value;
  }

  [[nodiscard]] std::size_t Size(const std::string& key) const {
    const json::Value& value = Get(key);
    const std::size_t size = value.as_size(0);
    if (size == 0)
      throw ConfigError(where_ + "." + key + " is not a positive integer");
    return size;
  }

  [[nodiscard]] float Number(const std::string& key) const {
    const json::Value& value = Get(key);
    if (!value.is_number() || !std::isfinite(value.as_double()))
      throw ConfigError(where_ + "." + key + " is not a number");
    return static_cast<float>(value.as_double());
  }

  [[nodiscard]] std::string String(const std::string& key) const {
    const json::Value& value = Get(key);
    if (!value.is_string())
      throw ConfigError(where_ + "." + key + " is not a string");
    return value.str();
  }

  [[nodiscard]] bool Bool(const std::string& key) const {
    const json::Value& value = Get(key);
    if (!value.is_bool())
      throw ConfigError(where_ + "." + key + " is not a boolean");
    return value.as_bool();
  }

  [[nodiscard]] std::vector<std::size_t> Sizes(const std::string& key) const {
    const json::Value& value = Get(key);
    if (!value.is_array() || value.empty())
      throw ConfigError(where_ + "." + key + " is not a non-empty array");
    std::vector<std::size_t> sizes;
    for (const json::Value& item : value.items()) {
      const std::size_t size = item.as_size(0);
      if (size == 0)
        throw ConfigError(where_ + "." + key + " holds a non-positive entry");
      sizes.push_back(size);
    }
    return sizes;
  }

  [[nodiscard]] std::vector<double> Numbers(const std::string& key) const {
    const json::Value& value = Get(key);
    if (!value.is_array() || value.empty())
      throw ConfigError(where_ + "." + key + " is not a non-empty array");
    std::vector<double> numbers;
    for (const json::Value& item : value.items()) {
      if (!item.is_number() || !std::isfinite(item.as_double()))
        throw ConfigError(where_ + "." + key + " holds a non-number");
      numbers.push_back(item.as_double());
    }
    return numbers;
  }

  [[nodiscard]] Fields Object(const std::string& key) const {
    return Fields(Get(key), where_ + "." + key);
  }

  void Expect(const std::string& key, std::string_view expected) const {
    const std::string actual = String(key);
    if (actual != expected)
      throw ConfigError(where_ + "." + key + " is \"" + actual +
                        "\", expected \"" + std::string(expected) + "\"");
  }

  void ExpectBool(const std::string& key, bool expected) const {
    if (Bool(key) != expected)
      throw ConfigError(where_ + "." + key + " must be " +
                        (expected ? "true" : "false"));
  }

private:
  const json::Value& object_;
  std::string where_;
};

json::Value ParseJson(std::string_view text, const std::string& where) {
  try {
    return json::parse(text);
  } catch (const std::exception& error) {
    throw ConfigError(where + ": " + error.what());
  }
}

std::size_t Product(const std::vector<std::size_t>& values) {
  return std::accumulate(values.begin(), values.end(), std::size_t{1},
                         std::multiplies<>());
}

void Require(bool condition, const std::string& message) {
  if (!condition)
    throw ConfigError(message);
}

BackboneConfig ParseBackbone(const Fields& llm) {
  llm.Expect("model_type", "qwen3");
  llm.Expect("hidden_act", "silu");
  llm.ExpectBool("attention_bias", false);
  BackboneConfig backbone;
  backbone.vocab_size = llm.Size("vocab_size");
  backbone.hidden_size = llm.Size("hidden_size");
  backbone.intermediate_size = llm.Size("intermediate_size");
  backbone.layers = llm.Size("num_hidden_layers");
  backbone.heads = llm.Size("num_attention_heads");
  backbone.kv_heads = llm.Size("num_key_value_heads");
  backbone.head_dim = llm.Size("head_dim");
  backbone.max_positions = llm.Size("max_position_embeddings");
  backbone.rms_norm_eps = llm.Number("rms_norm_eps");
  // Transformers 5 moved rope_theta into rope_parameters.
  const Fields rope = llm.Object("rope_parameters");
  rope.Expect("rope_type", "default");
  backbone.rope_theta = rope.Number("rope_theta");
  Require(backbone.heads % backbone.kv_heads == 0,
          "llm_config: attention heads are not a multiple of the KV heads");
  Require(backbone.rms_norm_eps > 0.0F && backbone.rope_theta > 0.0F,
          "llm_config: rms_norm_eps and rope_theta must be positive");
  return backbone;
}

SemanticEncoderConfig ParseSemantic(const Fields& codec,
                                    const Fields& semantic) {
  semantic.Expect("model_type", "hubert");
  // The variant the checkpoint uses: group-normalized feature extractor
  // without convolution bias, post-norm encoder layers, GELU throughout.
  semantic.Expect("feat_extract_norm", "group");
  semantic.Expect("feat_extract_activation", "gelu");
  semantic.Expect("hidden_act", "gelu");
  semantic.ExpectBool("conv_bias", false);
  semantic.ExpectBool("do_stable_layer_norm", false);
  semantic.ExpectBool("feat_proj_layer_norm", true);
  SemanticEncoderConfig config;
  config.sample_rate =
      static_cast<std::uint32_t>(codec.Size("semantic_sample_rate"));
  config.frame_samples = codec.Size("downsample_factor");
  config.hidden_size = semantic.Size("hidden_size");
  config.intermediate_size = semantic.Size("intermediate_size");
  config.heads = semantic.Size("num_attention_heads");
  config.layers = semantic.Size("num_hidden_layers");
  config.position_kernel = semantic.Size("num_conv_pos_embeddings");
  config.position_groups = semantic.Size("num_conv_pos_embedding_groups");
  config.layer_norm_eps = semantic.Number("layer_norm_eps");
  config.conv_channels = semantic.Sizes("conv_dim");
  config.conv_kernels = semantic.Sizes("conv_kernel");
  config.conv_strides = semantic.Sizes("conv_stride");
  Require(config.conv_channels.size() == config.conv_kernels.size() &&
              config.conv_kernels.size() == config.conv_strides.size(),
          "semantic_model_config: conv_dim, conv_kernel and conv_stride "
          "differ in length");
  Require(Product(config.conv_strides) == config.frame_samples,
          "semantic_model_config: the feature extractor's strides don't "
          "multiply to downsample_factor");
  Require(config.hidden_size % config.heads == 0 &&
              config.hidden_size % config.position_groups == 0,
          "semantic_model_config: hidden_size is not a multiple of the heads "
          "and the positional convolution groups");
  return config;
}

AcousticConfig ParseAcoustic(const Fields& acoustic,
                             std::size_t frame_samples) {
  acoustic.Expect("model_type", "dac");
  AcousticConfig config;
  config.encoder_channels = acoustic.Size("encoder_hidden_size");
  config.decoder_channels = acoustic.Size("decoder_hidden_size");
  config.latent_channels = acoustic.Size("hidden_size");
  config.downsampling_ratios = acoustic.Sizes("downsampling_ratios");
  config.upsampling_ratios = acoustic.Sizes("upsampling_ratios");
  Require(Product(config.downsampling_ratios) == frame_samples &&
              Product(config.upsampling_ratios) == frame_samples,
          "acoustic_model_config: the resampling ratios don't multiply to "
          "hop_length");
  Require(config.decoder_channels %
                  (std::size_t{1} << config.upsampling_ratios.size()) ==
              0,
          "acoustic_model_config: decoder_hidden_size can't be halved once "
          "per upsampling block");
  return config;
}

CodecConfig ParseCodec(const json::Value& codec_root,
                       const json::Value& preprocessor_root) {
  const Fields codec(codec_root, "audio_tokenizer/config.json");
  const Fields preprocessor(preprocessor_root,
                            "audio_tokenizer/preprocessor_config.json");
  codec.Expect("model_type", "higgs_audio_v2_tokenizer");
  const Fields acoustic = codec.Object("acoustic_model_config");
  CodecConfig config;
  config.sample_rate =
      static_cast<std::uint32_t>(preprocessor.Size("sampling_rate"));
  config.frame_samples = preprocessor.Size("hop_length");
  // acoustic_model_config.sampling_rate (16 kHz) is a leftover of the DAC
  // base config; the codec runs at the preprocessor's rate.
  Require(acoustic.Size("hop_length") == config.frame_samples,
          "audio_tokenizer: the acoustic hop_length differs from the "
          "preprocessor's");
  Require(codec.Size("sample_rate") == config.sample_rate,
          "audio_tokenizer: sample_rate differs from the preprocessor's");
  Require(config.sample_rate % config.frame_samples == 0,
          "audio_tokenizer: hop_length doesn't divide the sample rate");
  config.codebook_size = codec.Size("codebook_size");
  config.codebook_dim = codec.Size("codebook_dim");
  // Higgs Audio v2 quantizes with as many codebooks as its highest target
  // bandwidth (kbit/s) carries: bandwidth / (frame rate * bits per code).
  const std::vector<double> bandwidths = codec.Numbers("target_bandwidths");
  const double bits = std::log2(static_cast<double>(config.codebook_size));
  const double codebooks =
      1000.0 * bandwidths.back() / (config.FrameRate() * bits);
  Require(bits == std::floor(bits) && codebooks >= 1.0 &&
              codebooks == std::floor(codebooks),
          "audio_tokenizer: target_bandwidths imply a fractional codebook "
          "count");
  config.codebooks = static_cast<std::size_t>(codebooks);
  config.semantic = ParseSemantic(codec, codec.Object("semantic_model_config"));
  config.acoustic = ParseAcoustic(acoustic, config.frame_samples);
  // HuBERT frames (16 kHz / 320) are pooled down to the codec frame rate.
  const std::size_t semantic_rate =
      config.semantic.sample_rate / config.semantic.frame_samples;
  Require(config.semantic.sample_rate % config.semantic.frame_samples == 0 &&
              semantic_rate % config.FrameRate() == 0,
          "audio_tokenizer: the semantic frame rate is not a multiple of the "
          "codec frame rate");
  config.semantic_stride = semantic_rate / config.FrameRate();
  return config;
}

std::string ReadFile(const std::filesystem::path& path) {
  // Configuration files are a few KiB; refuse anything implausibly large.
  constexpr std::uintmax_t kMaxBytes = 1U << 20U;
  std::error_code code;
  const std::uintmax_t bytes = std::filesystem::file_size(path, code);
  if (code)
    throw ConfigError("cannot read " + path.string() + ": " + code.message());
  if (bytes > kMaxBytes)
    throw ConfigError(path.string() + " is too large for a configuration");
  std::ifstream stream(path, std::ios::binary);
  std::string text((std::istreambuf_iterator<char>(stream)),
                   std::istreambuf_iterator<char>());
  if (!stream.good() && !stream.eof())
    throw ConfigError("cannot read " + path.string());
  return text;
}

}  // namespace

std::optional<Config> ParseConfig(std::string_view model_json,
                                  std::string_view codec_json,
                                  std::string_view preprocessor_json,
                                  std::string* error) {
  try {
    const json::Value model_root = ParseJson(model_json, "config.json");
    const json::Value codec_root =
        ParseJson(codec_json, "audio_tokenizer/config.json");
    const json::Value preprocessor_root = ParseJson(
        preprocessor_json, "audio_tokenizer/preprocessor_config.json");
    const Fields model(model_root, "config.json");
    model.Expect("model_type", "omnivoice");
    Config config;
    config.backbone = ParseBackbone(model.Object("llm_config"));
    config.audio_codebooks = model.Size("num_audio_codebook");
    config.audio_vocab_size = model.Size("audio_vocab_size");
    config.audio_mask_id =
        static_cast<std::uint32_t>(model.Size("audio_mask_id"));
    config.codec = ParseCodec(codec_root, preprocessor_root);
    Require(config.audio_codebooks == config.codec.codebooks,
            "config.json: num_audio_codebook differs from the audio "
            "tokenizer's codebook count");
    Require(config.audio_vocab_size == config.codec.codebook_size + 1 &&
                config.audio_mask_id == config.codec.codebook_size,
            "config.json: the audio vocabulary must be the codebook plus one "
            "mask entry at its end");
    return config;
  } catch (const ConfigError& failure) {
    if (error != nullptr)
      *error = failure.what();
    return std::nullopt;
  }
}

std::optional<Config> LoadConfig(std::string_view model_root,
                                 std::string* error) {
  const std::filesystem::path root(model_root);
  std::string model, codec, preprocessor;
  try {
    model = ReadFile(root / "config.json");
    codec = ReadFile(root / "audio_tokenizer" / "config.json");
    preprocessor =
        ReadFile(root / "audio_tokenizer" / "preprocessor_config.json");
  } catch (const ConfigError& failure) {
    if (error != nullptr)
      *error = failure.what();
    return std::nullopt;
  }
  return ParseConfig(model, codec, preprocessor, error);
}

bool IsSupported(const Config& config, std::string* error) {
  const auto fail = [error](const char* message) {
    if (error != nullptr)
      *error = message;
    return false;
  };
  const BackboneConfig& backbone = config.backbone;
  if (backbone.head_dim != 128)
    return fail("the backbone's attention kernels need a head size of 128");
  if (config.codec.codebooks != 8)
    return fail("the generator handles exactly 8 audio codebooks");
  if (config.codec.semantic.conv_channels.front() !=
      config.codec.semantic.conv_channels.back())
    return fail("the semantic feature extractor must keep its width");
  return true;
}

}  // namespace gufo::models::omnivoice
