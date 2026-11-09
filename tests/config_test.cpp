#include "config.hpp"

#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "config_samples.hpp"

namespace ov = gufo::models::omnivoice;
using omnivoice_test::kCodecConfig;
using omnivoice_test::kModelConfig;
using omnivoice_test::kPreprocessorConfig;
using omnivoice_test::Replace;

namespace {

void Require(bool condition, const std::string& message) {
  if (!condition)
    throw std::runtime_error(message);
}

void TestReleasedCheckpoint() {
  std::string error;
  const auto config =
      ov::ParseConfig(kModelConfig, kCodecConfig, kPreprocessorConfig, &error);
  Require(config.has_value(), "released configuration rejected: " + error);
  const ov::BackboneConfig& backbone = config->backbone;
  Require(backbone.vocab_size == 151676 && backbone.hidden_size == 1024 &&
              backbone.intermediate_size == 3072 && backbone.layers == 28 &&
              backbone.heads == 16 && backbone.kv_heads == 8 &&
              backbone.head_dim == 128 && backbone.max_positions == 40960,
          "backbone shape");
  Require(backbone.rms_norm_eps == 1e-6F && backbone.rope_theta == 1e6F,
          "backbone norm epsilon and RoPE base");
  Require(config->audio_codebooks == 8 && config->audio_vocab_size == 1025 &&
              config->audio_mask_id == 1024,
          "audio vocabulary");
  const ov::CodecConfig& codec = config->codec;
  Require(codec.sample_rate == 24000 && codec.frame_samples == 960 &&
              codec.FrameRate() == 25,
          "codec frame rate");
  Require(codec.codebooks == 8 && codec.codebook_size == 1024 &&
              codec.codebook_dim == 64,
          "codec codebooks");
  Require(codec.semantic_stride == 2 && codec.semantic.sample_rate == 16000 &&
              codec.semantic.frame_samples == 320 &&
              codec.semantic.layers == 12 && codec.semantic.heads == 12 &&
              codec.semantic.position_kernel == 128 &&
              codec.semantic.position_groups == 16 &&
              codec.semantic.conv_strides.size() == 7,
          "semantic encoder");
  Require(codec.acoustic.encoder_channels == 64 &&
              codec.acoustic.decoder_channels == 1024 &&
              codec.acoustic.latent_channels == 256 &&
              codec.acoustic.upsampling_ratios ==
                  std::vector<std::size_t>({8, 5, 4, 2, 3}),
          "acoustic codec");
  Require(ov::IsSupported(*config, &error),
          "released checkpoint unsupported: " + error);
}

void ExpectRejected(std::string_view name, const std::string& model,
                    const std::string& codec, const std::string& preprocessor,
                    std::string_view reason) {
  std::string error;
  const auto config = ov::ParseConfig(model, codec, preprocessor, &error);
  Require(!config.has_value(), std::string(name) + " was accepted");
  Require(error.find(reason) != std::string::npos,
          std::string(name) + " failed for another reason: " + error);
}

void TestInconsistentConfigurations() {
  const std::string model(kModelConfig), codec(kCodecConfig),
      preprocessor(kPreprocessorConfig);
  ExpectRejected("another model type",
                 Replace(model, R"("model_type": "omnivoice")",
                         R"("model_type": "qwen3_tts")"),
                 codec, preprocessor, "model_type");
  ExpectRejected("a codebook count mismatch",
                 Replace(model, R"("num_audio_codebook": 8)",
                         R"("num_audio_codebook": 9)"),
                 codec, preprocessor, "num_audio_codebook");
  ExpectRejected(
      "a mask id inside the codebook",
      Replace(model, R"("audio_mask_id": 1024)", R"("audio_mask_id": 0)"),
      codec, preprocessor, "mask");
  ExpectRejected("a missing RoPE base",
                 Replace(model, R"("rope_theta": 1000000, )", ""), codec,
                 preprocessor, "rope_theta");
  ExpectRejected("grouped heads that don't divide",
                 Replace(model, R"("num_key_value_heads": 8)",
                         R"("num_key_value_heads": 6)"),
                 codec, preprocessor, "KV heads");
  ExpectRejected("a hop the strides don't reach", model,
                 Replace(codec, R"("hop_length": 960)", R"("hop_length": 480)"),
                 preprocessor, "hop_length");
  ExpectRejected("a layer-norm HuBERT variant", model,
                 Replace(codec, R"("feat_extract_norm": "group")",
                         R"("feat_extract_norm": "layer")"),
                 preprocessor, "feat_extract_norm");
  ExpectRejected("a bandwidth between codebook counts", model,
                 Replace(codec, R"([0.5, 1, 1.5, 2])", R"([0.5, 1, 1.5, 2.1])"),
                 preprocessor, "fractional");
  ExpectRejected("malformed JSON", model, codec, "{", "preprocessor_config");
}

void TestUnsupportedShapes() {
  const auto config = ov::ParseConfig(
      Replace(kModelConfig, R"("head_dim": 128)", R"("head_dim": 64)"),
      kCodecConfig, kPreprocessorConfig);
  Require(config.has_value(), "head size 64 should parse");
  std::string error;
  Require(!ov::IsSupported(*config, &error) &&
              error.find("head size") != std::string::npos,
          "head size 64 should be unsupported");
}

}  // namespace

int main() {
  const std::vector<std::pair<const char*, std::function<void()>>> tests = {
      {"released checkpoint", TestReleasedCheckpoint},
      {"inconsistent configurations", TestInconsistentConfigurations},
      {"unsupported shapes", TestUnsupportedShapes},
  };
  int failures = 0;
  for (const auto& [name, test] : tests) {
    try {
      test();
      std::cout << "ok   " << name << '\n';
    } catch (const std::exception& error) {
      std::cout << "FAIL " << name << ": " << error.what() << '\n';
      ++failures;
    }
  }
  return failures == 0 ? 0 : 1;
}
