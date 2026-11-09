#ifndef GUFO_MODELS_OMNIVOICE_CONFIG_HPP_
#define GUFO_MODELS_OMNIVOICE_CONFIG_HPP_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace gufo::models::omnivoice {

/// The Qwen3 backbone (`llm_config`). OmniVoice runs it bidirectionally over
/// text and audio-code positions; its text LM head is unused.
struct BackboneConfig {
  std::size_t vocab_size = 0;
  std::size_t hidden_size = 0;
  std::size_t intermediate_size = 0;
  std::size_t layers = 0;
  std::size_t heads = 0;
  std::size_t kv_heads = 0;
  std::size_t head_dim = 0;
  std::size_t max_positions = 0;
  float rms_norm_eps = 0.0F;
  float rope_theta = 0.0F;
};

/// HuBERT encoder of the audio tokenizer's semantic branch, at
/// `sample_rate` with one frame per `frame_samples` input samples.
struct SemanticEncoderConfig {
  std::uint32_t sample_rate = 0;
  std::size_t frame_samples = 0;
  std::size_t hidden_size = 0;
  std::size_t intermediate_size = 0;
  std::size_t heads = 0;
  std::size_t layers = 0;
  std::size_t position_kernel = 0;
  std::size_t position_groups = 0;
  float layer_norm_eps = 0.0F;
  std::vector<std::size_t> conv_channels;
  std::vector<std::size_t> conv_kernels;
  std::vector<std::size_t> conv_strides;
};

/// DAC encoder and decoder of the audio tokenizer's acoustic branch.
struct AcousticConfig {
  std::size_t encoder_channels = 0;
  std::size_t decoder_channels = 0;
  std::size_t latent_channels = 0;
  std::vector<std::size_t> downsampling_ratios;
  std::vector<std::size_t> upsampling_ratios;
};

/// Higgs Audio v2 tokenizer: waveform at `sample_rate` <-> residual vector
/// quantized codes, one frame per `frame_samples` samples.
struct CodecConfig {
  std::uint32_t sample_rate = 0;
  std::size_t frame_samples = 0;
  std::size_t codebooks = 0;
  std::size_t codebook_size = 0;
  std::size_t codebook_dim = 0;
  /// Semantic encoder frames per codec frame.
  std::size_t semantic_stride = 0;
  SemanticEncoderConfig semantic;
  AcousticConfig acoustic;

  [[nodiscard]] std::uint32_t FrameRate() const {
    return frame_samples == 0
               ? 0
               : sample_rate / static_cast<std::uint32_t>(frame_samples);
  }
};

struct Config {
  BackboneConfig backbone;
  /// Codebooks the backbone embeds and predicts per audio frame.
  std::size_t audio_codebooks = 0;
  /// Entries per codebook, including the mask.
  std::size_t audio_vocab_size = 0;
  std::uint32_t audio_mask_id = 0;
  CodecConfig codec;
};

/// Parses `config.json`, `audio_tokenizer/config.json` and
/// `audio_tokenizer/preprocessor_config.json` and checks that they describe a
/// consistent OmniVoice checkpoint.
[[nodiscard]] std::optional<Config> ParseConfig(
    std::string_view model_json, std::string_view codec_json,
    std::string_view preprocessor_json, std::string* error = nullptr);

/// Reads and parses the three files of the model directory `model_root`.
[[nodiscard]] std::optional<Config> LoadConfig(std::string_view model_root,
                                               std::string* error = nullptr);

/// True when this extension implements the checkpoint's architecture.
[[nodiscard]] bool IsSupported(const Config& config,
                               std::string* error = nullptr);

}  // namespace gufo::models::omnivoice

#endif  // GUFO_MODELS_OMNIVOICE_CONFIG_HPP_
