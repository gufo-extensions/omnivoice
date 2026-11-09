#ifndef GUFO_MODELS_OMNIVOICE_TESTS_CONFIG_SAMPLES_HPP_
#define GUFO_MODELS_OMNIVOICE_TESTS_CONFIG_SAMPLES_HPP_

#include <stdexcept>
#include <string>
#include <string_view>

namespace omnivoice_test {

// The fields this extension reads from k2-fsa/OmniVoice's three
// configuration files, with the released checkpoint's values.
inline constexpr std::string_view kModelConfig = R"({
  "architectures": ["OmniVoice"],
  "model_type": "omnivoice",
  "audio_mask_id": 1024,
  "audio_vocab_size": 1025,
  "num_audio_codebook": 8,
  "llm_config": {
    "model_type": "qwen3",
    "attention_bias": false,
    "head_dim": 128,
    "hidden_act": "silu",
    "hidden_size": 1024,
    "intermediate_size": 3072,
    "max_position_embeddings": 40960,
    "num_attention_heads": 16,
    "num_hidden_layers": 28,
    "num_key_value_heads": 8,
    "rms_norm_eps": 1e-06,
    "rope_parameters": {"rope_theta": 1000000, "rope_type": "default"},
    "vocab_size": 151676
  }
})";

inline constexpr std::string_view kCodecConfig = R"({
  "model_type": "higgs_audio_v2_tokenizer",
  "codebook_dim": 64,
  "codebook_size": 1024,
  "downsample_factor": 320,
  "sample_rate": 24000,
  "semantic_sample_rate": 16000,
  "target_bandwidths": [0.5, 1, 1.5, 2],
  "acoustic_model_config": {
    "model_type": "dac",
    "decoder_hidden_size": 1024,
    "downsampling_ratios": [8, 5, 4, 2, 3],
    "encoder_hidden_size": 64,
    "hidden_size": 256,
    "hop_length": 960,
    "sampling_rate": 16000,
    "upsampling_ratios": [8, 5, 4, 2, 3]
  },
  "semantic_model_config": {
    "model_type": "hubert",
    "conv_bias": false,
    "conv_dim": [512, 512, 512, 512, 512, 512, 512],
    "conv_kernel": [10, 3, 3, 3, 3, 2, 2],
    "conv_stride": [5, 2, 2, 2, 2, 2, 2],
    "do_stable_layer_norm": false,
    "feat_extract_activation": "gelu",
    "feat_extract_norm": "group",
    "feat_proj_layer_norm": true,
    "hidden_act": "gelu",
    "hidden_size": 768,
    "intermediate_size": 3072,
    "layer_norm_eps": 1e-05,
    "num_attention_heads": 12,
    "num_conv_pos_embedding_groups": 16,
    "num_conv_pos_embeddings": 128,
    "num_hidden_layers": 12
  }
})";

inline constexpr std::string_view kPreprocessorConfig = R"({
  "feature_extractor_type": "DacFeatureExtractor",
  "hop_length": 960,
  "sampling_rate": 24000
})";

/// `text` with its single occurrence of `from` replaced by `to`.
inline std::string Replace(std::string_view text, std::string_view from,
                           std::string_view to) {
  std::string result(text);
  const std::size_t at = result.find(from);
  if (at == std::string::npos || result.find(from, at + 1) != std::string::npos)
    throw std::logic_error("sample edit must match once: " + std::string(from));
  result.replace(at, from.size(), to);
  return result;
}

}  // namespace omnivoice_test

#endif  // GUFO_MODELS_OMNIVOICE_TESTS_CONFIG_SAMPLES_HPP_
