#ifndef GUFO_MODELS_OMNIVOICE_PROBE_HPP_
#define GUFO_MODELS_OMNIVOICE_PROBE_HPP_

#include <string>
#include <string_view>

namespace gufo::models::omnivoice {

/// `model_root` is a directory holding a supported OmniVoice checkpoint's
/// configuration (Hugging Face layout: `config.json` with model_type
/// "omnivoice" plus `audio_tokenizer/`).
[[nodiscard]] bool ProbeOmniVoice(std::string_view model_root);

/// Checks that the configuration is supported and the weight and tokenizer
/// files exist.
[[nodiscard]] bool ResolveOmniVoiceArtifacts(std::string_view model_root,
                                             std::string* error = nullptr);

}  // namespace gufo::models::omnivoice

#endif  // GUFO_MODELS_OMNIVOICE_PROBE_HPP_
