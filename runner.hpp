#ifndef GUFO_MODELS_OMNIVOICE_RUNNER_HPP_
#define GUFO_MODELS_OMNIVOICE_RUNNER_HPP_

#include <gufo/ext/tts.hpp>
#include <memory>
#include <string>

namespace gufo::models::omnivoice {

/// Gufo's speech synthesis runner for an OmniVoice model directory.
[[nodiscard]] std::unique_ptr<ext::TtsRunner> CreateOmniVoiceRunner(
    const ext::TtsRunnerOptions& options, std::string* error);

}  // namespace gufo::models::omnivoice

#endif  // GUFO_MODELS_OMNIVOICE_RUNNER_HPP_
