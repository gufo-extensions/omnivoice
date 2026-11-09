#include "probe.hpp"

#include <array>
#include <filesystem>
#include <string>
#include <system_error>

#include "config.hpp"

namespace gufo::models::omnivoice {

bool ProbeOmniVoice(std::string_view model_root) {
  std::error_code code;
  const std::filesystem::path root(model_root);
  if (!std::filesystem::is_directory(root, code) ||
      !std::filesystem::is_regular_file(root / "config.json", code))
    return false;
  const auto config = LoadConfig(model_root);
  return config.has_value() && IsSupported(*config);
}

bool ResolveOmniVoiceArtifacts(std::string_view model_root,
                               std::string* error) {
  const auto config = LoadConfig(model_root, error);
  if (!config.has_value() || !IsSupported(*config, error))
    return false;
  const std::filesystem::path root(model_root);
  constexpr std::array<const char*, 3> kFiles = {
      "model.safetensors", "tokenizer.json",
      "audio_tokenizer/model.safetensors"};
  for (const char* file : kFiles) {
    std::error_code code;
    if (!std::filesystem::is_regular_file(root / file, code)) {
      if (error != nullptr)
        *error = "OmniVoice model directory lacks " + std::string(file);
      return false;
    }
  }
  return true;
}

}  // namespace gufo::models::omnivoice
