#include "probe.hpp"

#include <unistd.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>

#include "config_samples.hpp"

namespace fs = std::filesystem;
namespace ov = gufo::models::omnivoice;

namespace {

constexpr int kSkip = 77;

void Require(bool condition, const std::string& message) {
  if (!condition)
    throw std::runtime_error(message);
}

void Write(const fs::path& path, std::string_view text) {
  fs::create_directories(path.parent_path());
  std::ofstream(path, std::ios::binary) << text;
}

/// A fresh directory with OmniVoice's three configuration files.
class SampleDirectory {
public:
  explicit SampleDirectory(std::string_view model_config) {
    root_ = fs::temp_directory_path() /
            ("omnivoice_probe_test_" + std::to_string(::getpid()));
    fs::remove_all(root_);
    Write(root_ / "config.json", model_config);
    Write(root_ / "audio_tokenizer" / "config.json",
          omnivoice_test::kCodecConfig);
    Write(root_ / "audio_tokenizer" / "preprocessor_config.json",
          omnivoice_test::kPreprocessorConfig);
  }
  ~SampleDirectory() {
    std::error_code ignored;
    fs::remove_all(root_, ignored);
  }
  SampleDirectory(const SampleDirectory&) = delete;
  SampleDirectory& operator=(const SampleDirectory&) = delete;

  [[nodiscard]] const fs::path& root() const { return root_; }

private:
  fs::path root_;
};

void TestSampleDirectories() {
  {
    const SampleDirectory directory(omnivoice_test::kModelConfig);
    Require(ov::ProbeOmniVoice(directory.root().string()),
            "probe rejects the released configuration");
    std::string error;
    Require(!ov::ResolveOmniVoiceArtifacts(directory.root().string(), &error) &&
                error.find("model.safetensors") != std::string::npos,
            "resolve accepts a directory without weights: " + error);
    Require(!ov::ProbeOmniVoice((directory.root() / "config.json").string()),
            "probe accepts a file");
  }
  {
    const SampleDirectory directory(omnivoice_test::Replace(
        omnivoice_test::kModelConfig, R"("model_type": "omnivoice")",
        R"("model_type": "qwen3_tts")"));
    Require(!ov::ProbeOmniVoice(directory.root().string()),
            "probe accepts another model type");
  }
  Require(!ov::ProbeOmniVoice("/nonexistent/omnivoice"),
          "probe accepts a missing directory");
}

/// The released checkpoint named by GUFO_OMNIVOICE_MODEL.
bool TestReleasedDirectory() {
  const char* root = std::getenv("GUFO_OMNIVOICE_MODEL");
  if (root == nullptr || *root == '\0')
    return false;
  std::string error;
  Require(ov::ProbeOmniVoice(root), "probe rejects GUFO_OMNIVOICE_MODEL");
  Require(ov::ResolveOmniVoiceArtifacts(root, &error),
          "resolve rejects GUFO_OMNIVOICE_MODEL: " + error);
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  const std::string_view mode = argc > 1 ? argv[1] : "samples";
  try {
    if (mode == "samples") {
      TestSampleDirectories();
    } else if (mode == "model") {
      if (!TestReleasedDirectory()) {
        std::cout << "skip: GUFO_OMNIVOICE_MODEL is not set\n";
        return kSkip;
      }
    } else {
      std::cerr << "usage: omnivoice_probe_test [samples|model]\n";
      return 2;
    }
  } catch (const std::exception& error) {
    std::cout << "FAIL " << mode << ": " << error.what() << '\n';
    return 1;
  }
  std::cout << "ok   " << mode << '\n';
  return 0;
}
