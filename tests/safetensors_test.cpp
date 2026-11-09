#include "safetensors.hpp"

#include <unistd.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace fs = std::filesystem;
namespace ov = gufo::models::omnivoice;

namespace {

constexpr int kSkip = 77;

void Require(bool condition, const std::string& message) {
  if (!condition)
    throw std::runtime_error(message);
}

fs::path TempPath(std::string_view name) {
  return fs::temp_directory_path() /
         ("omnivoice_safetensors_test_" + std::to_string(::getpid()) + "_" +
          std::string(name));
}

/// Writes a safetensors file with this header and payload; removes it when
/// destroyed.
class SampleFile {
public:
  SampleFile(std::string_view name, std::string_view header,
             const std::vector<std::uint8_t>& payload)
      : path_(TempPath(name)) {
    std::ofstream stream(path_, std::ios::binary);
    const std::uint64_t length = header.size();
    for (int index = 0; index < 8; ++index)
      stream.put(static_cast<char>((length >> (8 * index)) & 0xFFU));
    stream.write(header.data(), static_cast<std::streamsize>(header.size()));
    stream.write(reinterpret_cast<const char*>(payload.data()),
                 static_cast<std::streamsize>(payload.size()));
  }
  ~SampleFile() {
    std::error_code ignored;
    fs::remove(path_, ignored);
  }
  SampleFile(const SampleFile&) = delete;
  SampleFile& operator=(const SampleFile&) = delete;

  [[nodiscard]] const fs::path& path() const { return path_; }

private:
  fs::path path_;
};

void TestWellFormedFile() {
  // A 2x3 F32 tensor (24 bytes) and a 4-element BF16 tensor (8 bytes).
  std::vector<std::uint8_t> payload(32);
  const float values[6] = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F};
  std::memcpy(payload.data(), values, sizeof(values));
  for (int index = 0; index < 8; ++index)
    payload[24 + index] = static_cast<std::uint8_t>(index);
  const SampleFile file(
      "valid.safetensors",
      R"({"__metadata__":{"format":"pt"},)"
      R"("b":{"dtype":"BF16","shape":[4],"data_offsets":[24,32]},)"
      R"("a":{"dtype":"F32","shape":[2,3],"data_offsets":[0,24]}})",
      payload);
  std::string error;
  const auto opened = ov::SafetensorsFile::Open(file.path(), &error);
  Require(opened != nullptr, "valid file rejected: " + error);
  Require(opened->tensors().size() == 2, "tensor count");
  const ov::Tensor* a = opened->Expect("a", ov::DType::kF32, {2, 3}, &error);
  Require(a != nullptr, "a: " + error);
  float read[6];
  std::memcpy(read, a->data, sizeof(read));
  Require(read[0] == 1.0F && read[5] == 6.0F && a->Elements() == 6,
          "a's payload");
  const ov::Tensor* b = opened->Find("b");
  Require(b != nullptr && b->dtype == ov::DType::kBF16 && b->bytes == 8 &&
              std::to_integer<int>(b->data[7]) == 7,
          "b's payload");
  Require(opened->Expect("a", ov::DType::kF32, {3, 2}, &error) == nullptr &&
              error.find("expected F32 [3, 2]") != std::string::npos,
          "a shape mismatch reported as: " + error);
  Require(opened->Expect("c", ov::DType::kF32, {1}, &error) == nullptr &&
              error.find("lacks c") != std::string::npos,
          "missing tensor reported as: " + error);
  Require(reinterpret_cast<const std::byte*>(b->data) - a->data ==
              static_cast<std::ptrdiff_t>(b->offset - a->offset),
          "offsets");
}

void ExpectRejected(std::string_view name, std::string_view header,
                    std::size_t payload_bytes, std::string_view reason) {
  const SampleFile file(std::string(name) + ".safetensors", header,
                        std::vector<std::uint8_t>(payload_bytes));
  std::string error;
  Require(ov::SafetensorsFile::Open(file.path(), &error) == nullptr,
          std::string(name) + " accepted");
  Require(error.find(reason) != std::string::npos,
          std::string(name) + " rejected for another reason: " + error);
}

void TestMalformedFiles() {
  ExpectRejected(
      "unknown_dtype",
      R"({"a":{"dtype":"F8_E4M3","shape":[1],"data_offsets":[0,1]}})", 1,
      "unsupported dtype");
  ExpectRejected("short_payload",
                 R"({"a":{"dtype":"F32","shape":[4],"data_offsets":[0,16]}})",
                 8, "outside the payload");
  ExpectRejected("size_mismatch",
                 R"({"a":{"dtype":"F32","shape":[3],"data_offsets":[0,16]}})",
                 16, "don't hold shape [3]");
  ExpectRejected("overlap",
                 R"({"a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},)"
                 R"("b":{"dtype":"F32","shape":[2],"data_offsets":[4,12]}})",
                 12, "overlaps");
  ExpectRejected("not_json", "{\"a\":", 0, "header");
  ExpectRejected("empty", "{}", 0, "no tensors");
}

void ExpectTensor(const ov::SafetensorsFile& file, std::string_view name,
                  std::initializer_list<std::size_t> shape) {
  std::string error;
  Require(file.Expect(name, ov::DType::kF32, shape, &error) != nullptr, error);
}

/// The released checkpoint's two weight files.
bool TestReleasedFiles() {
  const char* root = std::getenv("GUFO_OMNIVOICE_MODEL");
  if (root == nullptr || *root == '\0')
    return false;
  std::string error;
  const auto model =
      ov::SafetensorsFile::Open(fs::path(root) / "model.safetensors", &error);
  Require(model != nullptr, error);
  Require(model->tensors().size() == 313, "generator tensor count");
  ExpectTensor(*model, "llm.embed_tokens.weight", {151676, 1024});
  ExpectTensor(*model, "audio_embeddings.weight", {8200, 1024});
  ExpectTensor(*model, "audio_heads.weight", {8200, 1024});
  ExpectTensor(*model, "llm.layers.27.self_attn.q_proj.weight", {2048, 1024});
  ExpectTensor(*model, "llm.layers.0.self_attn.k_norm.weight", {128});
  const auto codec = ov::SafetensorsFile::Open(
      fs::path(root) / "audio_tokenizer" / "model.safetensors", &error);
  Require(codec != nullptr, error);
  Require(codec->tensors().size() == 527, "audio tokenizer tensor count");
  ExpectTensor(*codec, "acoustic_decoder.conv1.weight", {1024, 256, 7});
  ExpectTensor(*codec, "quantizer.quantizers.7.codebook.embed", {1024, 64});
  ExpectTensor(*codec,
               "semantic_model.encoder.pos_conv_embed.conv."
               "parametrizations.weight.original1",
               {768, 48, 128});
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  const std::string_view mode = argc > 1 ? argv[1] : "samples";
  try {
    if (mode == "samples") {
      TestWellFormedFile();
      TestMalformedFiles();
    } else if (mode == "model") {
      if (!TestReleasedFiles()) {
        std::cout << "skip: GUFO_OMNIVOICE_MODEL is not set\n";
        return kSkip;
      }
    } else {
      std::cerr << "usage: omnivoice_safetensors_test [samples|model]\n";
      return 2;
    }
  } catch (const std::exception& error) {
    std::cout << "FAIL " << mode << ": " << error.what() << '\n';
    return 1;
  }
  std::cout << "ok   " << mode << '\n';
  return 0;
}
