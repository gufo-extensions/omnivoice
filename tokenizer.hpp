#ifndef GUFO_MODELS_OMNIVOICE_TOKENIZER_HPP_
#define GUFO_MODELS_OMNIVOICE_TOKENIZER_HPP_

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace gufo::models::omnivoice {

/// OmniVoice's Qwen2 byte-level BPE tokenizer, read from `tokenizer.json`.
///
/// Encodes like Hugging Face's `tokenizer(text, add_special_tokens=False)`:
/// added tokens match verbatim; the text between them is NFC-normalized,
/// split with the Qwen2 pre-tokenizer pattern and byte-pair encoded.
class Tokenizer {
public:
  [[nodiscard]] static std::unique_ptr<Tokenizer> Load(
      const std::filesystem::path& tokenizer_json,
      std::string* error = nullptr);

  [[nodiscard]] bool Encode(std::string_view text,
                            std::vector<std::uint32_t>* ids,
                            std::string* error = nullptr) const;

  /// The ID of an added token such as `<|text_start|>`.
  [[nodiscard]] std::optional<std::uint32_t> AddedTokenId(
      std::string_view content) const;

  /// One past the largest token ID.
  [[nodiscard]] std::size_t vocabulary_size() const noexcept {
    return vocabulary_size_;
  }

private:
  struct AddedToken {
    std::string content;
    std::uint32_t id = 0;
  };

  Tokenizer() = default;

  [[nodiscard]] bool EncodeOrdinary(std::string_view text,
                                    std::vector<std::uint32_t>* ids,
                                    std::string* error) const;
  [[nodiscard]] bool EncodePiece(std::string_view piece,
                                 std::vector<std::uint32_t>* ids,
                                 std::string* error) const;

  std::unordered_map<std::string, std::uint32_t> vocabulary_;
  std::unordered_map<std::string, std::uint32_t> merge_ranks_;
  /// Longest first, so the leftmost match is also the longest.
  std::vector<AddedToken> added_tokens_;
  std::array<std::string, 256> byte_symbols_;
  std::size_t vocabulary_size_ = 0;
};

}  // namespace gufo::models::omnivoice

#endif  // GUFO_MODELS_OMNIVOICE_TOKENIZER_HPP_
