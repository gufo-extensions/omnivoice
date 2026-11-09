#ifndef GUFO_MODELS_OMNIVOICE_UNICODE_HPP_
#define GUFO_MODELS_OMNIVOICE_UNICODE_HPP_

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace gufo::models::omnivoice {

/// One code point of a UTF-8 string and its byte range.
struct Codepoint {
  std::uint32_t value = 0;
  std::size_t begin = 0;
  std::size_t end = 0;
};

/// Decodes `text`; false on malformed UTF-8 (overlong forms, surrogates).
[[nodiscard]] bool DecodeUtf8(std::string_view text,
                              std::vector<Codepoint>* output);

void AppendUtf8(std::uint32_t value, std::string* output);

/// Python's str.isspace(): Unicode White_Space plus the ASCII information
/// separators U+001C..U+001F. Oniguruma's and Python's `\s` agree with it.
[[nodiscard]] bool IsSpace(std::uint32_t value);

/// The CJK Unified Ideographs block U+4E00..U+9FFF that OmniVoice uses to
/// detect Chinese text.
[[nodiscard]] constexpr bool IsCjkIdeograph(std::uint32_t value) {
  return value >= 0x4E00U && value <= 0x9FFFU;
}

[[nodiscard]] bool ContainsCjkIdeograph(std::string_view text);

/// Python's str.strip() without arguments. `text` must be valid UTF-8.
[[nodiscard]] std::string StripSpace(std::string_view text);

/// Python's str.lower(), via ICU's root-locale full case mapping.
[[nodiscard]] std::string ToLower(std::string_view text);

}  // namespace gufo::models::omnivoice

#endif  // GUFO_MODELS_OMNIVOICE_UNICODE_HPP_
