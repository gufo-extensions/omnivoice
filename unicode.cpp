// DecodeUtf8 and AppendUtf8 are adapted from Gufo's Qwen3-TTS tokenizer
// (MIT, see licenses/gufo.txt).
#include "unicode.hpp"

#include <unicode/locid.h>
#include <unicode/stringpiece.h>
#include <unicode/uchar.h>
#include <unicode/unistr.h>

namespace gufo::models::omnivoice {

bool DecodeUtf8(std::string_view text, std::vector<Codepoint>* output) {
  output->clear();
  for (std::size_t index = 0; index < text.size();) {
    const std::size_t begin = index;
    const auto first = static_cast<unsigned char>(text[index++]);
    std::uint32_t value = 0;
    std::size_t continuation = 0;
    if (first <= 0x7FU) {
      value = first;
    } else if (first >= 0xC2U && first <= 0xDFU) {
      value = first & 0x1FU;
      continuation = 1;
    } else if (first >= 0xE0U && first <= 0xEFU) {
      value = first & 0x0FU;
      continuation = 2;
    } else if (first >= 0xF0U && first <= 0xF4U) {
      value = first & 0x07U;
      continuation = 3;
    } else {
      return false;
    }
    if (continuation > text.size() - index)
      return false;
    for (std::size_t offset = 0; offset < continuation; ++offset) {
      const auto next = static_cast<unsigned char>(text[index++]);
      if ((next & 0xC0U) != 0x80U)
        return false;
      value = (value << 6U) | (next & 0x3FU);
    }
    if ((continuation == 1 && value < 0x80U) ||
        (continuation == 2 &&
         (value < 0x800U || (value >= 0xD800U && value <= 0xDFFFU))) ||
        (continuation == 3 && (value < 0x10000U || value > 0x10FFFFU)))
      return false;
    output->push_back({value, begin, index});
  }
  return true;
}

void AppendUtf8(std::uint32_t value, std::string* output) {
  if (value <= 0x7FU) {
    output->push_back(static_cast<char>(value));
  } else if (value <= 0x7FFU) {
    output->push_back(static_cast<char>(0xC0U | (value >> 6U)));
    output->push_back(static_cast<char>(0x80U | (value & 0x3FU)));
  } else if (value <= 0xFFFFU) {
    output->push_back(static_cast<char>(0xE0U | (value >> 12U)));
    output->push_back(static_cast<char>(0x80U | ((value >> 6U) & 0x3FU)));
    output->push_back(static_cast<char>(0x80U | (value & 0x3FU)));
  } else {
    output->push_back(static_cast<char>(0xF0U | (value >> 18U)));
    output->push_back(static_cast<char>(0x80U | ((value >> 12U) & 0x3FU)));
    output->push_back(static_cast<char>(0x80U | ((value >> 6U) & 0x3FU)));
    output->push_back(static_cast<char>(0x80U | (value & 0x3FU)));
  }
}

bool IsSpace(std::uint32_t value) {
  return u_isUWhiteSpace(static_cast<UChar32>(value)) != 0 ||
         (value >= 0x1CU && value <= 0x1FU);
}

bool ContainsCjkIdeograph(std::string_view text) {
  std::vector<Codepoint> points;
  if (!DecodeUtf8(text, &points))
    return false;
  for (const Codepoint& point : points) {
    if (IsCjkIdeograph(point.value))
      return true;
  }
  return false;
}

std::string StripSpace(std::string_view text) {
  std::vector<Codepoint> points;
  if (!DecodeUtf8(text, &points))
    return std::string(text);
  std::size_t first = 0;
  std::size_t last = points.size();
  while (first < last && IsSpace(points[first].value))
    ++first;
  while (last > first && IsSpace(points[last - 1].value))
    --last;
  if (first == last)
    return {};
  return std::string(text.substr(points[first].begin,
                                 points[last - 1].end - points[first].begin));
}

std::string ToLower(std::string_view text) {
  icu::UnicodeString unicode = icu::UnicodeString::fromUTF8(
      icu::StringPiece(text.data(), static_cast<std::int32_t>(text.size())));
  unicode.toLower(icu::Locale::getRoot());
  std::string lower;
  unicode.toUTF8String(lower);
  return lower;
}

}  // namespace gufo::models::omnivoice
