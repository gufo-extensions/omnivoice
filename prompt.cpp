// Ported from k2-fsa/OmniVoice's omnivoice/models/omnivoice.py
// (_prepare_inference_inputs, _tokenize_with_nonverbal_tags; Apache-2.0, see
// licenses/OmniVoice.txt).
#include "prompt.hpp"

#include <array>

#include "text.hpp"

namespace gufo::models::omnivoice {
namespace {

constexpr std::array<std::string_view, 13> kNonverbalTags = {
    "[laughter]",           "[sigh]",        "[confirmation-en]",
    "[question-en]",        "[question-ah]", "[question-oh]",
    "[question-ei]",        "[question-yi]", "[surprise-ah]",
    "[surprise-oh]",        "[surprise-wa]", "[surprise-yo]",
    "[dissatisfaction-hnn]"};

bool Append(const Tokenizer& tokenizer, std::string_view text,
            std::vector<std::uint32_t>* ids, std::string* error) {
  std::vector<std::uint32_t> piece;
  if (!tokenizer.Encode(text, &piece, error))
    return false;
  ids->insert(ids->end(), piece.begin(), piece.end());
  return true;
}

}  // namespace

bool EncodeWithNonverbalTags(const Tokenizer& tokenizer, std::string_view text,
                             std::vector<std::uint32_t>* ids,
                             std::string* error) {
  ids->clear();
  std::size_t start = 0;
  while (true) {
    std::size_t tag_at = std::string_view::npos;
    std::string_view tag;
    for (const std::string_view candidate : kNonverbalTags) {
      const std::size_t at = text.find(candidate, start);
      if (at < tag_at) {
        tag_at = at;
        tag = candidate;
      }
    }
    const std::size_t end =
        tag_at == std::string_view::npos ? text.size() : tag_at;
    if (end > start &&
        !Append(tokenizer, text.substr(start, end - start), ids, error))
      return false;
    if (tag_at == std::string_view::npos)
      return true;
    if (!Append(tokenizer, tag, ids, error))
      return false;
    start = tag_at + tag.size();
  }
}

bool EncodePromptText(const Tokenizer& tokenizer, const PromptText& prompt,
                      std::vector<std::uint32_t>* ids, std::string* error) {
  const auto or_none = [](const std::optional<std::string>& value) {
    return value.has_value() && !value->empty() ? *value : std::string("None");
  };
  std::string style;
  if (prompt.denoise && prompt.has_reference)
    style += "<|denoise|>";
  style += "<|lang_start|>" + or_none(prompt.language) + "<|lang_end|>";
  style += "<|instruct_start|>" + or_none(prompt.instruct) + "<|instruct_end|>";
  if (!tokenizer.Encode(style, ids, error))
    return false;
  const std::string text = "<|text_start|>" +
                           CombineText(prompt.text, prompt.reference_text) +
                           "<|text_end|>";
  std::vector<std::uint32_t> text_ids;
  if (!EncodeWithNonverbalTags(tokenizer, text, &text_ids, error))
    return false;
  ids->insert(ids->end(), text_ids.begin(), text_ids.end());
  return true;
}

}  // namespace gufo::models::omnivoice
