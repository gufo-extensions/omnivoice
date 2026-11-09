#ifndef GUFO_MODELS_OMNIVOICE_PROMPT_HPP_
#define GUFO_MODELS_OMNIVOICE_PROMPT_HPP_

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "tokenizer.hpp"

namespace gufo::models::omnivoice {

/// The text side of one generation: what to say, the reference transcript
/// when cloning, the resolved language ID and voice-design instruction.
struct PromptText {
  std::string text;
  std::optional<std::string> reference_text;
  std::optional<std::string> language;
  std::optional<std::string> instruct;
  /// Adds `<|denoise|>` when a reference voice is present.
  bool denoise = true;
  bool has_reference = false;
};

/// `_prepare_inference_inputs`' text tokens, which precede the reference and
/// target audio frames:
///   [<|denoise|>]<|lang_start|>LANG<|lang_end|>
///   <|instruct_start|>INSTRUCT<|instruct_end|>
///   <|text_start|>REFERENCE TEXT<|text_end|>
/// Non-verbal tags such as [laughter] are encoded on their own.
[[nodiscard]] bool EncodePromptText(const Tokenizer& tokenizer,
                                    const PromptText& prompt,
                                    std::vector<std::uint32_t>* ids,
                                    std::string* error = nullptr);

/// `_tokenize_with_nonverbal_tags`: encodes the text between non-verbal tags
/// and each tag separately.
[[nodiscard]] bool EncodeWithNonverbalTags(const Tokenizer& tokenizer,
                                           std::string_view text,
                                           std::vector<std::uint32_t>* ids,
                                           std::string* error = nullptr);

}  // namespace gufo::models::omnivoice

#endif  // GUFO_MODELS_OMNIVOICE_PROMPT_HPP_
