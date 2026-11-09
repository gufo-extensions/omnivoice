#ifndef GUFO_MODELS_OMNIVOICE_TEXT_HPP_
#define GUFO_MODELS_OMNIVOICE_TEXT_HPP_

#include <cstddef>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

// Text preprocessing of OmniVoice's reference implementation
// (k2-fsa/OmniVoice, Apache-2.0): language and voice-design resolution, the
// rule-based duration estimator, prompt text cleanup and punctuation
// chunking. Every function reproduces the Python helper named in its
// comment.
namespace gufo::models::omnivoice {

/// `_resolve_language`: a language ID (`"en"`) is kept, a language name
/// (`"English"`, any case) maps to its ID; "none", unknown names and empty
/// input select the language-agnostic prompt (nullopt).
[[nodiscard]] std::optional<std::string> ResolveLanguage(
    std::string_view language);

/// `_resolve_instruct`: validates comma-separated voice-design items
/// (gender, age, pitch, whisper, an English accent or a Chinese dialect),
/// unifies them to Chinese when `use_chinese` or a dialect asks for it and
/// to English otherwise. nullopt with `*error` empty means no instruction;
/// with `*error` set, the instruction is invalid.
[[nodiscard]] std::optional<std::string> ResolveInstruct(
    std::string_view instruct, bool use_chinese, std::string* error);

/// `RuleDurationEstimator.calculate_total_weight`: phonetic weight of the
/// text's characters.
[[nodiscard]] double TextWeight(std::string_view text);

/// `OmniVoice._estimate_target_tokens`: audio frames for `text`, scaled from
/// the reference's frames per weight (or "Nice to meet you." at 25 frames
/// without a reference) and divided by `speed`.
[[nodiscard]] std::size_t EstimateTargetFrames(
    std::string_view text, const std::optional<std::string>& reference_text,
    std::size_t reference_frames, double speed = 1.0);

/// `_combine_text`: the reference transcript and the text joined by a
/// space, without line breaks, with ASCII parentheses, single spaces and no
/// spaces next to Chinese characters.
[[nodiscard]] std::string CombineText(
    std::string_view text, const std::optional<std::string>& reference_text);

/// `add_punctuation`: strips the text and ends it with "." (or "。" for
/// Chinese) unless it already ends with punctuation.
[[nodiscard]] std::string AddPunctuation(std::string_view text);

/// `chunk_text_punctuation`: splits at sentence and clause punctuation
/// (not after common abbreviations), merges pieces up to `chunk_length`
/// code points and folds pieces shorter than `min_chunk_length` into a
/// neighbour.
[[nodiscard]] std::vector<std::string> ChunkTextPunctuation(
    std::string_view text, std::size_t chunk_length,
    std::size_t min_chunk_length = 3);

}  // namespace gufo::models::omnivoice

#endif  // GUFO_MODELS_OMNIVOICE_TEXT_HPP_
