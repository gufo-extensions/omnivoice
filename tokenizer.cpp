// NFC normalization, the byte-level alphabet and the BPE merge
// loop are adapted from Gufo's Qwen3-TTS tokenizer (MIT, see
// licenses/gufo.txt). The pre-tokenizer follows OmniVoice's Qwen2 pattern,
// which differs from Qwen3-TTS's: letters exclude combining marks and digits
// split one at a time.
#include "tokenizer.hpp"

#include <unicode/normalizer2.h>
#include <unicode/stringpiece.h>
#include <unicode/uchar.h>
#include <unicode/unistr.h>

#include <algorithm>
#include <fstream>
#include <gufo/ext/json.hpp>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <utility>

#include "unicode.hpp"

namespace gufo::models::omnivoice {
namespace {

namespace json = gufo::json;

constexpr std::size_t kMaxTokenizerBytes = 64U << 20U;
constexpr char kPairSeparator = '\0';

void SetError(std::string* error, std::string message) {
  if (error != nullptr)
    *error = std::move(message);
}

bool NormalizeNfc(std::string_view input, std::string* output) {
  UErrorCode status = U_ZERO_ERROR;
  const icu::Normalizer2* normalizer = icu::Normalizer2::getNFCInstance(status);
  if (U_FAILURE(status) || normalizer == nullptr)
    return false;
  const icu::UnicodeString source = icu::UnicodeString::fromUTF8(
      icu::StringPiece(input.data(), static_cast<std::int32_t>(input.size())));
  icu::UnicodeString normalized;
  normalizer->normalize(source, normalized, status);
  if (U_FAILURE(status))
    return false;
  output->clear();
  normalized.toUTF8String(*output);
  return true;
}

// Character classes of the pattern (IsSpace is Oniguruma's \s)
//   (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}
//   | ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
bool IsLetter(std::uint32_t value) {
  const auto category =
      static_cast<UCharCategory>(u_charType(static_cast<UChar32>(value)));
  return category == U_UPPERCASE_LETTER || category == U_LOWERCASE_LETTER ||
         category == U_TITLECASE_LETTER || category == U_MODIFIER_LETTER ||
         category == U_OTHER_LETTER;
}

bool IsNumber(std::uint32_t value) {
  const auto category =
      static_cast<UCharCategory>(u_charType(static_cast<UChar32>(value)));
  return category == U_DECIMAL_DIGIT_NUMBER || category == U_LETTER_NUMBER ||
         category == U_OTHER_NUMBER;
}

bool IsNewline(std::uint32_t value) {
  return value == '\r' || value == '\n';
}

std::size_t ContractionLength(const std::vector<Codepoint>& points,
                              std::size_t index) {
  static constexpr std::array<std::string_view, 7> kSuffixes = {
      "s", "t", "re", "ve", "m", "ll", "d"};
  if (points[index].value != '\'')
    return 0;
  for (const std::string_view suffix : kSuffixes) {
    if (suffix.size() >= points.size() - index)
      continue;
    bool matches = true;
    for (std::size_t offset = 0; offset < suffix.size(); ++offset) {
      std::uint32_t value = points[index + 1 + offset].value;
      if (value >= 'A' && value <= 'Z')
        value += 'a' - 'A';
      if (value != static_cast<unsigned char>(suffix[offset])) {
        matches = false;
        break;
      }
    }
    if (matches)
      return suffix.size() + 1;
  }
  return 0;
}

/// Length of the pattern's match at `index`, trying its alternatives in
/// order.
std::size_t MatchLength(const std::vector<Codepoint>& points,
                        std::size_t index) {
  const std::size_t count = points.size();
  const auto value = [&](std::size_t at) { return points[at].value; };
  if (const std::size_t length = ContractionLength(points, index))
    return length;
  // [^\r\n\p{L}\p{N}]?\p{L}+
  std::size_t letters = index;
  if (!IsLetter(value(index)) && !IsNewline(value(index)) &&
      !IsNumber(value(index)) && index + 1 < count &&
      IsLetter(value(index + 1)))
    letters = index + 1;
  if (IsLetter(value(letters))) {
    std::size_t end = letters + 1;
    while (end < count && IsLetter(value(end)))
      ++end;
    return end - index;
  }
  // \p{N}
  if (IsNumber(value(index)))
    return 1;
  //  ?[^\s\p{L}\p{N}]+[\r\n]*
  const auto is_symbol = [&](std::size_t at) {
    return !IsSpace(value(at)) && !IsLetter(value(at)) && !IsNumber(value(at));
  };
  std::size_t symbols = index;
  if (value(index) == ' ' && index + 1 < count && is_symbol(index + 1))
    symbols = index + 1;
  if (is_symbol(symbols)) {
    std::size_t end = symbols + 1;
    while (end < count && is_symbol(end))
      ++end;
    while (end < count && IsNewline(value(end)))
      ++end;
    return end - index;
  }
  // Every other character is whitespace.
  std::size_t run_end = index + 1;
  while (run_end < count && IsSpace(value(run_end)))
    ++run_end;
  // \s*[\r\n]+ ends after the run's last newline.
  for (std::size_t at = run_end; at > index; --at) {
    if (IsNewline(value(at - 1)))
      return at - index;
  }
  // \s+(?!\S) leaves the run's last whitespace before a non-space; \s+ takes
  // a single whitespace character.
  if (run_end == count)
    return run_end - index;
  return run_end - index > 1 ? run_end - index - 1 : 1;
}

std::string PairKey(std::string_view left, std::string_view right) {
  std::string key;
  key.reserve(left.size() + right.size() + 1);
  key.append(left);
  key.push_back(kPairSeparator);
  key.append(right);
  return key;
}

std::string ReadFile(const std::filesystem::path& path) {
  std::error_code code;
  const std::uintmax_t bytes = std::filesystem::file_size(path, code);
  if (code)
    throw std::runtime_error("cannot read " + path.string() + ": " +
                             code.message());
  if (bytes > kMaxTokenizerBytes)
    throw std::runtime_error(path.string() + " is implausibly large");
  std::ifstream stream(path, std::ios::binary);
  std::string text((std::istreambuf_iterator<char>(stream)),
                   std::istreambuf_iterator<char>());
  if (text.size() != bytes)
    throw std::runtime_error("cannot read " + path.string());
  return text;
}

std::uint32_t TokenId(const json::Value& value) {
  const std::size_t id = value.as_size(std::numeric_limits<std::size_t>::max());
  if (id > std::numeric_limits<std::uint32_t>::max())
    throw std::runtime_error("token ID is not a 32-bit integer");
  return static_cast<std::uint32_t>(id);
}

}  // namespace

std::unique_ptr<Tokenizer> Tokenizer::Load(
    const std::filesystem::path& tokenizer_json, std::string* error) {
  std::unique_ptr<Tokenizer> tokenizer(new Tokenizer());
  try {
    const json::Value root = json::parse(ReadFile(tokenizer_json));
    if (!root.is_object())
      throw std::runtime_error("root is not an object");
    const json::Value* normalizer = root.find("normalizer");
    if (normalizer == nullptr || normalizer->member_str("type") != "NFC")
      throw std::runtime_error("expected an NFC normalizer");
    const json::Value* model = root.find("model");
    if (model == nullptr || model->member_str("type") != "BPE")
      throw std::runtime_error("expected a BPE model");
    const json::Value* vocabulary = model->find("vocab");
    const json::Value* merges = model->find("merges");
    if (vocabulary == nullptr || !vocabulary->is_object() ||
        merges == nullptr || !merges->is_array())
      throw std::runtime_error("BPE model lacks vocab or merges");
    std::size_t limit = 0;
    for (const auto& [symbol, id_value] : vocabulary->members()) {
      const std::uint32_t id = TokenId(id_value);
      if (!tokenizer->vocabulary_.emplace(symbol, id).second)
        throw std::runtime_error("duplicate vocabulary entry");
      limit = std::max<std::size_t>(limit, std::size_t{id} + 1);
    }
    std::uint32_t rank = 0;
    for (const json::Value& merge : merges->items()) {
      std::string key;
      if (merge.is_array() && merge.size() == 2 &&
          merge.items()[0].is_string() && merge.items()[1].is_string()) {
        key = PairKey(merge.items()[0].str(), merge.items()[1].str());
      } else if (merge.is_string()) {
        const std::string& text = merge.str();
        const std::size_t space = text.find(' ');
        if (space == std::string::npos || space == 0 ||
            space + 1 == text.size())
          throw std::runtime_error("malformed merge");
        key = PairKey(std::string_view(text).substr(0, space),
                      std::string_view(text).substr(space + 1));
      } else {
        throw std::runtime_error("malformed merge");
      }
      if (!tokenizer->merge_ranks_.emplace(std::move(key), rank++).second)
        throw std::runtime_error("duplicate merge");
    }
    const json::Value* added = root.find("added_tokens");
    if (added == nullptr || !added->is_array())
      throw std::runtime_error("missing added_tokens");
    for (const json::Value& token : added->items()) {
      const std::string content = token.member_str("content");
      const json::Value* id = token.find("id");
      if (content.empty() || id == nullptr)
        throw std::runtime_error("malformed added token");
      // Only verbatim matches are implemented.
      for (const char* flag :
           {"normalized", "lstrip", "rstrip", "single_word"}) {
        const json::Value* option = token.find(flag);
        if (option != nullptr && option->as_bool())
          throw std::runtime_error("added token " + content + " sets " + flag);
      }
      tokenizer->added_tokens_.push_back({content, TokenId(*id)});
      limit = std::max<std::size_t>(limit, std::size_t{TokenId(*id)} + 1);
    }
    std::ranges::sort(tokenizer->added_tokens_,
                      [](const AddedToken& left, const AddedToken& right) {
                        if (left.content.size() != right.content.size())
                          return left.content.size() > right.content.size();
                        return left.content < right.content;
                      });
    tokenizer->vocabulary_size_ = limit;
    // GPT-2's byte-to-unicode alphabet.
    std::uint32_t shifted = 0;
    for (std::uint32_t byte = 0; byte < 256; ++byte) {
      const bool visible = (byte >= '!' && byte <= '~') ||
                           (byte >= 0xA1U && byte <= 0xACU) ||
                           (byte >= 0xAEU && byte <= 0xFFU);
      AppendUtf8(visible ? byte : 256U + shifted++,
                 &tokenizer->byte_symbols_[byte]);
    }
  } catch (const std::exception& failure) {
    SetError(error, "OmniVoice tokenizer " + tokenizer_json.string() + ": " +
                        failure.what());
    return nullptr;
  }
  return tokenizer;
}

std::optional<std::uint32_t> Tokenizer::AddedTokenId(
    std::string_view content) const {
  for (const AddedToken& token : added_tokens_) {
    if (token.content == content)
      return token.id;
  }
  return std::nullopt;
}

bool Tokenizer::EncodePiece(std::string_view piece,
                            std::vector<std::uint32_t>* ids,
                            std::string* error) const {
  std::vector<std::string> symbols;
  symbols.reserve(piece.size());
  for (const unsigned char byte : piece)
    symbols.push_back(byte_symbols_[byte]);
  while (symbols.size() > 1) {
    std::optional<std::uint32_t> best_rank;
    std::size_t best = 0;
    for (std::size_t index = 0; index + 1 < symbols.size(); ++index) {
      const auto found =
          merge_ranks_.find(PairKey(symbols[index], symbols[index + 1]));
      if (found != merge_ranks_.end() &&
          (!best_rank.has_value() || found->second < *best_rank)) {
        best_rank = found->second;
        best = index;
      }
    }
    if (!best_rank.has_value())
      break;
    const std::string left = symbols[best];
    const std::string right = symbols[best + 1];
    std::vector<std::string> merged;
    merged.reserve(symbols.size() - 1);
    for (std::size_t index = 0; index < symbols.size();) {
      if (index + 1 < symbols.size() && symbols[index] == left &&
          symbols[index + 1] == right) {
        merged.push_back(left + right);
        index += 2;
      } else {
        merged.push_back(std::move(symbols[index++]));
      }
    }
    symbols = std::move(merged);
  }
  for (const std::string& symbol : symbols) {
    const auto found = vocabulary_.find(symbol);
    if (found == vocabulary_.end()) {
      SetError(error, "OmniVoice tokenizer: BPE symbol outside the vocabulary");
      return false;
    }
    ids->push_back(found->second);
  }
  return true;
}

bool Tokenizer::EncodeOrdinary(std::string_view text,
                               std::vector<std::uint32_t>* ids,
                               std::string* error) const {
  std::string normalized;
  if (!NormalizeNfc(text, &normalized)) {
    SetError(error, "OmniVoice tokenizer: NFC normalization failed");
    return false;
  }
  std::vector<Codepoint> points;
  if (!DecodeUtf8(normalized, &points)) {
    SetError(error, "OmniVoice tokenizer: text is not valid UTF-8");
    return false;
  }
  for (std::size_t index = 0; index < points.size();) {
    const std::size_t length = MatchLength(points, index);
    const std::size_t begin = points[index].begin;
    const std::size_t end = points[index + length - 1].end;
    if (!EncodePiece(std::string_view(normalized).substr(begin, end - begin),
                     ids, error))
      return false;
    index += length;
  }
  return true;
}

bool Tokenizer::Encode(std::string_view text, std::vector<std::uint32_t>* ids,
                       std::string* error) const {
  std::vector<Codepoint> validated;
  if (!DecodeUtf8(text, &validated)) {
    SetError(error, "OmniVoice tokenizer: text is not valid UTF-8");
    return false;
  }
  ids->clear();
  std::size_t start = 0;
  while (start < text.size()) {
    const AddedToken* selected = nullptr;
    std::size_t selected_at = std::string_view::npos;
    for (const AddedToken& candidate : added_tokens_) {
      const std::size_t at = text.find(candidate.content, start);
      if (at != std::string_view::npos && at < selected_at) {
        selected = &candidate;
        selected_at = at;
      }
    }
    const std::size_t ordinary_end =
        selected == nullptr ? text.size() : selected_at;
    if (ordinary_end > start &&
        !EncodeOrdinary(text.substr(start, ordinary_end - start), ids, error)) {
      ids->clear();
      return false;
    }
    if (selected == nullptr)
      break;
    ids->push_back(selected->id);
    start = selected_at + selected->content.size();
  }
  return true;
}

}  // namespace gufo::models::omnivoice
