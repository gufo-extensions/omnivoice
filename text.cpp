// Ported from k2-fsa/OmniVoice (Apache-2.0, see licenses/OmniVoice.txt):
// omnivoice/models/omnivoice.py (_resolve_language, _resolve_instruct,
// _estimate_target_tokens, _combine_text), omnivoice/utils/voice_design.py,
// omnivoice/utils/duration.py and omnivoice/utils/text.py.
#include "text.hpp"

#include <unicode/uchar.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iterator>
#include <string_view>
#include <utility>

#include "unicode.hpp"

namespace gufo::models::omnivoice {
namespace {

struct LanguageEntry {
  std::string_view name;
  std::string_view id;
};

constexpr LanguageEntry kLanguages[] = {
#include "language_table.inc"
};

// ---- Voice design -------------------------------------------------------

struct InstructPair {
  std::string_view english;
  std::string_view chinese;
};

// Each group is one mutually exclusive category.
constexpr std::array<InstructPair, 2> kGender = {
    {{"male", "男"}, {"female", "女"}}};
constexpr std::array<InstructPair, 5> kAge = {{{"child", "儿童"},
                                               {"teenager", "少年"},
                                               {"young adult", "青年"},
                                               {"middle-aged", "中年"},
                                               {"elderly", "老年"}}};
constexpr std::array<InstructPair, 5> kPitch = {
    {{"very low pitch", "极低音调"},
     {"low pitch", "低音调"},
     {"moderate pitch", "中音调"},
     {"high pitch", "高音调"},
     {"very high pitch", "极高音调"}}};
constexpr std::array<InstructPair, 1> kStyle = {{{"whisper", "耳语"}}};
constexpr std::array<std::string_view, 10> kAccents = {
    "american accent", "british accent",    "australian accent",
    "chinese accent",  "canadian accent",   "indian accent",
    "korean accent",   "portuguese accent", "russian accent",
    "japanese accent"};
constexpr std::array<std::string_view, 12> kDialects = {
    "河南话", "陕西话",   "四川话", "贵州话", "云南话", "桂林话",
    "济南话", "石家庄话", "甘肃话", "宁夏话", "青岛话", "东北话"};

template<std::size_t N>
const InstructPair* FindPair(const std::array<InstructPair, N>& pairs,
                             std::string_view item) {
  for (const InstructPair& pair : pairs) {
    if (pair.english == item || pair.chinese == item)
      return &pair;
  }
  return nullptr;
}

const InstructPair* FindAnyPair(std::string_view item) {
  if (const InstructPair* pair = FindPair(kGender, item))
    return pair;
  if (const InstructPair* pair = FindPair(kAge, item))
    return pair;
  if (const InstructPair* pair = FindPair(kPitch, item))
    return pair;
  return FindPair(kStyle, item);
}

template<std::size_t N>
bool Contains(const std::array<std::string_view, N>& items,
              std::string_view item) {
  return std::find(items.begin(), items.end(), item) != items.end();
}

/// Index of the item's mutually exclusive category.
int CategoryOf(std::string_view item) {
  if (FindPair(kGender, item) != nullptr)
    return 0;
  if (FindPair(kAge, item) != nullptr)
    return 1;
  if (FindPair(kPitch, item) != nullptr)
    return 2;
  if (FindPair(kStyle, item) != nullptr)
    return 3;
  if (Contains(kAccents, item))
    return 4;
  if (Contains(kDialects, item))
    return 5;
  return -1;
}

std::string ValidItems() {
  std::string text;
  const auto add = [&text](std::string_view item) {
    if (!text.empty())
      text += ", ";
    text += item;
  };
  for (const InstructPair& pair : kGender)
    add(pair.english);
  for (const InstructPair& pair : kAge)
    add(pair.english);
  for (const InstructPair& pair : kPitch)
    add(pair.english);
  for (const InstructPair& pair : kStyle)
    add(pair.english);
  for (const std::string_view accent : kAccents)
    add(accent);
  return text;
}

/// Splits on half- and full-width commas with surrounding whitespace
/// (`re.split(r"\s*[,，]\s*", ...)`), dropping empty items.
std::vector<std::string> SplitInstruct(std::string_view text) {
  std::vector<Codepoint> points;
  std::vector<std::string> items;
  if (!DecodeUtf8(text, &points))
    return items;
  std::size_t start = 0;
  const auto flush = [&](std::size_t end) {
    // Trailing whitespace before the comma belongs to the separator.
    std::size_t last = end;
    while (last > start && IsSpace(points[last - 1].value))
      --last;
    if (last > start)
      items.emplace_back(text.substr(
          points[start].begin, points[last - 1].end - points[start].begin));
  };
  for (std::size_t index = 0; index < points.size(); ++index) {
    if (points[index].value != ',' && points[index].value != 0xFF0CU)
      continue;
    flush(index);
    start = index + 1;
    while (start < points.size() && IsSpace(points[start].value))
      ++start;
    index = start - 1;
  }
  flush(points.size());
  return items;
}

// ---- Duration -----------------------------------------------------------

enum class Script {
  kCjk,
  kHangul,
  kKana,
  kEthiopic,
  kYi,
  kIndic,
  kThaiLao,
  kKhmerMyanmar,
  kArabic,
  kHebrew,
  kLatin,
  kCyrillic,
  kGreek,
  kArmenian,
  kGeorgian,
  kDefault,
};

double ScriptWeight(Script script) {
  switch (script) {
    case Script::kCjk:
    case Script::kEthiopic:
    case Script::kYi:
      return 3.0;
    case Script::kHangul:
      return 2.5;
    case Script::kKana:
      return 2.2;
    case Script::kIndic:
    case Script::kKhmerMyanmar:
      return 1.8;
    case Script::kThaiLao:
    case Script::kArabic:
    case Script::kHebrew:
      return 1.5;
    case Script::kLatin:
    case Script::kCyrillic:
    case Script::kGreek:
    case Script::kArmenian:
    case Script::kGeorgian:
    case Script::kDefault:
      return 1.0;
  }
  return 1.0;
}

constexpr double kPunctuationWeight = 0.5;
constexpr double kSpaceWeight = 0.2;
constexpr double kDigitWeight = 3.5;
constexpr double kMarkWeight = 0.0;

struct ScriptRange {
  std::uint32_t last;
  Script script;
};

// Upper bounds of the script blocks, as RuleDurationEstimator.ranges.
constexpr ScriptRange kScriptRanges[] = {
    {0x02AF, Script::kLatin},        {0x03FF, Script::kGreek},
    {0x052F, Script::kCyrillic},     {0x058F, Script::kArmenian},
    {0x05FF, Script::kHebrew},       {0x077F, Script::kArabic},
    {0x089F, Script::kArabic},       {0x08FF, Script::kArabic},
    {0x097F, Script::kIndic},        {0x09FF, Script::kIndic},
    {0x0A7F, Script::kIndic},        {0x0AFF, Script::kIndic},
    {0x0B7F, Script::kIndic},        {0x0BFF, Script::kIndic},
    {0x0C7F, Script::kIndic},        {0x0CFF, Script::kIndic},
    {0x0D7F, Script::kIndic},        {0x0DFF, Script::kIndic},
    {0x0EFF, Script::kThaiLao},      {0x0FFF, Script::kIndic},
    {0x109F, Script::kKhmerMyanmar}, {0x10FF, Script::kGeorgian},
    {0x11FF, Script::kHangul},       {0x137F, Script::kEthiopic},
    {0x139F, Script::kEthiopic},     {0x13FF, Script::kDefault},
    {0x167F, Script::kDefault},      {0x169F, Script::kDefault},
    {0x16FF, Script::kDefault},      {0x171F, Script::kDefault},
    {0x173F, Script::kDefault},      {0x175F, Script::kDefault},
    {0x177F, Script::kDefault},      {0x17FF, Script::kKhmerMyanmar},
    {0x18AF, Script::kDefault},      {0x18FF, Script::kDefault},
    {0x194F, Script::kIndic},        {0x19DF, Script::kIndic},
    {0x19FF, Script::kKhmerMyanmar}, {0x1A1F, Script::kIndic},
    {0x1AAF, Script::kIndic},        {0x1B7F, Script::kIndic},
    {0x1BBF, Script::kIndic},        {0x1BFF, Script::kIndic},
    {0x1C4F, Script::kIndic},        {0x1C7F, Script::kIndic},
    {0x1C8F, Script::kCyrillic},     {0x1CBF, Script::kGeorgian},
    {0x1CCF, Script::kIndic},        {0x1CFF, Script::kIndic},
    {0x1D7F, Script::kLatin},        {0x1DBF, Script::kLatin},
    {0x1DFF, Script::kDefault},      {0x1EFF, Script::kLatin},
    {0x309F, Script::kKana},         {0x30FF, Script::kKana},
    {0x312F, Script::kCjk},          {0x318F, Script::kHangul},
    {0x9FFF, Script::kCjk},          {0xA4CF, Script::kYi},
    {0xA4FF, Script::kDefault},      {0xA63F, Script::kDefault},
    {0xA69F, Script::kCyrillic},     {0xA6FF, Script::kDefault},
    {0xA7FF, Script::kLatin},        {0xA82F, Script::kIndic},
    {0xA87F, Script::kDefault},      {0xA8DF, Script::kIndic},
    {0xA8FF, Script::kIndic},        {0xA92F, Script::kIndic},
    {0xA95F, Script::kIndic},        {0xA97F, Script::kHangul},
    {0xA9DF, Script::kIndic},        {0xA9FF, Script::kKhmerMyanmar},
    {0xAA5F, Script::kIndic},        {0xAA7F, Script::kKhmerMyanmar},
    {0xAADF, Script::kIndic},        {0xAAFF, Script::kIndic},
    {0xAB2F, Script::kEthiopic},     {0xAB6F, Script::kLatin},
    {0xABBF, Script::kDefault},      {0xABFF, Script::kIndic},
    {0xD7AF, Script::kHangul},       {0xFAFF, Script::kCjk},
    {0xFDFF, Script::kArabic},       {0xFE6F, Script::kDefault},
    {0xFEFF, Script::kArabic},       {0xFFEF, Script::kLatin},
};

double CharacterWeight(std::uint32_t code) {
  if ((code >= 'A' && code <= 'Z') || (code >= 'a' && code <= 'z'))
    return ScriptWeight(Script::kLatin);
  if (code == ' ')
    return kSpaceWeight;
  if (code == 0x0640U)  // Arabic tatweel
    return kMarkWeight;
  switch (static_cast<UCharCategory>(u_charType(static_cast<UChar32>(code)))) {
    case U_NON_SPACING_MARK:
    case U_ENCLOSING_MARK:
    case U_COMBINING_SPACING_MARK:
      return kMarkWeight;
    case U_DASH_PUNCTUATION:
    case U_START_PUNCTUATION:
    case U_END_PUNCTUATION:
    case U_CONNECTOR_PUNCTUATION:
    case U_OTHER_PUNCTUATION:
    case U_INITIAL_PUNCTUATION:
    case U_FINAL_PUNCTUATION:
    case U_MATH_SYMBOL:
    case U_CURRENCY_SYMBOL:
    case U_MODIFIER_SYMBOL:
    case U_OTHER_SYMBOL:
      return kPunctuationWeight;
    case U_SPACE_SEPARATOR:
    case U_LINE_SEPARATOR:
    case U_PARAGRAPH_SEPARATOR:
      return kSpaceWeight;
    case U_DECIMAL_DIGIT_NUMBER:
    case U_LETTER_NUMBER:
    case U_OTHER_NUMBER:
      return kDigitWeight;
    default:
      break;
  }
  const auto* range =
      std::lower_bound(std::begin(kScriptRanges), std::end(kScriptRanges), code,
                       [](const ScriptRange& entry, std::uint32_t value) {
                         return entry.last < value;
                       });
  if (range != std::end(kScriptRanges))
    return ScriptWeight(range->script);
  return code > 0x20000U ? ScriptWeight(Script::kCjk)
                         : ScriptWeight(Script::kDefault);
}

/// RuleDurationEstimator.estimate_duration with its default low threshold
/// (50 frames) and boost strength (3).
double EstimateDuration(std::string_view target, std::string_view reference,
                        double reference_duration) {
  if (reference_duration <= 0.0 || reference.empty())
    return 0.0;
  const double reference_weight = TextWeight(reference);
  if (reference_weight == 0.0)
    return 0.0;
  const double speed = reference_weight / reference_duration;
  const double estimate = TextWeight(target) / speed;
  constexpr double kLowThreshold = 50.0;
  constexpr double kBoostStrength = 3.0;
  if (estimate < kLowThreshold)
    return kLowThreshold *
           std::pow(estimate / kLowThreshold, 1.0 / kBoostStrength);
  return estimate;
}

// ---- Chunking -----------------------------------------------------------

bool IsSplitPunctuation(std::uint32_t value) {
  static constexpr std::u32string_view kSplit = U".,;:!?。，；：！？";
  return kSplit.find(static_cast<char32_t>(value)) != std::u32string_view::npos;
}

bool IsClosingMark(std::uint32_t value) {
  static constexpr std::u32string_view kClosing = U"\"'“”‘’）]》>」】";
  return kClosing.find(static_cast<char32_t>(value)) !=
         std::u32string_view::npos;
}

bool IsEndPunctuation(std::uint32_t value) {
  static constexpr std::u32string_view kEnd =
      U";:,.!?…)]}\"'“”‘’；：，。！？、）】";
  return kEnd.find(static_cast<char32_t>(value)) != std::u32string_view::npos;
}

constexpr std::string_view kAbbreviations[] = {
    "Mr.",  "Mrs.",  "Ms.",     "Dr.",   "Prof.", "Sr.",  "Jr.",   "Rev.",
    "Fr.",  "Hon.",  "Pres.",   "Gov.",  "Capt.", "Gen.", "Sen.",  "Rep.",
    "Col.", "Maj.",  "Lt.",     "Cmdr.", "Sgt.",  "Cpl.", "Co.",   "Corp.",
    "Inc.", "Ltd.",  "Est.",    "Dept.", "St.",   "Ave.", "Blvd.", "Rd.",
    "Mt.",  "Ft.",   "No.",     "Jan.",  "Feb.",  "Mar.", "Apr.",  "Aug.",
    "Sep.", "Sept.", "Oct.",    "Nov.",  "Dec.",  "i.e.", "e.g.",  "vs.",
    "Vs.",  "Etc.",  "approx.", "fig.",  "def."};

std::string Encode(const std::vector<std::uint32_t>& points) {
  std::string text;
  for (const std::uint32_t point : points)
    AppendUtf8(point, &text);
  return text;
}

/// `"".join(sentence).strip().split()[-1]`.
std::string LastWord(const std::vector<std::uint32_t>& points) {
  std::size_t end = points.size();
  while (end > 0 && IsSpace(points[end - 1]))
    --end;
  std::size_t begin = end;
  while (begin > 0 && !IsSpace(points[begin - 1]))
    --begin;
  return Encode(
      std::vector<std::uint32_t>(points.begin() + begin, points.begin() + end));
}

}  // namespace

std::optional<std::string> ResolveLanguage(std::string_view language) {
  if (language.empty())
    return std::nullopt;
  const std::string lower = ToLower(language);
  if (lower == "none")
    return std::nullopt;
  for (const LanguageEntry& entry : kLanguages) {
    if (entry.id == language)
      return std::string(language);
  }
  const auto* found =
      std::lower_bound(std::begin(kLanguages), std::end(kLanguages), lower,
                       [](const LanguageEntry& entry, const std::string& key) {
                         return entry.name < key;
                       });
  if (found != std::end(kLanguages) && found->name == lower)
    return std::string(found->id);
  return std::nullopt;
}

std::optional<std::string> ResolveInstruct(std::string_view instruct,
                                           bool use_chinese,
                                           std::string* error) {
  std::string ignored;
  if (error == nullptr)
    error = &ignored;
  error->clear();
  const std::string stripped = StripSpace(instruct);
  if (stripped.empty())
    return std::nullopt;
  std::vector<std::string> items;
  std::string unknown;
  for (const std::string& raw : SplitInstruct(stripped)) {
    std::string item = ToLower(StripSpace(raw));
    if (CategoryOf(item) < 0) {
      unknown += (unknown.empty() ? "'" : ", '") + raw + "'";
      continue;
    }
    items.push_back(std::move(item));
  }
  if (!unknown.empty()) {
    *error = "unsupported voice design items " + unknown +
             "; use comma-separated items from: " + ValidItems() +
             ", or a Chinese dialect such as 四川话";
    return std::nullopt;
  }
  const bool has_dialect = std::any_of(
      items.begin(), items.end(),
      [](const std::string& item) { return CategoryOf(item) == 5; });
  const bool has_accent = std::any_of(
      items.begin(), items.end(),
      [](const std::string& item) { return CategoryOf(item) == 4; });
  if (has_dialect && has_accent) {
    *error =
        "a voice design can't mix a Chinese dialect with an English accent";
    return std::nullopt;
  }
  if (has_dialect)
    use_chinese = true;
  else if (has_accent)
    use_chinese = false;
  for (std::string& item : items) {
    if (const InstructPair* pair = FindAnyPair(item))
      item = std::string(use_chinese ? pair->chinese : pair->english);
  }
  std::array<int, 6> counts{};
  for (const std::string& item : items) {
    if (++counts[static_cast<std::size_t>(CategoryOf(item))] > 1) {
      *error =
          "a voice design allows one item per category (gender, age, "
          "pitch, style, accent, dialect)";
      return std::nullopt;
    }
  }
  const bool chinese = std::any_of(
      items.begin(), items.end(),
      [](const std::string& item) { return ContainsCjkIdeograph(item); });
  std::string result;
  for (const std::string& item : items) {
    if (!result.empty())
      result += chinese ? "，" : ", ";
    result += item;
  }
  return result;
}

double TextWeight(std::string_view text) {
  std::vector<Codepoint> points;
  if (!DecodeUtf8(text, &points))
    return 0.0;
  double weight = 0.0;
  for (const Codepoint& point : points)
    weight += CharacterWeight(point.value);
  return weight;
}

std::size_t EstimateTargetFrames(
    std::string_view text, const std::optional<std::string>& reference_text,
    std::size_t reference_frames, double speed) {
  std::string_view reference = "Nice to meet you.";
  double frames = 25.0;
  if (reference_text.has_value() && !reference_text->empty() &&
      reference_frames > 0) {
    reference = *reference_text;
    frames = static_cast<double>(reference_frames);
  }
  double estimate = EstimateDuration(text, reference, frames);
  if (speed > 0.0 && speed != 1.0)
    estimate /= speed;
  return std::max<std::size_t>(1, static_cast<std::size_t>(estimate));
}

std::string CombineText(std::string_view text,
                        const std::optional<std::string>& reference_text) {
  std::string joined = StripSpace(text);
  if (reference_text.has_value() && !reference_text->empty())
    joined = StripSpace(*reference_text) + " " + joined;
  std::vector<Codepoint> decoded;
  if (!DecodeUtf8(joined, &decoded))
    return joined;
  // Drop line breaks, use ASCII parentheses, collapse space and tab runs.
  std::vector<std::uint32_t> points;
  for (const Codepoint& point : decoded) {
    std::uint32_t value = point.value;
    if (value == '\r' || value == '\n')
      continue;
    if (value == 0xFF08U)
      value = '(';
    else if (value == 0xFF09U)
      value = ')';
    const bool blank = value == ' ' || value == '\t';
    if (blank && !points.empty() && points.back() == ' ')
      continue;
    points.push_back(blank ? ' ' : value);
  }
  // Remove whitespace runs that touch a Chinese character.
  std::vector<std::uint32_t> cleaned;
  for (std::size_t index = 0; index < points.size();) {
    if (!IsSpace(points[index])) {
      cleaned.push_back(points[index++]);
      continue;
    }
    std::size_t end = index;
    while (end < points.size() && IsSpace(points[end]))
      ++end;
    const bool after_chinese = index > 0 && IsCjkIdeograph(points[index - 1]);
    const bool before_chinese =
        end < points.size() && IsCjkIdeograph(points[end]);
    if (!after_chinese && !before_chinese)
      cleaned.insert(cleaned.end(), points.begin() + index,
                     points.begin() + end);
    index = end;
  }
  return Encode(cleaned);
}

std::string AddPunctuation(std::string_view text) {
  std::string stripped = StripSpace(text);
  std::vector<Codepoint> points;
  if (stripped.empty() || !DecodeUtf8(stripped, &points))
    return stripped;
  if (!IsEndPunctuation(points.back().value))
    stripped += ContainsCjkIdeograph(stripped) ? "。" : ".";
  return stripped;
}

std::vector<std::string> ChunkTextPunctuation(std::string_view text,
                                              std::size_t chunk_length,
                                              std::size_t min_chunk_length) {
  std::vector<Codepoint> decoded;
  if (!DecodeUtf8(text, &decoded))
    return {std::string(text)};
  using Piece = std::vector<std::uint32_t>;
  // 1. Split after punctuation; punctuation that opens a sentence joins the
  // previous one.
  std::vector<Piece> sentences;
  Piece current;
  for (const Codepoint& point : decoded) {
    const std::uint32_t value = point.value;
    if (current.empty() && !sentences.empty() &&
        (IsSplitPunctuation(value) || IsClosingMark(value))) {
      sentences.back().push_back(value);
      continue;
    }
    current.push_back(value);
    if (!IsSplitPunctuation(value))
      continue;
    if (value == '.') {
      const std::string word = LastWord(current);
      if (std::find(std::begin(kAbbreviations), std::end(kAbbreviations),
                    word) != std::end(kAbbreviations))
        continue;
    }
    sentences.push_back(std::move(current));
    current.clear();
  }
  if (!current.empty())
    sentences.push_back(std::move(current));
  // 2. Merge sentences up to the chunk length.
  std::vector<Piece> merged;
  Piece chunk;
  for (Piece& sentence : sentences) {
    if (chunk.size() + sentence.size() <= chunk_length) {
      chunk.insert(chunk.end(), sentence.begin(), sentence.end());
    } else {
      if (!chunk.empty())
        merged.push_back(std::move(chunk));
      chunk = std::move(sentence);
    }
  }
  if (!chunk.empty())
    merged.push_back(std::move(chunk));
  // 3. Fold undersized chunks into the previous one; a short first chunk
  // takes the second.
  std::vector<Piece> final_chunks;
  const bool first_short =
      !merged.empty() && merged.front().size() < min_chunk_length;
  for (std::size_t index = 0; index < merged.size(); ++index) {
    Piece& piece = merged[index];
    if ((index == 1 && first_short) ||
        (piece.size() < min_chunk_length && !final_chunks.empty())) {
      final_chunks.back().insert(final_chunks.back().end(), piece.begin(),
                                 piece.end());
    } else {
      final_chunks.push_back(std::move(piece));
    }
  }
  std::vector<std::string> result;
  for (const Piece& piece : final_chunks) {
    std::string stripped = StripSpace(Encode(piece));
    if (!stripped.empty())
      result.push_back(std::move(stripped));
  }
  return result;
}

}  // namespace gufo::models::omnivoice
