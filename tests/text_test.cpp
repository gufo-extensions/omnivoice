#include "text.hpp"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <gufo/ext/json.hpp>
#include <iostream>
#include <iterator>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "prompt.hpp"
#include "tokenizer.hpp"

namespace json = gufo::json;
namespace ov = gufo::models::omnivoice;

namespace {

constexpr int kSkip = 77;

int failures = 0;

void Fail(std::string_view what, const std::string& got,
          const std::string& expected) {
  std::cout << "FAIL " << what << "\n  got      " << got << "\n  expected "
            << expected << '\n';
  ++failures;
}

std::optional<std::string> OptionalString(const json::Value& entry,
                                          const std::string& key) {
  const json::Value* value = entry.find(key);
  if (value == nullptr || value->is_null())
    return std::nullopt;
  return value->str();
}

std::string Show(const std::optional<std::string>& value) {
  return value.has_value() ? "\"" + *value + "\"" : "None";
}

std::string Show(const std::vector<std::string>& values) {
  std::string text = "[";
  for (const std::string& value : values)
    text += "\"" + value + "\" ";
  return text + "]";
}

std::string Show(const std::vector<std::uint32_t>& values) {
  std::string text = "[";
  for (const std::uint32_t value : values)
    text += std::to_string(value) + " ";
  return text + "]";
}

void CheckLanguages(const json::Value& golden) {
  for (const json::Value& entry : golden.find("languages")->items()) {
    const std::string& input = entry.find("input")->str();
    const auto expected = OptionalString(entry, "id");
    const auto got = ov::ResolveLanguage(input);
    if (got != expected)
      Fail("language " + input, Show(got), Show(expected));
  }
}

void CheckInstructs(const json::Value& golden) {
  for (const json::Value& entry : golden.find("instructs")->items()) {
    const std::string& input = entry.find("input")->str();
    const bool use_chinese = entry.find("use_zh")->as_bool();
    std::string error;
    const auto got = ov::ResolveInstruct(input, use_chinese, &error);
    if (entry.find("error") != nullptr) {
      if (error.empty())
        Fail("instruct " + input, Show(got), "an error");
      continue;
    }
    const auto expected = OptionalString(entry, "output");
    if (!error.empty() || got != expected)
      Fail("instruct " + input, error.empty() ? Show(got) : error,
           Show(expected));
  }
}

void CheckWeightsAndDurations(const json::Value& golden) {
  for (const json::Value& entry : golden.find("weights")->items()) {
    const std::string& text = entry.find("text")->str();
    const double expected = entry.find("weight")->as_double();
    const double got = ov::TextWeight(text);
    if (std::abs(got - expected) > 1e-9)
      Fail("weight " + text, std::to_string(got), std::to_string(expected));
  }
  for (const json::Value& entry : golden.find("durations")->items()) {
    const std::string& text = entry.find("text")->str();
    const auto reference = OptionalString(entry, "ref_text");
    const json::Value* frames_value = entry.find("ref_frames");
    const std::size_t reference_frames =
        frames_value->is_null() ? 0 : frames_value->as_size();
    const double speed = entry.find("speed")->as_double();
    const std::size_t expected = entry.find("frames")->as_size();
    const std::size_t got =
        ov::EstimateTargetFrames(text, reference, reference_frames, speed);
    if (got != expected)
      Fail("duration " + text + " / " + Show(reference) + " speed " +
               std::to_string(speed),
           std::to_string(got), std::to_string(expected));
  }
}

void CheckTextHelpers(const json::Value& golden) {
  for (const json::Value& entry : golden.find("combined")->items()) {
    const std::string& text = entry.find("text")->str();
    const auto reference = OptionalString(entry, "ref_text");
    const std::string& expected = entry.find("combined")->str();
    const std::string got = ov::CombineText(text, reference);
    if (got != expected)
      Fail("combine " + text, got, expected);
  }
  for (const json::Value& entry : golden.find("punctuation")->items()) {
    const std::string& input = entry.find("input")->str();
    const std::string& expected = entry.find("output")->str();
    const std::string got = ov::AddPunctuation(input);
    if (got != expected)
      Fail("punctuation " + input, got, expected);
  }
  for (const json::Value& entry : golden.find("chunks")->items()) {
    const std::string& text = entry.find("text")->str();
    std::vector<std::string> expected;
    for (const json::Value& chunk : entry.find("chunks")->items())
      expected.push_back(chunk.str());
    const auto got =
        ov::ChunkTextPunctuation(text, entry.find("chunk_len")->as_size(), 3);
    if (got != expected)
      Fail("chunks " + text, Show(got), Show(expected));
  }
}

bool CheckPrompts(const json::Value& golden) {
  const char* root = std::getenv("GUFO_OMNIVOICE_MODEL");
  if (root == nullptr || *root == '\0')
    return false;
  std::string error;
  const auto tokenizer = ov::Tokenizer::Load(
      std::filesystem::path(root) / "tokenizer.json", &error);
  if (tokenizer == nullptr) {
    Fail("tokenizer", error, "loaded");
    return true;
  }
  for (const json::Value& entry : golden.find("prompts")->items()) {
    ov::PromptText prompt;
    prompt.text = entry.find("text")->str();
    prompt.reference_text = OptionalString(entry, "ref_text");
    prompt.language = OptionalString(entry, "lang");
    prompt.instruct = OptionalString(entry, "instruct");
    prompt.denoise = entry.find("denoise")->as_bool();
    prompt.has_reference = entry.find("ref_frames")->as_size() > 0;
    std::vector<std::uint32_t> expected;
    for (const json::Value& id : entry.find("text_ids")->items())
      expected.push_back(static_cast<std::uint32_t>(id.as_size()));
    std::vector<std::uint32_t> got;
    if (!ov::EncodePromptText(*tokenizer, prompt, &got, &error))
      Fail("prompt " + prompt.text, error, Show(expected));
    else if (got != expected)
      Fail("prompt " + prompt.text, Show(got), Show(expected));
  }
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  const std::string_view mode = argc > 1 ? argv[1] : "text";
  std::ifstream stream(GUFO_OMNIVOICE_PROMPT_GOLDEN);
  const json::Value golden =
      json::parse(std::string((std::istreambuf_iterator<char>(stream)),
                              std::istreambuf_iterator<char>()));
  if (mode == "text") {
    CheckLanguages(golden);
    CheckInstructs(golden);
    CheckWeightsAndDurations(golden);
    CheckTextHelpers(golden);
  } else if (mode == "prompt") {
    if (!CheckPrompts(golden)) {
      std::cout << "skip: GUFO_OMNIVOICE_MODEL is not set\n";
      return kSkip;
    }
  } else {
    std::cerr << "usage: omnivoice_text_test [text|prompt]\n";
    return 2;
  }
  std::cout << (failures == 0 ? "ok   " : "FAIL ") << mode << ", " << failures
            << " failures\n";
  return failures == 0 ? 0 : 1;
}
