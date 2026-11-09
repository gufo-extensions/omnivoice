// Development driver: synthesizes one text to a 16-bit mono WAV file.
//
//   omnivoice_synthesize MODEL_DIR OUTPUT.wav TEXT [--language L]
//       [--instruct ITEMS] [--reference WAV --reference-text TEXT]
//       [--seed N] [--greedy] [--repeat N]
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <string_view>
#include <vector>

#include "synthesizer.hpp"

namespace ov = gufo::models::omnivoice;

namespace {

void WriteLittleEndian(std::ofstream& stream, std::uint32_t value, int bytes) {
  for (int index = 0; index < bytes; ++index)
    stream.put(static_cast<char>((value >> (8 * index)) & 0xFFU));
}

bool WriteWav(const std::string& path, const std::vector<float>& samples,
              std::uint32_t rate) {
  std::ofstream stream(path, std::ios::binary);
  const auto data_bytes = static_cast<std::uint32_t>(samples.size() * 2);
  stream.write("RIFF", 4);
  WriteLittleEndian(stream, 36 + data_bytes, 4);
  stream.write("WAVEfmt ", 8);
  WriteLittleEndian(stream, 16, 4);
  WriteLittleEndian(stream, 1, 2);  // PCM
  WriteLittleEndian(stream, 1, 2);  // mono
  WriteLittleEndian(stream, rate, 4);
  WriteLittleEndian(stream, rate * 2, 4);
  WriteLittleEndian(stream, 2, 2);
  WriteLittleEndian(stream, 16, 2);
  stream.write("data", 4);
  WriteLittleEndian(stream, data_bytes, 4);
  for (const float sample : samples) {
    const auto value =
        static_cast<std::int16_t>(std::clamp(sample, -1.0F, 1.0F) * 32767.0F);
    WriteLittleEndian(stream, static_cast<std::uint16_t>(value), 2);
  }
  return stream.good();
}

std::uint32_t ReadLittleEndian(const std::string& bytes, std::size_t at,
                               int count) {
  std::uint32_t value = 0;
  for (int index = count - 1; index >= 0; --index)
    value = (value << 8U) | static_cast<unsigned char>(bytes[at + index]);
  return value;
}

/// PCM16, PCM24, PCM32 or float32 WAV to interleaved floats.
bool ReadWav(const std::string& path, std::vector<float>* samples,
             std::uint32_t* channels, std::uint32_t* rate) {
  std::ifstream stream(path, std::ios::binary);
  const std::string bytes((std::istreambuf_iterator<char>(stream)),
                          std::istreambuf_iterator<char>());
  if (bytes.size() < 12 || bytes.compare(0, 4, "RIFF") != 0 ||
      bytes.compare(8, 4, "WAVE") != 0)
    return false;
  std::uint32_t format = 0, bits = 0;
  for (std::size_t at = 12; at + 8 <= bytes.size();) {
    const std::string id = bytes.substr(at, 4);
    const std::uint32_t size = ReadLittleEndian(bytes, at + 4, 4);
    const std::size_t body = at + 8;
    if (body + size > bytes.size())
      return false;
    if (id == "fmt ") {
      format = ReadLittleEndian(bytes, body, 2);
      *channels = ReadLittleEndian(bytes, body + 2, 2);
      *rate = ReadLittleEndian(bytes, body + 4, 4);
      bits = ReadLittleEndian(bytes, body + 14, 2);
      if (format == 0xFFFEU && size >= 26)
        format = ReadLittleEndian(bytes, body + 24, 2);
    } else if (id == "data") {
      const std::size_t width = bits / 8;
      if (width == 0 || *channels == 0)
        return false;
      for (std::size_t i = body; i + width <= body + size; i += width) {
        const std::uint32_t raw =
            ReadLittleEndian(bytes, i, static_cast<int>(width));
        float value = 0.0F;
        if (format == 3 && bits == 32) {
          std::memcpy(&value, &raw, sizeof(value));
        } else if (format == 1) {
          const auto shift = static_cast<int>(32 - bits);
          const auto sign_extended =
              static_cast<std::int32_t>(raw << shift) >> shift;
          value = static_cast<float>(sign_extended) /
                  static_cast<float>(1U << (bits - 1));
        } else {
          return false;
        }
        samples->push_back(value);
      }
      return true;
    }
    at = body + size + (size & 1U);
  }
  return false;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 4) {
    std::cerr << "usage: omnivoice_synthesize MODEL_DIR OUTPUT.wav TEXT "
                 "[--language L] [--instruct ITEMS] [--reference WAV "
                 "--reference-text TEXT] [--seed N] [--greedy] [--repeat N]\n";
    return 2;
  }
  ov::SynthesisOptions options;
  options.text = argv[3];
  std::string reference_path, reference_text;
  int repeats = 1;
  for (int index = 4; index < argc; ++index) {
    const std::string_view flag = argv[index];
    if (flag == "--greedy") {
      options.decoding.position_temperature = 0.0F;
    } else if (index + 1 < argc && flag == "--language") {
      options.language = argv[++index];
    } else if (index + 1 < argc && flag == "--instruct") {
      options.instruct = argv[++index];
    } else if (index + 1 < argc && flag == "--reference") {
      reference_path = argv[++index];
    } else if (index + 1 < argc && flag == "--reference-text") {
      reference_text = argv[++index];
    } else if (index + 1 < argc && flag == "--repeat") {
      repeats = std::max(1, std::atoi(argv[++index]));
    } else if (index + 1 < argc && flag == "--seed") {
      options.seed = std::strtoull(argv[++index], nullptr, 10);
    } else {
      std::cerr << "unknown option " << flag << '\n';
      return 2;
    }
  }
  std::string error;
  const auto load_start = std::chrono::steady_clock::now();
  auto synthesizer = ov::Synthesizer::Load(argv[1], &error);
  if (synthesizer == nullptr) {
    std::cerr << error << '\n';
    return 1;
  }
  std::cout << "load "
            << std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                             load_start)
                   .count()
            << " s\n";
  ov::ReferenceVoice voice;
  if (!reference_path.empty()) {
    std::vector<float> recording;
    std::uint32_t channels = 0, rate = 0;
    if (!ReadWav(reference_path, &recording, &channels, &rate)) {
      std::cerr << "cannot read " << reference_path << '\n';
      return 1;
    }
    const auto clone_start = std::chrono::steady_clock::now();
    if (!synthesizer->CloneVoice(recording, channels, rate, reference_text,
                                 &voice, &error)) {
      std::cerr << error << '\n';
      return 1;
    }
    std::cout << "reference: " << voice.frames << " frames, encoded in "
              << std::chrono::duration<double>(
                     std::chrono::steady_clock::now() - clone_start)
                     .count()
              << " s\n";
    options.reference = &voice;
  }
  std::vector<float> audio;
  for (int run = 0; run < repeats; ++run) {
    const auto start = std::chrono::steady_clock::now();
    if (!synthesizer->Synthesize(options, {}, &audio, nullptr, &error)) {
      std::cerr << error << '\n';
      return 1;
    }
    const double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
            .count();
    const double duration =
        static_cast<double>(audio.size()) / synthesizer->SampleRate();
    std::cout << "run " << run + 1 << ": synthesis " << seconds << " s for "
              << duration << " s of audio (RTF " << seconds / duration << ")\n";
  }
  if (!WriteWav(argv[2], audio, synthesizer->SampleRate())) {
    std::cerr << "cannot write " << argv[2] << '\n';
    return 1;
  }
  return 0;
}
