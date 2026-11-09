// Ported from k2-fsa/OmniVoice's omnivoice/utils/audio.py and
// create_voice_clone_prompt (Apache-2.0, see licenses/OmniVoice.txt). The
// silence detection reproduces pydub 0.25.1 (MIT) and the resampler
// torchaudio.functional.resample (BSD-2-Clause); see THIRD_PARTY_NOTICES.md.
#include "audio.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

namespace gufo::models::omnivoice {
namespace {

constexpr double kPi = 3.14159265358979323846;

// ---- pydub on mono PCM16 ------------------------------------------------

/// A mono PCM16 AudioSegment with prefix sums of squared samples.
class Segment {
public:
  Segment(std::vector<std::int16_t> samples, std::uint32_t rate)
      : samples_(std::move(samples)), rate_(rate) {
    squares_.resize(samples_.size() + 1, 0);
    for (std::size_t index = 0; index < samples_.size(); ++index) {
      const std::int64_t value = samples_[index];
      squares_[index + 1] = squares_[index] + value * value;
    }
  }

  [[nodiscard]] const std::vector<std::int16_t>& samples() const {
    return samples_;
  }

  /// len(segment): milliseconds, rounded half to even.
  [[nodiscard]] std::int64_t LengthMs() const {
    return static_cast<std::int64_t>(std::nearbyint(
        1000.0 * (static_cast<double>(samples_.size()) / rate_)));
  }

  /// Sample range of segment[start_ms:end_ms]. The range may run up to
  /// half a millisecond past the data, which pydub fills with zeros.
  void Range(std::int64_t start_ms, std::int64_t end_ms, std::int64_t* begin,
             std::int64_t* end) const {
    const std::int64_t length = LengthMs();
    start_ms = std::min(start_ms, length);
    end_ms = std::min(end_ms, length);
    *begin = static_cast<std::int64_t>(start_ms * (rate_ / 1000.0));
    *end = static_cast<std::int64_t>(end_ms * (rate_ / 1000.0));
  }

  /// segment[start_ms:end_ms].rms: audioop.rms, truncated to an integer.
  [[nodiscard]] std::uint64_t Rms(std::int64_t start_ms,
                                  std::int64_t end_ms) const {
    std::int64_t begin = 0, end = 0;
    Range(start_ms, end_ms, &begin, &end);
    if (end <= begin)
      return 0;
    const auto size = static_cast<std::int64_t>(samples_.size());
    const std::int64_t sum =
        squares_[static_cast<std::size_t>(std::min(end, size))] -
        squares_[static_cast<std::size_t>(std::min(begin, size))];
    return static_cast<std::uint64_t>(
        std::sqrt(static_cast<double>(sum) / static_cast<double>(end - begin)));
  }

  /// segment[start_ms:end_ms].
  [[nodiscard]] std::vector<std::int16_t> Slice(std::int64_t start_ms,
                                                std::int64_t end_ms) const {
    std::int64_t begin = 0, end = 0;
    Range(start_ms, end_ms, &begin, &end);
    std::vector<std::int16_t> slice;
    if (end <= begin)
      return slice;
    const auto size = static_cast<std::int64_t>(samples_.size());
    if (begin < size)
      slice.assign(samples_.begin() + begin,
                   samples_.begin() + std::min(end, size));
    slice.resize(static_cast<std::size_t>(end - begin), 0);
    return slice;
  }

private:
  std::vector<std::int16_t> samples_;
  std::uint32_t rate_;
  std::vector<std::int64_t> squares_;
};

double DbToAmplitude(double db) {
  return std::pow(10.0, db / 20.0);
}

/// AudioSegment.dBFS of segment[start_ms:end_ms].
double Dbfs(const Segment& segment, std::int64_t start_ms,
            std::int64_t end_ms) {
  const std::uint64_t rms = segment.Rms(start_ms, end_ms);
  if (rms == 0)
    return -std::numeric_limits<double>::infinity();
  // math.log(ratio, 10) divides natural logarithms.
  return 20.0 * (std::log(static_cast<double>(rms) / 32768.0) / std::log(10.0));
}

using Ranges = std::vector<std::pair<std::int64_t, std::int64_t>>;

/// pydub.silence.detect_silence.
Ranges DetectSilence(const Segment& segment, std::int64_t min_silence_ms,
                     double threshold_db, std::int64_t seek_ms) {
  const std::int64_t length = segment.LengthMs();
  if (length < min_silence_ms)
    return {};
  const double threshold = DbToAmplitude(threshold_db) * 32768.0;
  const std::int64_t last_start = length - min_silence_ms;
  std::vector<std::int64_t> starts;
  const auto check = [&](std::int64_t start) {
    if (static_cast<double>(segment.Rms(start, start + min_silence_ms)) <=
        threshold)
      starts.push_back(start);
  };
  for (std::int64_t start = 0; start <= last_start; start += seek_ms)
    check(start);
  if (last_start % seek_ms != 0)
    check(last_start);
  if (starts.empty())
    return {};
  Ranges ranges;
  std::int64_t previous = starts.front();
  std::int64_t range_start = previous;
  for (std::size_t index = 1; index < starts.size(); ++index) {
    const std::int64_t start = starts[index];
    const bool continuous = start == previous + seek_ms;
    const bool has_gap = start > previous + min_silence_ms;
    if (!continuous && has_gap) {
      ranges.emplace_back(range_start, previous + min_silence_ms);
      range_start = start;
    }
    previous = start;
  }
  ranges.emplace_back(range_start, previous + min_silence_ms);
  return ranges;
}

/// pydub.silence.detect_nonsilent.
Ranges DetectNonsilent(const Segment& segment, std::int64_t min_silence_ms,
                       double threshold_db, std::int64_t seek_ms) {
  const Ranges silent =
      DetectSilence(segment, min_silence_ms, threshold_db, seek_ms);
  const std::int64_t length = segment.LengthMs();
  if (silent.empty())
    return {{0, length}};
  if (silent.front().first == 0 && silent.front().second == length)
    return {};
  Ranges nonsilent;
  std::int64_t previous_end = 0;
  for (const auto& [start, end] : silent) {
    nonsilent.emplace_back(previous_end, start);
    previous_end = end;
  }
  if (silent.back().second != length)
    nonsilent.emplace_back(previous_end, length);
  if (nonsilent.front() == std::pair<std::int64_t, std::int64_t>{0, 0})
    nonsilent.erase(nonsilent.begin());
  return nonsilent;
}

/// The concatenated pieces of pydub.silence.split_on_silence.
std::vector<std::int16_t> JoinNonsilent(const Segment& segment,
                                        std::int64_t min_silence_ms,
                                        double threshold_db,
                                        std::int64_t keep_ms,
                                        std::int64_t seek_ms) {
  Ranges ranges =
      DetectNonsilent(segment, min_silence_ms, threshold_db, seek_ms);
  for (auto& [start, end] : ranges) {
    start -= keep_ms;
    end += keep_ms;
  }
  for (std::size_t index = 0; index + 1 < ranges.size(); ++index) {
    const std::int64_t last_end = ranges[index].second;
    const std::int64_t next_start = ranges[index + 1].first;
    if (next_start < last_end) {
      // Python's floor division.
      const std::int64_t sum = last_end + next_start;
      ranges[index].second = sum >= 0 ? sum / 2 : -((-sum + 1) / 2);
      ranges[index + 1].first = ranges[index].second;
    }
  }
  std::vector<std::int16_t> joined;
  const std::int64_t length = segment.LengthMs();
  for (const auto& [start, end] : ranges) {
    const auto piece =
        segment.Slice(std::max<std::int64_t>(start, 0), std::min(end, length));
    joined.insert(joined.end(), piece.begin(), piece.end());
  }
  return joined;
}

/// pydub.silence.detect_leading_silence with 10 ms chunks.
std::int64_t LeadingSilenceMs(const Segment& segment, double threshold_db) {
  const std::int64_t length = segment.LengthMs();
  std::int64_t trim = 0;
  while (Dbfs(segment, trim, trim + 10) < threshold_db && trim < length)
    trim += 10;
  return std::min(trim, length);
}

/// omnivoice.utils.audio.remove_silence_edges.
std::vector<std::int16_t> TrimEdges(std::vector<std::int16_t> samples,
                                    std::uint32_t rate, std::int64_t leading_ms,
                                    std::int64_t trailing_ms) {
  constexpr double kThreshold = -50.0;
  for (const std::int64_t keep : {leading_ms, trailing_ms}) {
    const Segment segment(std::move(samples), rate);
    const std::int64_t start =
        std::max<std::int64_t>(0, LeadingSilenceMs(segment, kThreshold) - keep);
    samples = segment.Slice(start, segment.LengthMs());
    std::reverse(samples.begin(), samples.end());
  }
  return samples;
}

std::vector<std::int16_t> ToPcm16(std::span<const float> samples) {
  std::vector<std::int16_t> pcm(samples.size());
  for (std::size_t index = 0; index < samples.size(); ++index) {
    const float scaled =
        std::clamp(samples[index] * 32768.0F, -32768.0F, 32767.0F);
    pcm[index] = static_cast<std::int16_t>(scaled);  // truncates toward zero
  }
  return pcm;
}

std::vector<float> FromPcm16(const std::vector<std::int16_t>& pcm) {
  std::vector<float> samples(pcm.size());
  for (std::size_t index = 0; index < pcm.size(); ++index)
    samples[index] = static_cast<float>(pcm[index]) / 32768.0F;
  return samples;
}

/// np.linspace(start, stop, count, dtype=np.float32)[index].
float Linspace(double start, double stop, std::size_t count,
               std::size_t index) {
  if (count > 1 && index == count - 1)
    return static_cast<float>(stop);
  const double step =
      count > 1 ? (stop - start) / static_cast<double>(count - 1) : 0.0;
  return static_cast<float>(static_cast<double>(index) * step + start);
}

}  // namespace

std::vector<float> Downmix(std::span<const float> interleaved,
                           std::uint32_t channels) {
  if (channels <= 1)
    return std::vector<float>(interleaved.begin(), interleaved.end());
  const std::size_t frames = interleaved.size() / channels;
  std::vector<float> mono(frames);
  for (std::size_t frame = 0; frame < frames; ++frame) {
    float sum = 0.0F;
    for (std::uint32_t channel = 0; channel < channels; ++channel)
      sum += interleaved[frame * channels + channel];
    mono[frame] = sum / static_cast<float>(channels);
  }
  return mono;
}

std::vector<float> Resample(std::span<const float> samples,
                            std::uint32_t from_rate, std::uint32_t to_rate) {
  if (from_rate == to_rate || samples.empty())
    return std::vector<float>(samples.begin(), samples.end());
  constexpr int kZeroCrossings = 6;
  constexpr double kRolloff = 0.99;
  const std::uint32_t gcd = std::gcd(from_rate, to_rate);
  const std::int64_t from = from_rate / gcd;
  const std::int64_t to = to_rate / gcd;
  const double base = static_cast<double>(std::min(from, to)) * kRolloff;
  const auto width = static_cast<std::int64_t>(
      std::ceil(kZeroCrossings * static_cast<double>(from) / base));
  const std::int64_t taps = 2 * width + from;
  // The kernel in float32, operation by operation as torchaudio builds it
  // for a float32 waveform.
  std::vector<float> kernel(static_cast<std::size_t>(to * taps));
  const auto base_f = static_cast<float>(base);
  const auto pi_f = static_cast<float>(kPi);
  const auto scale_f = static_cast<float>(base / static_cast<double>(from));
  for (std::int64_t phase = 0; phase < to; ++phase) {
    for (std::int64_t tap = 0; tap < taps; ++tap) {
      const float index =
          static_cast<float>(tap - width) / static_cast<float>(from);
      float t = static_cast<float>(-phase) / static_cast<float>(to) + index;
      t *= base_f;
      t = std::clamp(t, -static_cast<float>(kZeroCrossings),
                     static_cast<float>(kZeroCrossings));
      const float cosine =
          std::cos(t * pi_f / static_cast<float>(kZeroCrossings) / 2.0F);
      const float window = cosine * cosine;
      t *= pi_f;
      const float sinc = t == 0.0F ? 1.0F : std::sin(t) / t;
      kernel[static_cast<std::size_t>(phase * taps + tap)] =
          sinc * (window * scale_f);
    }
  }
  const auto length = static_cast<std::int64_t>(samples.size());
  const auto output_length = static_cast<std::int64_t>(
      std::ceil(static_cast<double>(to) * static_cast<double>(length) /
                static_cast<double>(from)));
  std::vector<float> output(static_cast<std::size_t>(output_length));
  // conv1d over the input padded by `width` before and `width + from`
  // after, stride `from`, one output channel per phase.
  for (std::int64_t out = 0; out < output_length; ++out) {
    const std::int64_t block = out / to;
    const std::int64_t phase = out % to;
    const float* weights = &kernel[static_cast<std::size_t>(phase * taps)];
    const std::int64_t origin = block * from - width;
    double sum = 0.0;
    const std::int64_t first = std::max<std::int64_t>(0, -origin);
    const std::int64_t last = std::min<std::int64_t>(taps, length - origin);
    for (std::int64_t tap = first; tap < last; ++tap)
      sum += static_cast<double>(weights[tap]) *
             samples[static_cast<std::size_t>(origin + tap)];
    output[static_cast<std::size_t>(out)] = static_cast<float>(sum);
  }
  return output;
}

double Rms(std::span<const float> samples) {
  if (samples.empty())
    return 0.0;
  double sum = 0.0;
  for (const float sample : samples)
    sum += static_cast<double>(sample) * sample;
  return std::sqrt(sum / static_cast<double>(samples.size()));
}

std::vector<float> RemoveSilence(std::span<const float> samples,
                                 std::uint32_t rate, int middle_ms,
                                 int leading_ms, int trailing_ms) {
  constexpr double kThreshold = -50.0;
  constexpr std::int64_t kSeekMs = 10;
  std::vector<std::int16_t> pcm = ToPcm16(samples);
  if (middle_ms > 0) {
    const Segment segment(std::move(pcm), rate);
    pcm = JoinNonsilent(segment, middle_ms, kThreshold, middle_ms, kSeekMs);
  }
  return FromPcm16(TrimEdges(std::move(pcm), rate, leading_ms, trailing_ms));
}

std::vector<float> FadeAndPad(std::span<const float> samples,
                              std::uint32_t rate, double pad_seconds,
                              double fade_seconds) {
  std::vector<float> processed(samples.begin(), samples.end());
  if (processed.empty())
    return processed;
  const auto fade = static_cast<std::size_t>(fade_seconds * rate);
  const auto pad = static_cast<std::size_t>(pad_seconds * rate);
  const std::size_t count = std::min(fade, processed.size() / 2);
  for (std::size_t index = 0; index < count; ++index) {
    processed[index] *= Linspace(0.0, 1.0, count, index);
    processed[processed.size() - count + index] *=
        Linspace(1.0, 0.0, count, index);
  }
  if (pad > 0) {
    processed.insert(processed.begin(), pad, 0.0F);
    processed.insert(processed.end(), pad, 0.0F);
  }
  return processed;
}

std::vector<float> CrossFadeChunks(
    const std::vector<std::vector<float>>& chunks, std::uint32_t rate,
    double gap_seconds) {
  if (chunks.empty())
    return {};
  std::vector<float> merged = chunks.front();
  if (chunks.size() == 1)
    return merged;
  const std::size_t fade = static_cast<std::size_t>(gap_seconds * rate) / 3;
  for (std::size_t chunk = 1; chunk < chunks.size(); ++chunk) {
    const std::size_t fade_out = std::min(fade, merged.size());
    for (std::size_t index = 0; index < fade_out; ++index)
      merged[merged.size() - fade_out + index] *=
          Linspace(1.0, 0.0, fade_out, index);
    merged.insert(merged.end(), fade, 0.0F);
    const std::vector<float>& next = chunks[chunk];
    const std::size_t fade_in = std::min(fade, next.size());
    const std::size_t offset = merged.size();
    merged.insert(merged.end(), next.begin(), next.end());
    for (std::size_t index = 0; index < fade_in; ++index)
      merged[offset + index] *= Linspace(0.0, 1.0, fade_in, index);
  }
  return merged;
}

std::optional<ReferenceAudio> PrepareReference(
    std::span<const float> interleaved, std::uint32_t channels,
    std::uint32_t source_rate, std::uint32_t rate, std::size_t frame_samples,
    std::string* error) {
  const auto fail = [error](const char* message) {
    if (error != nullptr)
      *error = message;
    return std::nullopt;
  };
  if (channels == 0 || source_rate == 0 || interleaved.size() < channels)
    return fail("reference audio is empty");
  std::vector<float> mono =
      Resample(Downmix(interleaved, channels), source_rate, rate);
  ReferenceAudio reference;
  reference.rms = Rms(mono);
  if (reference.rms > 0.0 && reference.rms < 0.1) {
    // ref_wav * 0.1 / ref_rms, both in float32.
    const auto level = static_cast<float>(reference.rms);
    for (float& sample : mono)
      sample = sample * 0.1F / level;
  }
  mono = RemoveSilence(mono, rate, 200, 100, 200);
  mono.resize(mono.size() - mono.size() % frame_samples);
  if (mono.empty())
    return fail(
        "reference audio is shorter than one codec frame after "
        "silence removal");
  reference.samples = std::move(mono);
  return reference;
}

std::vector<float> PostProcess(std::span<const float> samples,
                               std::uint32_t rate,
                               std::optional<double> reference_rms) {
  std::vector<float> audio = RemoveSilence(samples, rate, 500, 100, 100);
  if (reference_rms.has_value() && *reference_rms < 0.1) {
    const auto level = static_cast<float>(*reference_rms);
    for (float& sample : audio)
      sample = sample * level / 0.1F;
  } else if (!reference_rms.has_value() && !audio.empty()) {
    float peak = 0.0F;
    for (const float sample : audio)
      peak = std::max(peak, std::abs(sample));
    if (peak > 1e-6F) {
      for (float& sample : audio)
        sample = sample / peak * 0.5F;
    }
  }
  return FadeAndPad(audio, rate);
}

}  // namespace gufo::models::omnivoice
