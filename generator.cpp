// OmniVoice's backbone over the CFG batch; the Qwen3 layers compute what
// Hugging Face transformers computes (Apache-2.0, see
// licenses/transformers.txt).
#include "generator.hpp"

#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <initializer_list>
#include <span>
#include <stdexcept>
#include <string_view>
#include <thread>
#include <utility>

#include "device.hpp"
#include "gemm.hpp"
#include "kernels/generator_kernels.hpp"
#include "safetensors.hpp"

namespace gufo::models::omnivoice {
namespace {

using kernels::Half;

/// Converts float32 tensors to float16 on the host, in parallel.
void ConvertToHalf(std::span<const float> input, _Float16* output) {
  const std::size_t count = input.size();
  const std::size_t workers =
      std::clamp<std::size_t>(std::thread::hardware_concurrency(), 1, 16);
  const std::size_t chunk = (count + workers - 1) / workers;
  std::vector<std::thread> threads;
  for (std::size_t worker = 0; worker < workers; ++worker) {
    const std::size_t begin = worker * chunk;
    const std::size_t end = std::min(count, begin + chunk);
    if (begin >= end)
      break;
    threads.emplace_back([&input, output, begin, end] {
      for (std::size_t index = begin; index < end; ++index)
        output[index] = static_cast<_Float16>(input[index]);
    });
  }
  for (std::thread& thread : threads)
    thread.join();
}

struct Layer {
  DeviceBuffer<Half> qkv;      // [(heads + 2 kv_heads) * head_dim, hidden]
  DeviceBuffer<Half> output;   // [hidden, heads * head_dim]
  DeviceBuffer<Half> gate_up;  // [2 * intermediate, hidden]
  DeviceBuffer<Half> down;     // [hidden, intermediate]
  DeviceBuffer<float> input_norm;
  DeviceBuffer<float> post_norm;
  DeviceBuffer<float> q_norm;
  DeviceBuffer<float> k_norm;
};

/// Activations of the packed CFG batch, grown on demand.
struct Workspace {
  DeviceBuffer<float> text;        // [text rows, hidden]
  DeviceBuffer<std::int32_t> ids;  // [codebooks, rows]
  DeviceBuffer<std::int32_t> positions;
  DeviceBuffer<std::int32_t> target_rows;
  DeviceBuffer<float> hidden;  // [rows, hidden]
  DeviceBuffer<Half> normed;   // [rows, max(hidden, ...)]
  DeviceBuffer<float> qkv;     // [rows, qkv width]
  DeviceBuffer<Half> q, k, v;  // [heads, rows, head_dim]
  DeviceBuffer<float> scores;  // [heads, longest, longest]
  DeviceBuffer<Half> probabilities;
  DeviceBuffer<float> attention;   // [heads, rows, head_dim]
  DeviceBuffer<Half> merged;       // [rows, heads * head_dim]
  DeviceBuffer<float> gate_up;     // [rows, 2 * intermediate]
  DeviceBuffer<Half> activated;    // [rows, intermediate]
  DeviceBuffer<float> final_rows;  // [2 * frames, hidden]
  DeviceBuffer<Half> final_normed;
  DeviceBuffer<float> logits;  // [2 * frames, codebooks * vocab]
  DeviceBuffer<std::int32_t> prediction;
  DeviceBuffer<float> confidence;
};

}  // namespace

struct Generator::State {
  Config config;
  std::unique_ptr<SafetensorsFile> weights;
  const Tensor* text_embeddings = nullptr;  // host, float32
  DeviceBuffer<float> audio_embeddings;     // [codebooks * vocab, hidden]
  DeviceBuffer<Half> audio_heads;           // [codebooks * vocab, hidden]
  DeviceBuffer<float> final_norm;
  std::vector<Layer> layers;
  Workspace work;
  hipStream_t stream = nullptr;
  hipblasHandle_t blas = nullptr;
  std::unique_ptr<HalfGemm> gemm;

  ~State() {
    gemm.reset();
    if (blas != nullptr)
      (void)hipblasDestroy(blas);
    if (stream != nullptr)
      (void)hipStreamDestroy(stream);
  }

  [[nodiscard]] std::size_t QkvWidth() const {
    const BackboneConfig& b = config.backbone;
    return (b.heads + 2 * b.kv_heads) * b.head_dim;
  }

  /// Y[rows, out] (+)= X[rows, in] * W[out, in]^T, float16 inputs, float32
  /// output; `accumulate` adds to Y (the residual stream).
  void Linear(const Half* x, const Half* w, float* y, std::size_t rows,
              std::size_t in, std::size_t out, bool accumulate) const {
    gemm->Linear(x, w, y, rows, in, out, accumulate);
  }

  /// Bidirectional attention within one sequence of `length` rows starting
  /// at `first`, for every head at once.
  void Attention(std::size_t first, std::size_t length, std::size_t rows) {
    const BackboneConfig& b = config.backbone;
    const std::size_t d = b.head_dim;
    const Half* q = work.q.get() + first * d;
    const Half* k = work.k.get() + first * d;
    const Half* v = work.v.get() + first * d;
    // Tuning is shared by sequences and batches of similar size.
    const std::size_t bucket = TuningBucket(length);
    const std::size_t bucket_rows = TuningBucket(rows);
    // S[i][j] = q_i . k_j, row-major [length][length] per head.
    MatmulShape scores;
    scores.transpose_a = true;
    scores.m = length;
    scores.n = length;
    scores.k = d;
    scores.lda = d;
    scores.ldb = d;
    scores.ldc = length;
    scores.stride_a = static_cast<std::int64_t>(rows * d);
    scores.stride_b = scores.stride_a;
    scores.stride_c = static_cast<std::int64_t>(length * length);
    scores.batch = b.heads;
    MatmulShape scores_bucket = scores;
    scores_bucket.m = scores_bucket.n = scores_bucket.ldc = bucket;
    scores_bucket.stride_a = scores_bucket.stride_b =
        static_cast<std::int64_t>(bucket_rows * d);
    scores_bucket.stride_c = static_cast<std::int64_t>(bucket * bucket);
    gemm->Matmul(scores, scores_bucket, k, q, work.scores.get());
    kernels::Softmax(work.scores.get(), work.probabilities.get(),
                     b.heads * length, length,
                     1.0F / std::sqrt(static_cast<float>(d)), stream);
    // O[i] = sum_j P[i][j] v_j into the head-major attention rows.
    MatmulShape values;
    values.m = d;
    values.n = length;
    values.k = length;
    values.lda = d;
    values.ldb = length;
    values.ldc = d;
    values.stride_a = static_cast<std::int64_t>(rows * d);
    values.stride_b = static_cast<std::int64_t>(length * length);
    values.stride_c = values.stride_a;
    values.batch = b.heads;
    MatmulShape values_bucket = values;
    values_bucket.n = values_bucket.k = values_bucket.ldb = bucket;
    values_bucket.stride_a = values_bucket.stride_c =
        static_cast<std::int64_t>(bucket_rows * d);
    values_bucket.stride_b = static_cast<std::int64_t>(bucket * bucket);
    gemm->Matmul(values, values_bucket, v, work.probabilities.get(),
                 work.attention.get() + first * d);
  }

  /// Uploads the packed CFG batch for `input` and returns its row count.
  std::size_t Prepare(const ConditionalInput& input) {
    const BackboneConfig& b = config.backbone;
    const std::size_t length = input.length;
    const std::size_t frames = input.target_frames;
    const std::size_t rows = length + frames;
    const std::size_t codebooks = input.codebooks;
    const std::size_t heads = b.heads;
    Workspace& w = work;
    w.text.Reserve(std::max<std::size_t>(input.audio_start, 1) * b.hidden_size);
    w.ids.Reserve(codebooks * rows);
    w.positions.Reserve(rows);
    w.target_rows.Reserve(2 * frames);
    w.hidden.Reserve(rows * b.hidden_size);
    w.normed.Reserve(rows * std::max(b.hidden_size, b.intermediate_size));
    w.qkv.Reserve(rows * QkvWidth());
    w.q.Reserve(heads * rows * b.head_dim);
    w.k.Reserve(heads * rows * b.head_dim);
    w.v.Reserve(heads * rows * b.head_dim);
    w.scores.Reserve(heads * length * length);
    w.probabilities.Reserve(heads * length * length);
    w.attention.Reserve(heads * rows * b.head_dim);
    w.merged.Reserve(rows * heads * b.head_dim);
    w.gate_up.Reserve(rows * 2 * b.intermediate_size);
    w.activated.Reserve(rows * b.intermediate_size);
    w.final_rows.Reserve(2 * frames * b.hidden_size);
    w.final_normed.Reserve(2 * frames * b.hidden_size);
    w.logits.Reserve(2 * frames * codebooks * config.audio_vocab_size);
    w.prediction.Reserve(codebooks * frames);
    w.confidence.Reserve(codebooks * frames);
    // Text embeddings: Hugging Face's float32 rows of the first codebook's
    // IDs.
    std::vector<float> text(input.audio_start * b.hidden_size);
    const auto* table = reinterpret_cast<const float*>(text_embeddings->data);
    for (std::size_t t = 0; t < input.audio_start; ++t) {
      const auto id = static_cast<std::size_t>(input.ids[t]);
      if (id >= b.vocab_size)
        throw GpuError("text token outside the vocabulary");
      std::memcpy(&text[t * b.hidden_size], table + id * b.hidden_size,
                  b.hidden_size * sizeof(float));
    }
    Check(hipMemcpyAsync(w.text.get(), text.data(), text.size() * sizeof(float),
                         hipMemcpyHostToDevice, stream),
          "upload text embeddings");
    std::vector<std::int32_t> positions(rows);
    std::vector<std::int32_t> target_rows(2 * frames);
    for (std::size_t r = 0; r < length; ++r)
      positions[r] = static_cast<std::int32_t>(r);
    for (std::size_t f = 0; f < frames; ++f) {
      positions[length + f] = static_cast<std::int32_t>(f);
      target_rows[f] = static_cast<std::int32_t>(length - frames + f);
      target_rows[frames + f] = static_cast<std::int32_t>(length + f);
    }
    Check(hipMemcpyAsync(w.positions.get(), positions.data(),
                         rows * sizeof(std::int32_t), hipMemcpyHostToDevice,
                         stream),
          "upload positions");
    Check(hipMemcpyAsync(w.target_rows.get(), target_rows.data(),
                         target_rows.size() * sizeof(std::int32_t),
                         hipMemcpyHostToDevice, stream),
          "upload target rows");
    UploadIds(input);
    // The uploads read host buffers that go out of scope here.
    Check(hipStreamSynchronize(stream), "upload");
    return rows;
  }

  /// Codes of the packed batch: the conditional sequence, then its target
  /// frames again as the unconditional sequence.
  void UploadIds(const ConditionalInput& input) {
    const std::size_t length = input.length;
    const std::size_t frames = input.target_frames;
    const std::size_t rows = length + frames;
    host_ids.resize(input.codebooks * rows);
    for (std::size_t c = 0; c < input.codebooks; ++c) {
      const std::int32_t* source = &input.ids[c * length];
      std::int32_t* row = &host_ids[c * rows];
      std::copy(source, source + length, row);
      std::copy(source + length - frames, source + length, row + length);
    }
    Check(hipMemcpyAsync(work.ids.get(), host_ids.data(),
                         host_ids.size() * sizeof(std::int32_t),
                         hipMemcpyHostToDevice, stream),
          "upload codes");
  }

  /// The backbone over the packed batch and the audio heads on the target
  /// rows; leaves the logits in work.logits.
  void RunForward(const ConditionalInput& input, std::size_t rows) {
    const BackboneConfig& b = config.backbone;
    const std::size_t length = input.length;
    const std::size_t frames = input.target_frames;
    const std::size_t hidden = b.hidden_size;
    Workspace& w = work;
    kernels::Embed(w.text.get(), input.audio_start, audio_embeddings.get(),
                   w.ids.get(), w.hidden.get(), rows, hidden, input.codebooks,
                   config.audio_vocab_size, stream);
    for (const Layer& layer : layers) {
      kernels::RmsNorm(w.hidden.get(), layer.input_norm.get(), w.normed.get(),
                       rows, hidden, b.rms_norm_eps, stream);
      Linear(w.normed.get(), layer.qkv.get(), w.qkv.get(), rows, hidden,
             QkvWidth(), false);
      kernels::QkNormRope(w.qkv.get(), layer.q_norm.get(), layer.k_norm.get(),
                          w.positions.get(), w.q.get(), w.k.get(), w.v.get(),
                          rows, b.heads, b.kv_heads, b.head_dim, b.rms_norm_eps,
                          b.rope_theta, stream);
      Attention(0, length, rows);
      Attention(length, frames, rows);
      kernels::MergeHeads(w.attention.get(), w.merged.get(), rows, b.heads,
                          b.head_dim, stream);
      Linear(w.merged.get(), layer.output.get(), w.hidden.get(), rows,
             b.heads * b.head_dim, hidden, true);
      kernels::RmsNorm(w.hidden.get(), layer.post_norm.get(), w.normed.get(),
                       rows, hidden, b.rms_norm_eps, stream);
      Linear(w.normed.get(), layer.gate_up.get(), w.gate_up.get(), rows, hidden,
             2 * b.intermediate_size, false);
      kernels::SwiGlu(w.gate_up.get(), w.activated.get(), rows,
                      b.intermediate_size, stream);
      Linear(w.activated.get(), layer.down.get(), w.hidden.get(), rows,
             b.intermediate_size, hidden, true);
    }
    kernels::GatherRows(w.hidden.get(), w.target_rows.get(), w.final_rows.get(),
                        2 * frames, hidden, stream);
    kernels::RmsNorm(w.final_rows.get(), final_norm.get(), w.final_normed.get(),
                     2 * frames, hidden, b.rms_norm_eps, stream);
    Linear(w.final_normed.get(), audio_heads.get(), w.logits.get(), 2 * frames,
           hidden, input.codebooks * config.audio_vocab_size, false);
  }

  std::vector<std::int32_t> host_ids;
};

namespace {

/// Uploads a float32 tensor of this exact shape.
void UploadFloat(const SafetensorsFile& file, std::string_view name,
                 std::initializer_list<std::size_t> shape,
                 DeviceBuffer<float>* buffer) {
  std::string error;
  const Tensor* tensor = file.Expect(name, DType::kF32, shape, &error);
  if (tensor == nullptr)
    throw GpuError(error);
  *buffer = DeviceBuffer<float>(tensor->Elements());
  Check(hipMemcpy(buffer->get(), tensor->data, tensor->bytes,
                  hipMemcpyHostToDevice),
        "upload weights");
}

/// Converts float32 tensors with `columns` columns to float16 and uploads
/// them stacked by rows.
void UploadHalf(
    const SafetensorsFile& file,
    std::initializer_list<std::pair<std::string, std::size_t>> parts,
    std::size_t columns, DeviceBuffer<Half>* buffer) {
  std::size_t rows = 0;
  std::vector<const Tensor*> tensors;
  for (const auto& [name, part_rows] : parts) {
    std::string error;
    const Tensor* tensor =
        file.Expect(name, DType::kF32, {part_rows, columns}, &error);
    if (tensor == nullptr)
      throw GpuError(error);
    tensors.push_back(tensor);
    rows += part_rows;
  }
  std::vector<_Float16> host(rows * columns);
  std::size_t offset = 0;
  for (const Tensor* tensor : tensors) {
    ConvertToHalf(
        std::span<const float>(reinterpret_cast<const float*>(tensor->data),
                               tensor->Elements()),
        host.data() + offset);
    offset += tensor->Elements();
  }
  *buffer = DeviceBuffer<Half>(host.size());
  Check(hipMemcpy(buffer->get(), host.data(), host.size() * sizeof(_Float16),
                  hipMemcpyHostToDevice),
        "upload weights");
}

void SetError(std::string* error, const std::string& message) {
  if (error != nullptr)
    *error = message;
}

}  // namespace

Generator::Generator(std::unique_ptr<State> state) : state_(std::move(state)) {}
Generator::~Generator() = default;

std::unique_ptr<Generator> Generator::Load(
    const Config& config, const std::filesystem::path& model_root,
    std::string* error) {
  auto state = std::make_unique<State>();
  state->config = config;
  try {
    std::string open_error;
    state->weights =
        SafetensorsFile::Open(model_root / "model.safetensors", &open_error);
    if (state->weights == nullptr)
      throw GpuError(open_error);
    const SafetensorsFile& file = *state->weights;
    const BackboneConfig& b = config.backbone;
    const std::size_t audio_rows =
        config.audio_codebooks * config.audio_vocab_size;
    std::string tensor_error;
    state->text_embeddings =
        file.Expect("llm.embed_tokens.weight", DType::kF32,
                    {b.vocab_size, b.hidden_size}, &tensor_error);
    if (state->text_embeddings == nullptr)
      throw GpuError(tensor_error);
    Check(hipStreamCreateWithFlags(&state->stream, hipStreamNonBlocking),
          "hipStreamCreate");
    Check(hipblasCreate(&state->blas), "hipblasCreate");
    Check(hipblasSetStream(state->blas, state->stream), "hipblasSetStream");
    // Reproducible sums: the same seed gives the same audio.
    Check(hipblasSetAtomicsMode(state->blas, HIPBLAS_ATOMICS_NOT_ALLOWED),
          "hipblasSetAtomicsMode");
    state->gemm = std::make_unique<HalfGemm>(state->blas, state->stream);
    UploadFloat(file, "audio_embeddings.weight", {audio_rows, b.hidden_size},
                &state->audio_embeddings);
    UploadHalf(file, {{"audio_heads.weight", audio_rows}}, b.hidden_size,
               &state->audio_heads);
    UploadFloat(file, "llm.norm.weight", {b.hidden_size}, &state->final_norm);
    const std::size_t q_rows = b.heads * b.head_dim;
    const std::size_t kv_rows = b.kv_heads * b.head_dim;
    state->layers.resize(b.layers);
    for (std::size_t index = 0; index < b.layers; ++index) {
      const std::string prefix = "llm.layers." + std::to_string(index) + ".";
      Layer& layer = state->layers[index];
      UploadHalf(file,
                 {{prefix + "self_attn.q_proj.weight", q_rows},
                  {prefix + "self_attn.k_proj.weight", kv_rows},
                  {prefix + "self_attn.v_proj.weight", kv_rows}},
                 b.hidden_size, &layer.qkv);
      UploadHalf(file, {{prefix + "self_attn.o_proj.weight", b.hidden_size}},
                 q_rows, &layer.output);
      UploadHalf(file,
                 {{prefix + "mlp.gate_proj.weight", b.intermediate_size},
                  {prefix + "mlp.up_proj.weight", b.intermediate_size}},
                 b.hidden_size, &layer.gate_up);
      UploadHalf(file, {{prefix + "mlp.down_proj.weight", b.hidden_size}},
                 b.intermediate_size, &layer.down);
      UploadFloat(file, prefix + "input_layernorm.weight", {b.hidden_size},
                  &layer.input_norm);
      UploadFloat(file, prefix + "post_attention_layernorm.weight",
                  {b.hidden_size}, &layer.post_norm);
      UploadFloat(file, prefix + "self_attn.q_norm.weight", {b.head_dim},
                  &layer.q_norm);
      UploadFloat(file, prefix + "self_attn.k_norm.weight", {b.head_dim},
                  &layer.k_norm);
    }
    // Only the text embeddings stay on the host; their rows fault back in
    // per request.
    file.ReleasePages();
  } catch (const std::exception& failure) {
    SetError(error, std::string("OmniVoice generator: ") + failure.what());
    return nullptr;
  }
  return std::unique_ptr<Generator>(new Generator(std::move(state)));
}

bool Generator::Forward(const ConditionalInput& input,
                        std::vector<float>* logits, std::string* error) {
  try {
    State& s = *state_;
    const std::size_t rows = s.Prepare(input);
    s.RunForward(input, rows);
    logits->resize(2 * input.target_frames * input.codebooks *
                   s.config.audio_vocab_size);
    Check(hipMemcpyAsync(logits->data(), s.work.logits.get(),
                         logits->size() * sizeof(float), hipMemcpyDeviceToHost,
                         s.stream),
          "download logits");
    Check(hipStreamSynchronize(s.stream), "forward");
  } catch (const std::exception& failure) {
    SetError(error, std::string("OmniVoice forward: ") + failure.what());
    return false;
  }
  return true;
}

bool Generator::Score(const ConditionalInput& input, float guidance,
                      std::vector<std::int32_t>* prediction,
                      std::vector<float>* confidence, std::string* error) {
  try {
    State& s = *state_;
    const std::size_t rows = s.Prepare(input);
    s.RunForward(input, rows);
    const std::size_t count = input.codebooks * input.target_frames;
    kernels::Score(s.work.logits.get(), s.work.prediction.get(),
                   s.work.confidence.get(), input.target_frames,
                   input.codebooks, s.config.audio_vocab_size,
                   s.config.audio_mask_id, guidance, s.stream);
    prediction->resize(count);
    confidence->resize(count);
    Check(hipMemcpyAsync(prediction->data(), s.work.prediction.get(),
                         count * sizeof(std::int32_t), hipMemcpyDeviceToHost,
                         s.stream),
          "download predictions");
    Check(
        hipMemcpyAsync(confidence->data(), s.work.confidence.get(),
                       count * sizeof(float), hipMemcpyDeviceToHost, s.stream),
        "download confidences");
    Check(hipStreamSynchronize(s.stream), "score");
  } catch (const std::exception& failure) {
    SetError(error, std::string("OmniVoice score: ") + failure.what());
    return false;
  }
  return true;
}

bool Generator::Generate(const ConditionalInput& input,
                         const DecodingConfig& decoding, std::uint64_t seed,
                         const std::function<bool()>& cancelled,
                         GenerationResult* result, std::string* error) {
  try {
    State& s = *state_;
    const std::size_t codebooks = input.codebooks;
    const std::size_t frames = input.target_frames;
    const std::size_t count = codebooks * frames;
    const std::uint32_t mask = s.config.audio_mask_id;
    ConditionalInput current = input;
    const std::size_t rows = s.Prepare(current);
    result->codes.assign(count, static_cast<std::int32_t>(mask));
    result->unmasked_at.assign(count, -1);
    std::vector<std::int32_t> prediction(count);
    std::vector<float> confidence(count);
    PositionSelector selector(decoding, seed);
    const std::vector<std::size_t> schedule =
        UnmaskSchedule(count, decoding.steps, decoding.t_shift);
    for (std::size_t step = 0; step < schedule.size(); ++step) {
      if (schedule[step] == 0)
        continue;
      if (cancelled && cancelled()) {
        SetError(error, "cancelled");
        return false;
      }
      s.RunForward(current, rows);
      kernels::Score(s.work.logits.get(), s.work.prediction.get(),
                     s.work.confidence.get(), frames, codebooks,
                     s.config.audio_vocab_size, mask, decoding.guidance,
                     s.stream);
      Check(hipMemcpyAsync(prediction.data(), s.work.prediction.get(),
                           count * sizeof(std::int32_t), hipMemcpyDeviceToHost,
                           s.stream),
            "download predictions");
      Check(hipMemcpyAsync(confidence.data(), s.work.confidence.get(),
                           count * sizeof(float), hipMemcpyDeviceToHost,
                           s.stream),
            "download confidences");
      Check(hipStreamSynchronize(s.stream), "unmasking step");
      for (const std::size_t index :
           selector.Fill(schedule[step], codebooks, frames, prediction,
                         confidence, mask, &result->codes))
        result->unmasked_at[index] = static_cast<std::int32_t>(step);
      // Write the filled target back into both sequences.
      for (std::size_t c = 0; c < codebooks; ++c)
        std::copy(&result->codes[c * frames],
                  &result->codes[c * frames] + frames,
                  &current.ids[c * current.length + current.length - frames]);
      s.UploadIds(current);
    }
    Check(hipStreamSynchronize(s.stream), "generation");
  } catch (const std::exception& failure) {
    SetError(error, std::string("OmniVoice generation: ") + failure.what());
    return false;
  }
  return true;
}

}  // namespace gufo::models::omnivoice
