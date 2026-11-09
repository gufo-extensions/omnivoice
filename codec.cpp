// The Higgs Audio v2 tokenizer's loading and decoder, ported from Hugging
// Face transformers' HiggsAudioV2TokenizerModel.decode and DAC decoder
// (Apache-2.0, see licenses/transformers.txt): RVQ decode, fc2 and the DAC
// decoder with stride-parity output padding and without the final tanh. The
// encoder is in codec_encoder.cpp.
#include "codec.hpp"

#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <initializer_list>
#include <string_view>
#include <utility>

#include "codec_state.hpp"
#include "device.hpp"
#include "kernels/codec_kernels.hpp"
#include "safetensors.hpp"

namespace gufo::models::omnivoice {
const Tensor& Require(const SafetensorsFile& file, std::string_view name,
                      std::initializer_list<std::size_t> shape) {
  std::string error;
  const Tensor* tensor = file.Expect(name, DType::kF32, shape, &error);
  if (tensor == nullptr)
    throw GpuError(error);
  return *tensor;
}

DeviceBuffer<float> Upload(const float* data, std::size_t count) {
  DeviceBuffer<float> buffer(count);
  Check(hipMemcpy(buffer.get(), data, count * sizeof(float),
                  hipMemcpyHostToDevice),
        "upload codec weights");
  return buffer;
}

DeviceBuffer<float> UploadTensor(const SafetensorsFile& file,
                                 std::string_view name,
                                 std::initializer_list<std::size_t> shape) {
  const Tensor& tensor = Require(file, name, shape);
  return Upload(reinterpret_cast<const float*>(tensor.data), tensor.Elements());
}

Conv LoadConv(const SafetensorsFile& file, const std::string& prefix,
              std::size_t in, std::size_t out, std::size_t kernel,
              std::size_t dilation) {
  const Tensor& weight = Require(file, prefix + ".weight", {out, in, kernel});
  const auto* source = reinterpret_cast<const float*>(weight.data);
  std::vector<float> rearranged(out * in * kernel);
  for (std::size_t o = 0; o < out; ++o)
    for (std::size_t i = 0; i < in; ++i)
      for (std::size_t k = 0; k < kernel; ++k)
        rearranged[(k * out + o) * in + i] = source[(o * in + i) * kernel + k];
  Conv conv;
  conv.weight = Upload(rearranged.data(), rearranged.size());
  conv.bias = UploadTensor(file, prefix + ".bias", {out});
  conv.in = in;
  conv.out = out;
  conv.kernel = kernel;
  conv.dilation = dilation;
  return conv;
}

ConvTranspose LoadConvTranspose(const SafetensorsFile& file,
                                const std::string& prefix, std::size_t in,
                                std::size_t out, std::size_t stride) {
  const std::size_t kernel = 2 * stride;
  const Tensor& weight = Require(file, prefix + ".weight", {in, out, kernel});
  const auto* source = reinterpret_cast<const float*>(weight.data);
  std::vector<float> rearranged(kernel * out * in);
  for (std::size_t i = 0; i < in; ++i)
    for (std::size_t o = 0; o < out; ++o)
      for (std::size_t k = 0; k < kernel; ++k)
        rearranged[(k * out + o) * in + i] = source[(i * out + o) * kernel + k];
  ConvTranspose conv;
  conv.weight = Upload(rearranged.data(), rearranged.size());
  conv.bias = UploadTensor(file, prefix + ".bias", {out});
  conv.in = in;
  conv.out = out;
  conv.stride = stride;
  return conv;
}

namespace {

void SetError(std::string* error, const std::string& message) {
  if (error != nullptr)
    *error = message;
}

}  // namespace

Codec::Codec(std::unique_ptr<State> state) : state_(std::move(state)) {}
Codec::~Codec() = default;

std::unique_ptr<Codec> Codec::Load(const Config& config,
                                   const std::filesystem::path& model_root,
                                   std::string* error) {
  auto state = std::make_unique<State>();
  state->config = config;
  try {
    std::string open_error;
    const auto file = SafetensorsFile::Open(
        model_root / "audio_tokenizer" / "model.safetensors", &open_error);
    if (file == nullptr)
      throw GpuError(open_error);
    const CodecConfig& codec = config.codec;
    const AcousticConfig& acoustic = codec.acoustic;
    const std::size_t hidden =
        codec.semantic.hidden_size + acoustic.latent_channels;
    Check(hipStreamCreateWithFlags(&state->stream, hipStreamNonBlocking),
          "hipStreamCreate");
    Check(hipblasCreate(&state->blas), "hipblasCreate");
    Check(hipblasSetStream(state->blas, state->stream), "hipblasSetStream");
    // Reproducible sums: the same seed gives the same audio.
    Check(hipblasSetAtomicsMode(state->blas, HIPBLAS_ATOMICS_NOT_ALLOWED),
          "hipblasSetAtomicsMode");
    // RVQ decode tables: codebook entries through each project_out.
    state->tables =
        DeviceBuffer<float>(codec.codebooks * codec.codebook_size * hidden);
    for (std::size_t c = 0; c < codec.codebooks; ++c) {
      const std::string prefix = "quantizer.quantizers." + std::to_string(c);
      DeviceBuffer<float> embed =
          UploadTensor(*file, prefix + ".codebook.embed",
                       {codec.codebook_size, codec.codebook_dim});
      DeviceBuffer<float> project = UploadTensor(
          *file, prefix + ".project_out.weight", {hidden, codec.codebook_dim});
      DeviceBuffer<float> bias =
          UploadTensor(*file, prefix + ".project_out.bias", {hidden});
      float* table = state->tables.get() + c * codec.codebook_size * hidden;
      const float alpha = 1.0F;
      const float beta = 0.0F;
      Check(hipblasSgemm(state->blas, HIPBLAS_OP_T, HIPBLAS_OP_N,
                         static_cast<int>(hidden),
                         static_cast<int>(codec.codebook_size),
                         static_cast<int>(codec.codebook_dim), &alpha,
                         project.get(), static_cast<int>(codec.codebook_dim),
                         embed.get(), static_cast<int>(codec.codebook_dim),
                         &beta, table, static_cast<int>(hidden)),
            "RVQ table");
      kernels::AddColumnBias(table, bias.get(), codec.codebook_size, hidden,
                             state->stream);
      Check(hipStreamSynchronize(state->stream), "RVQ table");
    }
    state->fc2_weight =
        UploadTensor(*file, "fc2.weight", {acoustic.latent_channels, hidden});
    state->fc2_bias =
        UploadTensor(*file, "fc2.bias", {acoustic.latent_channels});
    state->input_conv =
        LoadConv(*file, "acoustic_decoder.conv1", acoustic.latent_channels,
                 acoustic.decoder_channels, kResidualKernel, 1);
    std::size_t channels = acoustic.decoder_channels;
    for (std::size_t index = 0; index < acoustic.upsampling_ratios.size();
         ++index) {
      const std::string prefix =
          "acoustic_decoder.block." + std::to_string(index);
      const std::size_t out = channels / 2;
      DecoderBlock block;
      block.alpha =
          UploadTensor(*file, prefix + ".snake1.alpha", {1, channels, 1});
      block.upsample =
          LoadConvTranspose(*file, prefix + ".conv_t1", channels, out,
                            acoustic.upsampling_ratios[index]);
      for (std::size_t unit = 0; unit < block.units.size(); ++unit) {
        const std::string name =
            prefix + ".res_unit" + std::to_string(unit + 1);
        ResidualUnit& residual = block.units[unit];
        residual.alpha1 =
            UploadTensor(*file, name + ".snake1.alpha", {1, out, 1});
        residual.alpha2 =
            UploadTensor(*file, name + ".snake2.alpha", {1, out, 1});
        residual.conv1 = LoadConv(*file, name + ".conv1", out, out,
                                  kResidualKernel, kDilations[unit]);
        residual.conv2 = LoadConv(*file, name + ".conv2", out, out, 1, 1);
      }
      state->blocks.push_back(std::move(block));
      channels = out;
    }
    state->final_alpha =
        UploadTensor(*file, "acoustic_decoder.snake1.alpha", {1, channels, 1});
    state->output_conv = LoadConv(*file, "acoustic_decoder.conv2", channels, 1,
                                  kResidualKernel, 1);
    LoadEncoder(config, *file, state->blas, state->stream, &state->encoder);
  } catch (const std::exception& failure) {
    SetError(error,
             std::string("OmniVoice audio tokenizer: ") + failure.what());
    return nullptr;
  }
  return std::unique_ptr<Codec>(new Codec(std::move(state)));
}

bool Codec::Decode(const std::vector<std::int32_t>& codes, std::size_t frames,
                   std::vector<float>* audio, std::string* error) {
  try {
    State& s = *state_;
    const CodecConfig& codec = s.config.codec;
    const AcousticConfig& acoustic = codec.acoustic;
    const std::size_t hidden =
        codec.semantic.hidden_size + acoustic.latent_channels;
    if (codes.size() != codec.codebooks * frames)
      throw GpuError("code count doesn't match the frames");
    for (const std::int32_t code : codes) {
      if (code < 0 || static_cast<std::size_t>(code) >= codec.codebook_size)
        throw GpuError("code outside the codebook");
    }
    s.Reserve(frames);
    Check(hipMemcpyAsync(s.codes.get(), codes.data(),
                         codes.size() * sizeof(std::int32_t),
                         hipMemcpyHostToDevice, s.stream),
          "upload codes");
    kernels::CodebookSum(s.tables.get(), s.codes.get(), s.quantized.get(),
                         codec.codebooks, codec.codebook_size, frames, hidden,
                         s.stream);
    // fc2 into the input convolution's padded, channel-major buffer.
    const std::size_t pad = s.input_conv.Pad();
    const std::size_t stride = frames + 2 * pad;
    Check(hipMemsetAsync(s.padded.get(), 0,
                         acoustic.latent_channels * stride * sizeof(float),
                         s.stream),
          "clear padding");
    const float alpha = 1.0F;
    const float beta = 0.0F;
    Check(hipblasSgemm(s.blas, HIPBLAS_OP_T, HIPBLAS_OP_N,
                       static_cast<int>(frames),
                       static_cast<int>(acoustic.latent_channels),
                       static_cast<int>(hidden), &alpha, s.quantized.get(),
                       static_cast<int>(hidden), s.fc2_weight.get(),
                       static_cast<int>(hidden), &beta, s.padded.get() + pad,
                       static_cast<int>(stride)),
          "fc2");
    kernels::AddChannelBias(s.padded.get() + pad, stride, s.fc2_bias.get(),
                            acoustic.latent_channels, frames, s.stream);
    s.Apply(s.input_conv, s.padded.get(), frames, s.a.get());
    std::size_t length = frames;
    float* current = s.a.get();
    float* other = s.b.get();
    for (const DecoderBlock& block : s.blocks) {
      length = s.Block(block, current, length, other);
      std::swap(current, other);
    }
    const std::size_t output_pad = s.output_conv.Pad();
    kernels::Snake(current, length, s.final_alpha.get(), s.padded.get(),
                   s.output_conv.in, length, output_pad, s.stream);
    s.Apply(s.output_conv, s.padded.get(), length, s.y.get());
    audio->resize(length);
    Check(hipMemcpyAsync(audio->data(), s.y.get(), length * sizeof(float),
                         hipMemcpyDeviceToHost, s.stream),
          "download audio");
    Check(hipStreamSynchronize(s.stream), "decode");
  } catch (const std::exception& failure) {
    SetError(error, std::string("OmniVoice decode: ") + failure.what());
    return false;
  }
  return true;
}

}  // namespace gufo::models::omnivoice
