#ifndef GUFO_MODELS_OMNIVOICE_CODEC_STATE_HPP_
#define GUFO_MODELS_OMNIVOICE_CODEC_STATE_HPP_

// Private to codec.cpp and codec_encoder.cpp: the audio tokenizer's device
// weights and work buffers.
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <string>
#include <string_view>
#include <vector>

#include "codec.hpp"
#include "device.hpp"
#include "kernels/codec_kernels.hpp"
#include "kernels/encoder_kernels.hpp"
#include "safetensors.hpp"

namespace gufo::models::omnivoice {

inline constexpr std::size_t kResidualKernel = 7;
inline constexpr std::array<std::size_t, 3> kDilations = {1, 3, 9};

/// Conv1d with weights rearranged to [tap][out][in].
struct Conv {
  DeviceBuffer<float> weight;
  DeviceBuffer<float> bias;
  std::size_t in = 0;
  std::size_t out = 0;
  std::size_t kernel = 0;
  std::size_t dilation = 1;

  [[nodiscard]] std::size_t Pad() const { return (kernel - 1) * dilation / 2; }
};

/// ConvTranspose1d with weights rearranged to [tap * out][in].
struct ConvTranspose {
  DeviceBuffer<float> weight;
  DeviceBuffer<float> bias;
  std::size_t in = 0;
  std::size_t out = 0;
  std::size_t stride = 0;
};

struct ResidualUnit {
  DeviceBuffer<float> alpha1;
  DeviceBuffer<float> alpha2;
  Conv conv1;
  Conv conv2;
};

struct DecoderBlock {
  DeviceBuffer<float> alpha;
  ConvTranspose upsample;
  std::array<ResidualUnit, 3> units;
};

/// nn.Linear: weight [out][in].
struct Dense {
  DeviceBuffer<float> weight;
  DeviceBuffer<float> bias;
  std::size_t in = 0;
  std::size_t out = 0;
};

/// Conv1d as im2col plus one GEMM; weight [out][in * kernel] as PyTorch
/// stores it, bias empty when the layer has none.
struct StridedConv {
  DeviceBuffer<float> weight;
  DeviceBuffer<float> bias;
  std::size_t in = 0;
  std::size_t out = 0;
  std::size_t kernel = 1;
  std::size_t stride = 1;
  std::size_t dilation = 1;
  std::size_t pad = 0;

  [[nodiscard]] std::size_t OutputLength(std::size_t length) const {
    return (length + 2 * pad - dilation * (kernel - 1) - 1) / stride + 1;
  }
};

/// A post-norm HuBERT encoder layer; q, k and v share one projection.
struct HubertLayer {
  Dense qkv;
  Dense output;
  DeviceBuffer<float> norm_weight;
  DeviceBuffer<float> norm_bias;
  Dense intermediate;
  Dense final_dense;
  DeviceBuffer<float> final_norm_weight;
  DeviceBuffer<float> final_norm_bias;
};

struct SemanticUnit {
  StridedConv conv1;
  StridedConv conv2;
};

struct SemanticBlock {
  std::vector<SemanticUnit> units;
  StridedConv conv;
};

struct AcousticBlock {
  std::array<ResidualUnit, 3> units;
  DeviceBuffer<float> alpha;
  StridedConv downsample;
};

struct Encoder {
  // HuBERT: feature extractor, projection, positional convolution
  // ([tap][group][out][in], weight norm folded) and layers.
  std::vector<StridedConv> features;
  DeviceBuffer<float> group_norm_weight;
  DeviceBuffer<float> group_norm_bias;
  DeviceBuffer<float> projection_norm_weight;
  DeviceBuffer<float> projection_norm_bias;
  Dense projection;
  DeviceBuffer<float> position_weight;
  DeviceBuffer<float> position_bias;
  DeviceBuffer<float> encoder_norm_weight;
  DeviceBuffer<float> encoder_norm_bias;
  std::vector<HubertLayer> layers;
  // Semantic encoder.
  StridedConv semantic_input;
  std::vector<SemanticBlock> semantic_blocks;
  // DAC encoder.
  StridedConv acoustic_input;
  std::vector<AcousticBlock> acoustic_blocks;
  DeviceBuffer<float> acoustic_alpha;
  StridedConv acoustic_output;
  Dense fc;
  // RVQ: project_in, the codebooks and their squared norms.
  std::vector<Dense> project_in;
  std::vector<DeviceBuffer<float>> codebooks;
  std::vector<DeviceBuffer<float>> codebook_norms;
};

const Tensor& Require(const SafetensorsFile& file, std::string_view name,
                      std::initializer_list<std::size_t> shape);
DeviceBuffer<float> Upload(const float* data, std::size_t count);
DeviceBuffer<float> UploadTensor(const SafetensorsFile& file,
                                 std::string_view name,
                                 std::initializer_list<std::size_t> shape);
Conv LoadConv(const SafetensorsFile& file, const std::string& prefix,
              std::size_t in, std::size_t out, std::size_t kernel,
              std::size_t dilation);
ConvTranspose LoadConvTranspose(const SafetensorsFile& file,
                                const std::string& prefix, std::size_t in,
                                std::size_t out, std::size_t stride);
/// Loads the encoder's weights (codec_encoder.cpp).
void LoadEncoder(const Config& config, const SafetensorsFile& file,
                 hipblasHandle_t blas, hipStream_t stream, Encoder* encoder);

struct Codec::State {
  Config config;
  hipStream_t stream = nullptr;
  hipblasHandle_t blas = nullptr;
  DeviceBuffer<float> tables;  // [codebooks][entries][hidden]
  DeviceBuffer<float> fc2_weight;
  DeviceBuffer<float> fc2_bias;
  Conv input_conv;
  std::vector<DecoderBlock> blocks;
  DeviceBuffer<float> final_alpha;
  Conv output_conv;
  // Work buffers.
  DeviceBuffer<std::int32_t> codes;
  DeviceBuffer<float> quantized;
  // a and b hold the signal between blocks; y, padded and columns are
  // scratch of the convolutions.
  DeviceBuffer<float> a, b, y, padded, columns;

  Encoder encoder;
  DeviceBuffer<float> encoder_columns;

  /// output[out][t] = Linear-style Conv1d over a channel-major input.
  /// Returns the output length.
  std::size_t Convolve(const StridedConv& conv, const float* input,
                       std::size_t length, float* output) {
    const std::size_t output_length = conv.OutputLength(length);
    const std::size_t taps = conv.in * conv.kernel;
    const float* columns = input;
    if (conv.kernel != 1 || conv.stride != 1 || conv.pad != 0) {
      encoder_columns.Reserve(taps * output_length);
      kernels::Im2Col(input, conv.in, length, conv.kernel, conv.stride,
                      conv.dilation, conv.pad, output_length,
                      encoder_columns.get(), stream);
      columns = encoder_columns.get();
    }
    const float alpha = 1.0F;
    const float beta = 0.0F;
    Check(hipblasSgemm(blas, HIPBLAS_OP_N, HIPBLAS_OP_N,
                       static_cast<int>(output_length),
                       static_cast<int>(conv.out), static_cast<int>(taps),
                       &alpha, columns, static_cast<int>(output_length),
                       conv.weight.get(), static_cast<int>(taps), &beta, output,
                       static_cast<int>(output_length)),
          "strided conv");
    if (conv.bias.size() > 0)
      kernels::AddChannelBias(output, output_length, conv.bias.get(), conv.out,
                              output_length, stream);
    return output_length;
  }

  /// y[rows, out] (+)= x[rows, in] * W^T + bias.
  void Linear(const Dense& dense, const float* x, std::size_t rows, float* y,
              bool accumulate) const {
    const float alpha = 1.0F;
    const float beta = accumulate ? 1.0F : 0.0F;
    Check(
        hipblasSgemm(blas, HIPBLAS_OP_T, HIPBLAS_OP_N,
                     static_cast<int>(dense.out), static_cast<int>(rows),
                     static_cast<int>(dense.in), &alpha, dense.weight.get(),
                     static_cast<int>(dense.in), x, static_cast<int>(dense.in),
                     &beta, y, static_cast<int>(dense.out)),
        "linear");
    kernels::AddColumnBias(y, dense.bias.get(), rows, dense.out, stream);
  }

  ~State() {
    if (blas != nullptr)
      (void)hipblasDestroy(blas);
    if (stream != nullptr)
      (void)hipStreamDestroy(stream);
  }

  /// output[out][t] = conv(input) for an input padded by conv.Pad() on both
  /// sides (row stride length + 2 * pad).
  void Apply(const Conv& conv, const float* input, std::size_t length,
             float* output) const {
    const std::size_t stride = length + 2 * conv.Pad();
    for (std::size_t k = 0; k < conv.kernel; ++k) {
      const float alpha = 1.0F;
      const float beta = k == 0 ? 0.0F : 1.0F;
      Check(hipblasSgemm(blas, HIPBLAS_OP_N, HIPBLAS_OP_N,
                         static_cast<int>(length), static_cast<int>(conv.out),
                         static_cast<int>(conv.in), &alpha,
                         input + k * conv.dilation, static_cast<int>(stride),
                         conv.weight.get() + k * conv.out * conv.in,
                         static_cast<int>(conv.in), &beta, output,
                         static_cast<int>(length)),
            "conv1d");
    }
    kernels::AddChannelBias(output, length, conv.bias.get(), conv.out, length,
                            stream);
  }

  /// x[c][t] += unit(x) over `channels` x `length`.
  void Residual(const ResidualUnit& unit, float* signal, std::size_t channels,
                std::size_t length) {
    const std::size_t pad = unit.conv1.Pad();
    kernels::Snake(signal, length, unit.alpha1.get(), padded.get(), channels,
                   length, pad, stream);
    Apply(unit.conv1, padded.get(), length, y.get());
    kernels::Snake(y.get(), length, unit.alpha2.get(), padded.get(), channels,
                   length, 0, stream);
    const float alpha = 1.0F;
    const float beta = 1.0F;
    Check(
        hipblasSgemm(blas, HIPBLAS_OP_N, HIPBLAS_OP_N, static_cast<int>(length),
                     static_cast<int>(channels), static_cast<int>(channels),
                     &alpha, padded.get(), static_cast<int>(length),
                     unit.conv2.weight.get(), static_cast<int>(channels), &beta,
                     signal, static_cast<int>(length)),
        "residual conv");
    kernels::AddChannelBias(signal, length, unit.conv2.bias.get(), channels,
                            length, stream);
  }

  /// One decoder block: snake, transposed convolution into `output` and the
  /// residual units; returns the new length.
  std::size_t Block(const DecoderBlock& block, const float* input,
                    std::size_t length, float* output) {
    const ConvTranspose& up = block.upsample;
    const std::size_t kernel = 2 * up.stride;
    kernels::Snake(input, length, block.alpha.get(), padded.get(), up.in,
                   length, 0, stream);
    const float alpha = 1.0F;
    const float beta = 0.0F;
    Check(
        hipblasSgemm(blas, HIPBLAS_OP_N, HIPBLAS_OP_N, static_cast<int>(length),
                     static_cast<int>(kernel * up.out), static_cast<int>(up.in),
                     &alpha, padded.get(), static_cast<int>(length),
                     up.weight.get(), static_cast<int>(up.in), &beta,
                     columns.get(), static_cast<int>(length)),
        "transposed conv");
    // padding = ceil(stride / 2); output_padding = stride % 2 keeps the
    // length at length * stride.
    kernels::Col2Im(columns.get(), up.bias.get(), output, up.out, length,
                    up.stride, kernel, (up.stride + 1) / 2, stream);
    const std::size_t upsampled = length * up.stride;
    for (const ResidualUnit& unit : block.units)
      Residual(unit, output, up.out, upsampled);
    return upsampled;
  }

  void Reserve(std::size_t frames) {
    const AcousticConfig& acoustic = config.codec.acoustic;
    std::size_t largest = 0, largest_columns = 0, largest_padded = 0;
    std::size_t length = frames;
    std::size_t channels = acoustic.decoder_channels;
    largest = channels * length;
    largest_padded =
        std::max(acoustic.latent_channels, channels) * (length + 6);
    for (const std::size_t stride : acoustic.upsampling_ratios) {
      const std::size_t out = channels / 2;
      largest_columns = std::max(largest_columns, 2 * stride * out * length);
      largest_padded = std::max(largest_padded, channels * length);
      length *= stride;
      channels = out;
      largest = std::max(largest, channels * length);
      largest_padded = std::max(largest_padded, channels * (length + 2 * 27));
    }
    codes.Reserve(config.codec.codebooks * frames);
    quantized.Reserve(frames * (config.codec.semantic.hidden_size +
                                acoustic.latent_channels));
    a.Reserve(largest);
    b.Reserve(largest);
    y.Reserve(largest);
    padded.Reserve(largest_padded);
    columns.Reserve(largest_columns);
  }
};

}  // namespace gufo::models::omnivoice

#endif  // GUFO_MODELS_OMNIVOICE_CODEC_STATE_HPP_
