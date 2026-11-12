// The Higgs Audio v2 tokenizer's encoder, ported from Hugging Face
// transformers' HiggsAudioV2TokenizerModel.encode, HuBERT and DAC encoder
// (Apache-2.0, see licenses/transformers.txt; Copyright 2025 Boson AI and The
// HuggingFace Team, Copyright 2021 The Fairseq Authors and the HuggingFace
// Inc. team, Copyright 2024 Descript and The HuggingFace Inc. team): HuBERT
// features at 16 kHz averaged over its hidden states, the semantic encoder,
// the DAC encoder, fc and the residual vector quantizer.
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>

#include <cmath>
#include <numeric>
#include <utility>

#include "audio.hpp"
#include "codec.hpp"
#include "codec_state.hpp"
#include "kernels/encoder_kernels.hpp"
#include "kernels/generator_kernels.hpp"

namespace gufo::models::omnivoice {
namespace {

Dense LoadDense(const SafetensorsFile& file, const std::string& prefix,
                std::size_t in, std::size_t out) {
  Dense dense;
  dense.weight = UploadTensor(file, prefix + ".weight", {out, in});
  dense.bias = UploadTensor(file, prefix + ".bias", {out});
  dense.in = in;
  dense.out = out;
  return dense;
}

/// q, k and v projections stacked into one.
Dense LoadQkv(const SafetensorsFile& file, const std::string& prefix,
              std::size_t width) {
  std::vector<float> weight, bias;
  for (const char* name : {"q_proj", "k_proj", "v_proj"}) {
    const Tensor& w = Require(file, prefix + name + ".weight", {width, width});
    const Tensor& b = Require(file, prefix + name + ".bias", {width});
    const auto* wd = reinterpret_cast<const float*>(w.data);
    const auto* bd = reinterpret_cast<const float*>(b.data);
    weight.insert(weight.end(), wd, wd + w.Elements());
    bias.insert(bias.end(), bd, bd + b.Elements());
  }
  Dense dense;
  dense.weight = Upload(weight.data(), weight.size());
  dense.bias = Upload(bias.data(), bias.size());
  dense.in = width;
  dense.out = 3 * width;
  return dense;
}

StridedConv LoadStridedConv(const SafetensorsFile& file,
                            const std::string& prefix, std::size_t in,
                            std::size_t out, std::size_t kernel,
                            std::size_t stride, std::size_t dilation,
                            std::size_t pad, bool bias) {
  StridedConv conv;
  conv.weight = UploadTensor(file, prefix + ".weight", {out, in, kernel});
  if (bias)
    conv.bias = UploadTensor(file, prefix + ".bias", {out});
  conv.in = in;
  conv.out = out;
  conv.kernel = kernel;
  conv.stride = stride;
  conv.dilation = dilation;
  conv.pad = pad;
  return conv;
}

}  // namespace

void LoadEncoder(const Config& config, const SafetensorsFile& file,
                 hipblasHandle_t blas, hipStream_t stream, Encoder* encoder) {
  const CodecConfig& codec = config.codec;
  const SemanticEncoderConfig& semantic = codec.semantic;
  const AcousticConfig& acoustic = codec.acoustic;
  const std::size_t width = semantic.hidden_size;
  // HuBERT feature extractor: bias-free convolutions, the first one with a
  // per-channel group norm.
  std::size_t in = 1;
  for (std::size_t i = 0; i < semantic.conv_channels.size(); ++i) {
    const std::string prefix =
        "semantic_model.feature_extractor.conv_layers." + std::to_string(i);
    encoder->features.push_back(LoadStridedConv(
        file, prefix + ".conv", in, semantic.conv_channels[i],
        semantic.conv_kernels[i], semantic.conv_strides[i], 1, 0, false));
    in = semantic.conv_channels[i];
  }
  const std::size_t features = in;
  encoder->group_norm_weight = UploadTensor(
      file, "semantic_model.feature_extractor.conv_layers.0.layer_norm.weight",
      {semantic.conv_channels[0]});
  encoder->group_norm_bias = UploadTensor(
      file, "semantic_model.feature_extractor.conv_layers.0.layer_norm.bias",
      {semantic.conv_channels[0]});
  encoder->projection_norm_weight = UploadTensor(
      file, "semantic_model.feature_projection.layer_norm.weight", {features});
  encoder->projection_norm_bias = UploadTensor(
      file, "semantic_model.feature_projection.layer_norm.bias", {features});
  encoder->projection = LoadDense(
      file, "semantic_model.feature_projection.projection", features, width);
  // Positional convolution: fold the weight norm (dim=2, one gain per tap)
  // and rearrange to [tap][group][out][in].
  {
    const std::size_t kernel = semantic.position_kernel;
    const std::size_t groups = semantic.position_groups;
    const std::size_t group_width = width / groups;
    const std::string prefix = "semantic_model.encoder.pos_conv_embed.conv.";
    const Tensor& gain = Require(
        file, prefix + "parametrizations.weight.original0", {1, 1, kernel});
    const Tensor& direction =
        Require(file, prefix + "parametrizations.weight.original1",
                {width, group_width, kernel});
    const auto* g = reinterpret_cast<const float*>(gain.data);
    const auto* v = reinterpret_cast<const float*>(direction.data);
    std::vector<double> norms(kernel, 0.0);
    for (std::size_t o = 0; o < width; ++o)
      for (std::size_t i = 0; i < group_width; ++i)
        for (std::size_t k = 0; k < kernel; ++k) {
          const double value = v[(o * group_width + i) * kernel + k];
          norms[k] += value * value;
        }
    std::vector<float> weight(kernel * width * group_width);
    for (std::size_t o = 0; o < width; ++o) {
      const std::size_t group = o / group_width;
      const std::size_t local = o % group_width;
      for (std::size_t i = 0; i < group_width; ++i)
        for (std::size_t k = 0; k < kernel; ++k) {
          const float folded =
              static_cast<float>(g[k] * v[(o * group_width + i) * kernel + k] /
                                 std::sqrt(norms[k]));
          weight[((k * groups + group) * group_width + local) * group_width +
                 i] = folded;
        }
    }
    encoder->position_weight = Upload(weight.data(), weight.size());
    encoder->position_bias = UploadTensor(file, prefix + "bias", {width});
  }
  encoder->encoder_norm_weight =
      UploadTensor(file, "semantic_model.encoder.layer_norm.weight", {width});
  encoder->encoder_norm_bias =
      UploadTensor(file, "semantic_model.encoder.layer_norm.bias", {width});
  for (std::size_t l = 0; l < semantic.layers; ++l) {
    const std::string prefix =
        "semantic_model.encoder.layers." + std::to_string(l) + ".";
    HubertLayer layer;
    layer.qkv = LoadQkv(file, prefix + "attention.", width);
    layer.output = LoadDense(file, prefix + "attention.out_proj", width, width);
    layer.norm_weight =
        UploadTensor(file, prefix + "layer_norm.weight", {width});
    layer.norm_bias = UploadTensor(file, prefix + "layer_norm.bias", {width});
    layer.intermediate =
        LoadDense(file, prefix + "feed_forward.intermediate_dense", width,
                  semantic.intermediate_size);
    layer.final_dense = LoadDense(file, prefix + "feed_forward.output_dense",
                                  semantic.intermediate_size, width);
    layer.final_norm_weight =
        UploadTensor(file, prefix + "final_layer_norm.weight", {width});
    layer.final_norm_bias =
        UploadTensor(file, prefix + "final_layer_norm.bias", {width});
    encoder->layers.push_back(std::move(layer));
  }
  // Semantic encoder: one bias-free k3 convolution, then blocks of ELU
  // residual units and a k3 convolution (the checkpoint's strides are 1).
  encoder->semantic_input = LoadStridedConv(file, "encoder_semantic.conv",
                                            width, width, 3, 1, 1, 1, false);
  for (std::size_t b = 0; b < 2; ++b) {
    const std::string prefix =
        "encoder_semantic.conv_blocks." + std::to_string(b);
    SemanticBlock block;
    for (std::size_t u = 0; u < 2; ++u) {
      const std::string unit = prefix + ".res_units." + std::to_string(u);
      SemanticUnit residual;
      residual.conv1 = LoadStridedConv(file, unit + ".conv1", width, width, 3,
                                       1, 1, 1, false);
      residual.conv2 = LoadStridedConv(file, unit + ".conv2", width, width, 1,
                                       1, 1, 0, false);
      block.units.push_back(std::move(residual));
    }
    block.conv =
        LoadStridedConv(file, prefix + ".conv", width, width, 3, 1, 1, 1, true);
    encoder->semantic_blocks.push_back(std::move(block));
  }
  // DAC encoder.
  encoder->acoustic_input = LoadStridedConv(file, "acoustic_encoder.conv1", 1,
                                            acoustic.encoder_channels,
                                            kResidualKernel, 1, 1, 3, true);
  std::size_t channels = acoustic.encoder_channels;
  for (std::size_t index = 0; index < acoustic.downsampling_ratios.size();
       ++index) {
    const std::string prefix =
        "acoustic_encoder.block." + std::to_string(index);
    const std::size_t stride = acoustic.downsampling_ratios[index];
    AcousticBlock block;
    for (std::size_t unit = 0; unit < block.units.size(); ++unit) {
      const std::string name = prefix + ".res_unit" + std::to_string(unit + 1);
      ResidualUnit& residual = block.units[unit];
      residual.alpha1 =
          UploadTensor(file, name + ".snake1.alpha", {1, channels, 1});
      residual.alpha2 =
          UploadTensor(file, name + ".snake2.alpha", {1, channels, 1});
      residual.conv1 = LoadConv(file, name + ".conv1", channels, channels,
                                kResidualKernel, kDilations[unit]);
      residual.conv2 =
          LoadConv(file, name + ".conv2", channels, channels, 1, 1);
    }
    block.alpha =
        UploadTensor(file, prefix + ".snake1.alpha", {1, channels, 1});
    block.downsample =
        LoadStridedConv(file, prefix + ".conv1", channels, 2 * channels,
                        2 * stride, stride, 1, (stride + 1) / 2, true);
    encoder->acoustic_blocks.push_back(std::move(block));
    channels *= 2;
  }
  encoder->acoustic_alpha =
      UploadTensor(file, "acoustic_encoder.snake1.alpha", {1, channels, 1});
  encoder->acoustic_output =
      LoadStridedConv(file, "acoustic_encoder.conv2", channels,
                      acoustic.latent_channels, 3, 1, 1, 1, true);
  const std::size_t hidden = width + acoustic.latent_channels;
  encoder->fc = LoadDense(file, "fc", hidden, hidden);
  for (std::size_t c = 0; c < codec.codebooks; ++c) {
    const std::string prefix = "quantizer.quantizers." + std::to_string(c);
    encoder->project_in.push_back(
        LoadDense(file, prefix + ".project_in", hidden, codec.codebook_dim));
    encoder->codebooks.push_back(
        UploadTensor(file, prefix + ".codebook.embed",
                     {codec.codebook_size, codec.codebook_dim}));
    DeviceBuffer<float> norms(codec.codebook_size);
    kernels::SquaredNorms(encoder->codebooks.back().get(), norms.get(),
                          codec.codebook_size, codec.codebook_dim, stream);
    encoder->codebook_norms.push_back(std::move(norms));
  }
  (void)blas;
  Check(hipStreamSynchronize(stream), "load encoder");
}

namespace {

void SetError(std::string* error, const std::string& message) {
  if (error != nullptr)
    *error = message;
}

}  // namespace

bool Codec::Encode(const std::vector<float>& samples,
                   std::vector<std::int32_t>* codes, std::size_t* frames,
                   std::string* error) {
  try {
    State& s = *state_;
    Encoder& e = s.encoder;
    const CodecConfig& codec = s.config.codec;
    const SemanticEncoderConfig& semantic = codec.semantic;
    const AcousticConfig& acoustic = codec.acoustic;
    const std::size_t width = semantic.hidden_size;
    const std::size_t hidden = width + acoustic.latent_channels;
    if (samples.empty() || samples.size() % codec.frame_samples != 0)
      throw GpuError("reference audio must be a whole number of frames");
    const std::size_t length = samples.size();
    const std::size_t count = length / codec.frame_samples;
    hipStream_t stream = s.stream;

    // ---- Semantic features: HuBERT at 16 kHz with 160 samples of padding.
    std::vector<float> resampled =
        Resample(samples, codec.sample_rate, semantic.sample_rate);
    resampled.insert(resampled.begin(), 160, 0.0F);
    resampled.insert(resampled.end(), 160, 0.0F);
    std::size_t feature_length = resampled.size();
    std::size_t largest = feature_length;
    for (const StridedConv& conv : e.features) {
      feature_length = conv.OutputLength(feature_length);
      largest = std::max(largest, conv.out * feature_length);
    }
    const std::size_t steps = feature_length;  // HuBERT frames
    DeviceBuffer<float> a(
        std::max(largest, length * acoustic.encoder_channels));
    DeviceBuffer<float> b(a.size());
    Check(hipMemcpyAsync(a.get(), resampled.data(),
                         resampled.size() * sizeof(float),
                         hipMemcpyHostToDevice, stream),
          "upload audio");
    std::size_t current = resampled.size();
    for (std::size_t i = 0; i < e.features.size(); ++i) {
      current = s.Convolve(e.features[i], a.get(), current, b.get());
      if (i == 0)
        kernels::GroupNormGelu(b.get(), e.group_norm_weight.get(),
                               e.group_norm_bias.get(), e.features[0].out,
                               current, 1e-5F, stream);
      else
        kernels::Gelu(b.get(), e.features[i].out * current, stream);
      std::swap(a, b);
    }
    const std::size_t feature_channels = e.features.back().out;
    // [channels][steps] -> [steps][channels], LayerNorm, projection.
    DeviceBuffer<float> rows(
        steps * std::max(feature_channels, semantic.intermediate_size));
    kernels::Transpose(a.get(), feature_channels, steps, rows.get(),
                       feature_channels, stream);
    kernels::LayerNorm(rows.get(), e.projection_norm_weight.get(),
                       e.projection_norm_bias.get(), rows.get(), steps,
                       feature_channels, semantic.layer_norm_eps, stream);
    DeviceBuffer<float> state_rows(steps * width);
    s.Linear(e.projection, rows.get(), steps, state_rows.get(), false);
    // Positional convolution over the channel-major, zero-padded states.
    const std::size_t kernel = semantic.position_kernel;
    const std::size_t pad = kernel / 2;
    const std::size_t padded_steps = steps + 2 * pad;
    const std::size_t groups = semantic.position_groups;
    const std::size_t group_width = width / groups;
    DeviceBuffer<float> padded(width * padded_steps);
    Check(
        hipMemsetAsync(padded.get(), 0, padded.size() * sizeof(float), stream),
        "clear padding");
    kernels::Transpose(state_rows.get(), steps, width, padded.get() + pad,
                       padded_steps, stream);
    DeviceBuffer<float> position(width * steps);
    for (std::size_t k = 0; k < kernel; ++k) {
      const float alpha = 1.0F;
      const float beta = k == 0 ? 0.0F : 1.0F;
      Check(
          hipblasSgemmStridedBatched(
              s.blas, HIPBLAS_OP_N, HIPBLAS_OP_N, static_cast<int>(steps),
              static_cast<int>(group_width), static_cast<int>(group_width),
              &alpha, padded.get() + k, static_cast<int>(padded_steps),
              static_cast<long long>(group_width * padded_steps),
              e.position_weight.get() + k * groups * group_width * group_width,
              static_cast<int>(group_width),
              static_cast<long long>(group_width * group_width), &beta,
              position.get(), static_cast<int>(steps),
              static_cast<long long>(group_width * steps),
              static_cast<int>(groups)),
          "positional conv");
    }
    kernels::AddChannelBias(position.get(), steps, e.position_bias.get(), width,
                            steps, stream);
    kernels::Gelu(position.get(), width * steps, stream);
    kernels::Transpose(position.get(), width, steps, rows.get(), width, stream);
    kernels::Accumulate(rows.get(), state_rows.get(), steps * width, stream);
    kernels::LayerNorm(state_rows.get(), e.encoder_norm_weight.get(),
                       e.encoder_norm_bias.get(), state_rows.get(), steps,
                       width, semantic.layer_norm_eps, stream);
    // Layers; the features average the layer input and every layer output.
    DeviceBuffer<float> mean(steps * width);
    Check(hipMemcpyAsync(mean.get(), state_rows.get(),
                         steps * width * sizeof(float), hipMemcpyDeviceToDevice,
                         stream),
          "copy hidden states");
    const std::size_t heads = semantic.heads;
    const std::size_t head_dim = width / heads;
    DeviceBuffer<float> qkv(steps * 3 * width);
    DeviceBuffer<float> scores(heads * steps * steps);
    DeviceBuffer<float> attention(steps * width);
    for (const HubertLayer& layer : e.layers) {
      s.Linear(layer.qkv, state_rows.get(), steps, qkv.get(), false);
      const float alpha = 1.0F;
      const float beta = 0.0F;
      const auto n = static_cast<int>(steps);
      const auto d = static_cast<int>(head_dim);
      const auto stride_qkv = static_cast<int>(3 * width);
      Check(hipblasSgemmStridedBatched(
                s.blas, HIPBLAS_OP_T, HIPBLAS_OP_N, n, n, d, &alpha,
                qkv.get() + width, stride_qkv, static_cast<long long>(head_dim),
                qkv.get(), stride_qkv, static_cast<long long>(head_dim), &beta,
                scores.get(), n, static_cast<long long>(steps * steps),
                static_cast<int>(heads)),
            "HuBERT scores");
      kernels::SoftmaxInPlace(scores.get(), heads * steps, steps,
                              1.0F / std::sqrt(static_cast<float>(head_dim)),
                              stream);
      Check(hipblasSgemmStridedBatched(
                s.blas, HIPBLAS_OP_N, HIPBLAS_OP_N, d, n, n, &alpha,
                qkv.get() + 2 * width, stride_qkv,
                static_cast<long long>(head_dim), scores.get(), n,
                static_cast<long long>(steps * steps), &beta, attention.get(),
                static_cast<int>(width), static_cast<long long>(head_dim),
                static_cast<int>(heads)),
            "HuBERT values");
      s.Linear(layer.output, attention.get(), steps, state_rows.get(), true);
      kernels::LayerNorm(state_rows.get(), layer.norm_weight.get(),
                         layer.norm_bias.get(), state_rows.get(), steps, width,
                         semantic.layer_norm_eps, stream);
      s.Linear(layer.intermediate, state_rows.get(), steps, rows.get(), false);
      kernels::Gelu(rows.get(), steps * semantic.intermediate_size, stream);
      s.Linear(layer.final_dense, rows.get(), steps, state_rows.get(), true);
      kernels::LayerNorm(state_rows.get(), layer.final_norm_weight.get(),
                         layer.final_norm_bias.get(), state_rows.get(), steps,
                         width, semantic.layer_norm_eps, stream);
      kernels::Accumulate(state_rows.get(), mean.get(), steps * width, stream);
    }
    kernels::Scale(mean.get(), steps * width,
                   1.0F / static_cast<float>(e.layers.size() + 1), stream);
    // Every semantic_stride-th frame, channel-major.
    if (steps < count * codec.semantic_stride)
      throw GpuError("reference audio is too short for the semantic encoder");
    std::vector<std::int32_t> picks(count);
    for (std::size_t f = 0; f < count; ++f)
      picks[f] = static_cast<std::int32_t>(f * codec.semantic_stride);
    DeviceBuffer<std::int32_t> pick_rows(count);
    Check(hipMemcpyAsync(pick_rows.get(), picks.data(),
                         count * sizeof(std::int32_t), hipMemcpyHostToDevice,
                         stream),
          "upload frame picks");
    kernels::GatherRows(mean.get(), pick_rows.get(), rows.get(), count, width,
                        stream);
    DeviceBuffer<float> semantic_a(width * count), semantic_b(width * count),
        semantic_c(width * count);
    kernels::Transpose(rows.get(), count, width, semantic_a.get(), count,
                       stream);
    // ---- Semantic encoder.
    s.Convolve(e.semantic_input, semantic_a.get(), count, semantic_b.get());
    for (const SemanticBlock& block : e.semantic_blocks) {
      for (const SemanticUnit& unit : block.units) {
        kernels::Elu(semantic_b.get(), semantic_a.get(), width * count, stream);
        s.Convolve(unit.conv1, semantic_a.get(), count, semantic_c.get());
        kernels::Elu(semantic_c.get(), semantic_c.get(), width * count, stream);
        s.Convolve(unit.conv2, semantic_c.get(), count, semantic_a.get());
        kernels::Accumulate(semantic_a.get(), semantic_b.get(), width * count,
                            stream);
      }
      s.Convolve(block.conv, semantic_b.get(), count, semantic_a.get());
      std::swap(semantic_a, semantic_b);
    }
    // ---- DAC encoder into the first latent rows of the joint embedding.
    DeviceBuffer<float> joint(hidden * count);
    Check(hipMemcpyAsync(joint.get() + acoustic.latent_channels * count,
                         semantic_b.get(), width * count * sizeof(float),
                         hipMemcpyDeviceToDevice, stream),
          "join semantic");
    Check(hipMemcpyAsync(b.get(), samples.data(), length * sizeof(float),
                         hipMemcpyHostToDevice, stream),
          "upload audio");
    current = s.Convolve(e.acoustic_input, b.get(), length, a.get());
    std::size_t channels = acoustic.encoder_channels;
    s.y.Reserve(channels * current);
    s.padded.Reserve(channels * (current + 2 * 27));
    for (const AcousticBlock& block : e.acoustic_blocks) {
      for (const ResidualUnit& unit : block.units)
        s.Residual(unit, a.get(), channels, current);
      kernels::Snake(a.get(), current, block.alpha.get(), s.padded.get(),
                     channels, current, 0, stream);
      current = s.Convolve(block.downsample, s.padded.get(), current, b.get());
      channels *= 2;
      std::swap(a, b);
    }
    kernels::Snake(a.get(), current, e.acoustic_alpha.get(), s.padded.get(),
                   channels, current, 0, stream);
    if (current != count)
      throw GpuError("acoustic and semantic frame counts differ");
    s.Convolve(e.acoustic_output, s.padded.get(), current, joint.get());
    // ---- fc and the residual vector quantizer.
    DeviceBuffer<float> residual(count * hidden);
    kernels::Transpose(joint.get(), hidden, count, rows.get(), hidden, stream);
    s.Linear(e.fc, rows.get(), count, residual.get(), false);
    DeviceBuffer<float> projected(count * codec.codebook_dim);
    DeviceBuffer<float> dots(count * codec.codebook_size);
    DeviceBuffer<std::int32_t> indices(codec.codebooks * count);
    for (std::size_t c = 0; c < codec.codebooks; ++c) {
      s.Linear(e.project_in[c], residual.get(), count, projected.get(), false);
      const float alpha = 1.0F;
      const float beta = 0.0F;
      Check(hipblasSgemm(
                s.blas, HIPBLAS_OP_T, HIPBLAS_OP_N,
                static_cast<int>(codec.codebook_size), static_cast<int>(count),
                static_cast<int>(codec.codebook_dim), &alpha,
                e.codebooks[c].get(), static_cast<int>(codec.codebook_dim),
                projected.get(), static_cast<int>(codec.codebook_dim), &beta,
                dots.get(), static_cast<int>(codec.codebook_size)),
            "codebook distances");
      kernels::NearestEntry(dots.get(), projected.get(),
                            e.codebook_norms[c].get(),
                            indices.get() + c * count, count,
                            codec.codebook_size, codec.codebook_dim, stream);
      kernels::SubtractEntries(
          residual.get(), s.tables.get() + c * codec.codebook_size * hidden,
          indices.get() + c * count, count, hidden, stream);
    }
    codes->resize(codec.codebooks * count);
    Check(hipMemcpyAsync(codes->data(), indices.get(),
                         codes->size() * sizeof(std::int32_t),
                         hipMemcpyDeviceToHost, stream),
          "download codes");
    Check(hipStreamSynchronize(stream), "encode");
    *frames = count;
  } catch (const std::exception& failure) {
    SetError(error, std::string("OmniVoice encode: ") + failure.what());
    return false;
  }
  return true;
}

}  // namespace gufo::models::omnivoice
