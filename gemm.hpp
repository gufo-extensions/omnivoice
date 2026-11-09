#ifndef GUFO_MODELS_OMNIVOICE_GEMM_HPP_
#define GUFO_MODELS_OMNIVOICE_GEMM_HPP_

#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <tuple>

namespace gufo::models::omnivoice {

/// A column-major (BLAS) product C[m][n] (+)= op(A)[m][k] * op(B)[k][n],
/// optionally strided-batched.
struct MatmulShape {
  bool transpose_a = false;
  bool transpose_b = false;
  std::size_t m = 0;
  std::size_t n = 0;
  std::size_t k = 0;
  std::size_t lda = 0;
  std::size_t ldb = 0;
  std::size_t ldc = 0;
  std::int64_t stride_a = 0;
  std::int64_t stride_b = 0;
  std::int64_t stride_c = 0;
  std::size_t batch = 1;
  bool accumulate = false;

  [[nodiscard]] auto Tie() const {
    return std::tie(transpose_a, transpose_b, m, n, k, lda, ldb, ldc, stride_a,
                    stride_b, stride_c, batch, accumulate);
  }
  bool operator<(const MatmulShape& other) const { return Tie() < other.Tie(); }
};

/// float16 products with float32 accumulation and output on hipBLASLt. On
/// gfx1151 hipBLASLt runs them on the matrix cores, where rocBLAS's
/// mixed-precision GEMM does not, and its ranking of algorithms is a poor
/// predictor, so candidates are timed once per representative shape. Shapes
/// hipBLASLt can't serve fall back to hipBLAS.
class HalfGemm {
public:
  HalfGemm(hipblasHandle_t fallback, hipStream_t stream);
  ~HalfGemm();
  HalfGemm(const HalfGemm&) = delete;
  HalfGemm& operator=(const HalfGemm&) = delete;

  /// y[rows][out] (+)= x[rows][in] * w[out][in]^T.
  void Linear(const void* x, const void* w, float* y, std::size_t rows,
              std::size_t in, std::size_t out, bool accumulate);

  /// `shape` with the algorithm tuned for `representative`, a shape of the
  /// same structure with sizes rounded up so that tuning is shared.
  void Matmul(const MatmulShape& shape, const MatmulShape& representative,
              const void* a, const void* b, float* c);

private:
  struct State;
  std::unique_ptr<State> state_;
};

/// Rounds up to a multiple of 128, the bucket size of tuned shapes.
[[nodiscard]] constexpr std::size_t TuningBucket(std::size_t size) {
  return (size + 127) / 128 * 128;
}

}  // namespace gufo::models::omnivoice

#endif  // GUFO_MODELS_OMNIVOICE_GEMM_HPP_
