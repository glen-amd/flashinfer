/*
 * Copyright (c) 2024 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef FLASHINFER_GEMM_BMM_FP8_CUH_
#define FLASHINFER_GEMM_BMM_FP8_CUH_

#include "../gpu_defines_cuda_hip.h"

#if defined(__HIPCC__) || (defined(__clang__) && defined(__HIP__)) || defined(__HIPCC_RTC__)
#include <hipblaslt.h>
#include <hip/hip_fp8.h>
#elif defined(__CUDACC__) || defined(__NVCC__) || (defined(__clang__) && defined(__CUDA__)) || defined(__CUDACC_RTC__)
#include <cublasLt.h>
#include <cuda_fp8.h>
#endif

#include <iostream>
#include <memory>
#include <type_traits>

#include "../exception.h"

#if defined(__HIPCC__) || (defined(__clang__) && defined(__HIP__)) || defined(__HIPCC_RTC__)
// library/src/include/auxiliary.hpp in the HIPBLASLt Git repo
static const char* cublasGetStatusString(hipblasStatus_t status) {
#define CASE(x) \
    case x:     \
        return #x
    switch(status) {
        CASE(HIPBLAS_STATUS_SUCCESS);
        CASE(HIPBLAS_STATUS_NOT_INITIALIZED);
        CASE(HIPBLAS_STATUS_ALLOC_FAILED);
        CASE(HIPBLAS_STATUS_INVALID_VALUE);
        CASE(HIPBLAS_STATUS_MAPPING_ERROR);
        CASE(HIPBLAS_STATUS_EXECUTION_FAILED);
        CASE(HIPBLAS_STATUS_INTERNAL_ERROR);
        CASE(HIPBLAS_STATUS_NOT_SUPPORTED);
        CASE(HIPBLAS_STATUS_ARCH_MISMATCH);
        CASE(HIPBLAS_STATUS_INVALID_ENUM);
        CASE(HIPBLAS_STATUS_UNKNOWN);
        CASE(HIPBLAS_STATUS_HANDLE_IS_NULLPTR);
    }
#undef CASE
    // We don't use default: so that the compiler warns us if any valid enums are
    // missing from our switch. If the value is not a valid hipblasStatus_t, we
    // return this string.
    return "<undefined hipblasStatus_t value>";
}
#endif

#define FLASHINFER_CUBLAS_CHECK(EXPR)                                           \
  {                                                                             \
    gpublasStatus_t e = (EXPR);                                                 \
    FLASHINFER_CHECK(e == GPUBLAS_STATUS_SUCCESS,                               \
                     "CUBLAS Error: " + std::string(cublasGetStatusString(e))); \
  }

#ifndef NDEBUG
#define FLASHINFER_CUBLAS_CALL(EXPR)                                                             \
  {                                                                                              \
    gpublasStatus_t e = (EXPR);                                                                  \
    if (e != GPUBLAS_STATUS_SUCCESS) {                                                           \
      std::cerr << "CUBLAS Error: " << cublasGetStatusString(e) << " (" << e << ") " << __FILE__ \
                << ": line " << __LINE__ << " at function " << #EXPR << std::endl;               \
      return e;                                                                                  \
    }                                                                                            \
  }
#else
#define FLASHINFER_CUBLAS_CALL(EXPR)   \
  {                                    \
    gpuError_t e = (EXPR);             \
    if (e != GPUBLAS_STATUS_SUCCESS) { \
      return e;                        \
    }                                  \
  }
#endif

namespace flashinfer {

namespace bmm_fp8 {

template <typename T, gpublasStatus_t (*destructor)(T*)>
struct CuBlasLtDeleter {
  void operator()(T* x) {
    if (x != nullptr) {
      FLASHINFER_CUBLAS_CHECK(destructor(x));
    }
  }
};

template <typename T, gpublasStatus_t (*destructor)(T*)>
class CuBlasLtDescriptor {
 public:
  T* descriptor() const { return descriptor_.get(); }
  T* descriptor() { return descriptor_.get(); }

 protected:
  std::unique_ptr<T, CuBlasLtDeleter<T, destructor>> descriptor_;
};

class CuBlasLtMatmulDescriptor
    : public CuBlasLtDescriptor<gpublasLtMatmulDescOpaque_t, &gpublasLtMatmulDescDestroy> {
 public:
  CuBlasLtMatmulDescriptor(gpublasComputeType_t compute_type, gpuDataType_t scale_type) {
    gpublasLtMatmulDesc_t raw_descriptor = nullptr;
    FLASHINFER_CUBLAS_CHECK(gpublasLtMatmulDescCreate(&raw_descriptor, compute_type, scale_type));
    descriptor_.reset(raw_descriptor);
  }
  template <typename T>
  inline void setAttribute(gpublasLtMatmulDescAttributes_t attr, const T value) {
    FLASHINFER_CUBLAS_CHECK(
        ::gpublasLtMatmulDescSetAttribute(descriptor(), attr, &value, sizeof(T)));
  }
};

class CuBlasLtMatrixLayout
    : public CuBlasLtDescriptor<gpublasLtMatrixLayoutOpaque_t, &gpublasLtMatrixLayoutDestroy> {
 public:
  CuBlasLtMatrixLayout(cudaDataType_t type, uint64_t rows, uint64_t cols, int64_t ld,
                       bool t = false) {
    gpublasLtMatrixLayout_t raw_descriptor = nullptr;
    FLASHINFER_CUBLAS_CHECK(
        gpublasLtMatrixLayoutCreate(&raw_descriptor, type, t ? cols : rows, t ? rows : cols, ld));
    descriptor_.reset(raw_descriptor);
  }
  template <typename T>
  inline void setAttribute(gpublasLtMatrixLayoutAttribute_t attr, const T value) {
    FLASHINFER_CUBLAS_CHECK(
        ::gpublasLtMatrixLayoutSetAttribute(descriptor(), attr, &value, sizeof(T)));
  }
};

class CuBlasLtMatmulPreference : public CuBlasLtDescriptor<gpublasLtMatmulPreferenceOpaque_t,
                                                           &gpublasLtMatmulPreferenceDestroy> {
 public:
  CuBlasLtMatmulPreference() {
    gpublasLtMatmulPreference_t raw_descriptor = nullptr;
    FLASHINFER_CUBLAS_CHECK(gpublasLtMatmulPreferenceCreate(&raw_descriptor));
    descriptor_.reset(raw_descriptor);
  }
  template <typename T>
  inline void setAttribute(gpublasLtMatmulPreferenceAttributes_t attr, const T value) {
    FLASHINFER_CUBLAS_CHECK(
        ::gpublasLtMatmulPreferenceSetAttribute(descriptor(), attr, &value, sizeof(T)));
  }
};

template <typename T>
gpuDataType_t get_cuda_data_type() {
  if constexpr (std::is_same_v<T, __gpu_fp8_e4m3>) {
    return GPU_R_8F_E4M3;
  } else if constexpr (std::is_same_v<T, __gpu_fp8_e5m2>) {
    return GPU_R_8F_E5M2;
  } else if constexpr (std::is_same_v<T, __gpu_bfloat16>) {
    return GPU_R_16BF;
  } else if constexpr (std::is_same_v<T, half>) {
    return GPU_R_16F;
  } else {
    FLASHINFER_ERROR("Unsupported type");
  }
}

template <typename AT, typename BT, typename DT>
gpublasStatus_t bmm_fp8_internal_cublaslt(void* workspace, size_t workspace_size_in_bytes,
                                          const AT* A, const BT* B, DT* D, int batch_size, int m,
                                          int n, int k, const float* A_scale, const float* B_scale,
                                          gpublasLtHandle_t lt_handle, gpuStream_t stream) {
  const void* A_scale_ptr = static_cast<const void*>(A_scale);
  const void* B_scale_ptr = static_cast<const void*>(B_scale);
  auto matmul_desp = CuBlasLtMatmulDescriptor(GPUBLAS_COMPUTE_32F, GPU_R_32F);
  matmul_desp.setAttribute(GPUBLASLT_MATMUL_DESC_TRANSA, GPUBLAS_OP_T);
  matmul_desp.setAttribute(GPUBLASLT_MATMUL_DESC_TRANSB, GPUBLAS_OP_N);
  int8_t fast_accum = 1;
  matmul_desp.setAttribute(GPUBLASLT_MATMUL_DESC_FAST_ACCUM, fast_accum);

  matmul_desp.setAttribute(GPUBLASLT_MATMUL_DESC_A_SCALE_POINTER, A_scale_ptr);
  matmul_desp.setAttribute(GPUBLASLT_MATMUL_DESC_B_SCALE_POINTER, B_scale_ptr);

  gpuDataType_t a_type = get_cuda_data_type<AT>();
  gpuDataType_t b_type = get_cuda_data_type<BT>();
  gpuDataType_t d_type = get_cuda_data_type<DT>();
  if (std::is_same_v<AT, __gpu_fp8_e5m2> && std::is_same_v<BT, __gpu_fp8_e5m2>) {
    FLASHINFER_ERROR("Unsupported combination: both A and B are e5m2");
  }

  auto a_desp = CuBlasLtMatrixLayout(a_type, m, k, k, true);
  auto b_desp = CuBlasLtMatrixLayout(b_type, k, n, k);
  auto d_desp = CuBlasLtMatrixLayout(d_type, m, n, m);

  if (batch_size > 1) {
    int64_t stride_a = m * k;
    int64_t stride_b = k * n;
    int64_t stride_d = m * n;
    a_desp.setAttribute(GPUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, batch_size);
    a_desp.setAttribute(GPUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, stride_a);
    b_desp.setAttribute(GPUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, batch_size);
    b_desp.setAttribute(GPUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, stride_b);
    d_desp.setAttribute(GPUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, batch_size);
    d_desp.setAttribute(GPUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, stride_d);
  }

  CuBlasLtMatmulPreference preference;
  preference.setAttribute(GPUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, workspace_size_in_bytes);
  gpublasLtMatmulHeuristicResult_t heuristic_result = {};
  int returned_result = 0;
  FLASHINFER_CUBLAS_CALL(gpublasLtMatmulAlgoGetHeuristic(
      lt_handle, matmul_desp.descriptor(), a_desp.descriptor(), b_desp.descriptor(),
      d_desp.descriptor(), d_desp.descriptor(), preference.descriptor(), 1, &heuristic_result,
      &returned_result));
  if (returned_result == 0) {
    return GPUBLAS_STATUS_NOT_SUPPORTED;
  }

  const float alpha = 1.0f;
  const float beta = 0.0f;
  FLASHINFER_CUBLAS_CALL(gpublasLtMatmul(
      lt_handle, matmul_desp.descriptor(), &alpha, A, a_desp.descriptor(), B, b_desp.descriptor(),
      &beta, nullptr, d_desp.descriptor(), D, d_desp.descriptor(), &heuristic_result.algo,
      workspace, workspace_size_in_bytes, stream));
  return GPUBLAS_STATUS_SUCCESS;
}

}  // namespace bmm_fp8
}  // namespace flashinfer

#endif  // FLASHINFER_GEMM_BMM_FP8_CUH_
