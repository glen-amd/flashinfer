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
#ifndef FLASHINFER_FRAG_LAYOUT_SWIZZLE_CUH_
#define FLASHINFER_FRAG_LAYOUT_SWIZZLE_CUH_

#if defined(__HIPCC__) || (defined(__clang__) && defined(__HIP__)) || defined(__HIPCC_RTC__)
#include <hip/hip_runtime.h>
#elif defined(__CUDACC__) || defined(__NVCC__) || (defined(__clang__) && defined(__CUDA__)) || defined(__CUDACC_RTC__)
#include <cuda_runtime.h>
#endif

#include <cstdint>

__device__ __forceinline__ uint32_t frag_layout_swizzle_16b_to_8b(uint32_t x) {
#if defined(__HIPCC__) || (defined(__clang__) && defined(__HIP__)) || defined(__HIPCC_RTC__)
  // FIXME: May miss the synchronization across threads in the warp.
  uint32_t tmp = __shfl_xor(x, 0x1);
#else
  uint32_t tmp = __shfl_xor_sync(0xffffffff, x, 0x1);
#endif
  x = __byte_perm(x, tmp, ((threadIdx.x & 0x1) == 0) ? 0x5410 : 0x3276);
#if defined(__HIPCC__) || (defined(__clang__) && defined(__HIP__)) || defined(__HIPCC_RTC__)
  tmp = __shfl_xor(x, 0x2);
#else
  tmp = __shfl_xor_sync(0xffffffff, x, 0x2);
#endif
  x = __byte_perm(x, tmp, ((threadIdx.x & 0x2) == 0) ? 0x5410 : 0x3276);
  return x;
}

__device__ __forceinline__ uint32_t frag_layout_swizzle_16b_to_8b_trans(uint32_t x) {
#if defined(__HIPCC__) || (defined(__clang__) && defined(__HIP__)) || defined(__HIPCC_RTC__)
  // FIXME: May miss the synchronization across threads in the warp.
  uint32_t tmp = __shfl_xor(x, 0x4);
#else
  uint32_t tmp = __shfl_xor_sync(0xffffffff, x, 0x4);
#endif
  x = __byte_perm(x, tmp, ((threadIdx.x & 0x4) == 0) ? 0x6420 : 0x3175);
#if defined(__HIPCC__) || (defined(__clang__) && defined(__HIP__)) || defined(__HIPCC_RTC__)
  tmp = __shfl_xor(x, 0x8);
#else
  tmp = __shfl_xor_sync(0xffffffff, x, 0x8);
#endif
  x = __byte_perm(x, tmp, ((threadIdx.x & 0x8) == 0) ? 0x5410 : 0x3276);
#if defined(__HIPCC__) || (defined(__clang__) && defined(__HIP__)) || defined(__HIPCC_RTC__)
  tmp = __shfl_xor(x, 0x10);
#else
  tmp = __shfl_xor_sync(0xffffffff, x, 0x10);
#endif
  x = __byte_perm(x, tmp, ((threadIdx.x & 0x10) == 0) ? 0x5410 : 0x3276);
  return x;
}

#endif  // FLASHINFER_FRAG_LAYOUT_SWIZZLE_CUH_
