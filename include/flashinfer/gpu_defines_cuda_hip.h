/**
 * NB This approach is NOT flexiable, NOT adaptive.
 */
#ifndef FLASHINFER_GPU_DEFINES_CUDA_HIP_H_
#define FLASHINFER_GPU_DEFINES_CUDA_HIP_H_

#ifdef __CUDACC__

#define gpuStream_t cudaStream_t
#define gpuDeviceProp_t cudaDeviceProp
#define gpuError_t cudaError_t
#define gpuSuccess cudaSuccess
#define gpuErrorNotReady cudaErrorNotReady
#define gpuGetDeviceCount cudaGetDeviceCount
#define gpuGetLastError cudaGetLastError
#define gpuPeekAtLastError cudaPeekAtLastError
#define gpuGetErrorName cudaGetErrorName
#define gpuGetErrorString cudaGetErrorString
#define gpuGetDeviceProperties cudaGetDeviceProperties
#define gpuStreamDefault cudaStreamDefault
#define gpuGetDevice cudaGetDevice
#define gpuSetDevice cudaSetDevice
#define gpuMalloc cudaMalloc
#define gpuFree cudaFree
#define gpuMemsetAsync cudaMemsetAsync
#define gpuMemcpyAsync cudaMemcpyAsync
#define gpuMemcpyDeviceToDevice cudaMemcpyDeviceToDevice
#define gpuMemcpyDeviceToHost cudaMemcpyDeviceToHost
#define gpuMemcpyHostToDevice cudaMemcpyHostToDevice
#define gpuStreamQuery cudaStreamQuery
#define gpuSharedMemConfig cudaSharedMemConfig
#define gpuDeviceSetSharedMemConfig cudaDeviceSetSharedMemConfig
#define gpuStreamSynchronize cudaStreamSynchronize
#define gpuDeviceSynchronize cudaDeviceSynchronize
#define gpuMemcpy cudaMemcpy
#define gpuDeviceGetAttribute cudaDeviceGetAttribute
#define gpuFuncSetAttribute cudaFuncSetAttribute
#define gpuLaunchKernel cudaLaunchKernel
#define gpuOccupancyMaxActiveBlocksPerMultiprocessor cudaOccupancyMaxActiveBlocksPerMultiprocessor
#define gpuFuncAttributeMaxDynamicSharedMemorySize cudaFuncAttributeMaxDynamicSharedMemorySize
#define gpuDevAttrMultiProcessorCount cudaDevAttrMultiProcessorCount

#elif defined(__HIPCC__)

#define gpuStream_t hipStream_t
#define gpuDeviceProp_t hipDeviceProp_t
#define gpuError_t hipError_t
#define gpuSuccess hipSuccess
#define gpuErrorNotReady hipErrorNotReady
#define gpuGetDeviceCount hipGetDeviceCount
#define gpuGetLastError hipGetLastError
#define gpuPeekAtLastError hipPeekAtLastError
#define gpuGetErrorName hipGetErrorName
#define gpuGetErrorString hipGetErrorString
#define gpuGetDeviceProperties hipGetDeviceProperties
#define gpuStreamDefault hipStreamDefault
#define gpuGetDevice hipGetDevice
#define gpuSetDevice hipSetDevice
#define gpuMalloc hipMalloc
#define gpuFree hipFree
#define gpuMemsetAsync hipMemsetAsync
#define gpuMemcpyAsync hipMemcpyAsync
#define gpuMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#define gpuMemcpyDeviceToHost hipMemcpyDeviceToHost
#define gpuMemcpyHostToDevice hipMemcpyHostToDevice
#define gpuStreamQuery hipStreamQuery
#define gpuSharedMemConfig hipSharedMemConfig
#define gpuDeviceSetSharedMemConfig hipDeviceSetSharedMemConfig
#define gpuStreamSynchronize hipStreamSynchronize
#define gpuDeviceSynchronize hipDeviceSynchronize
#define gpuMemcpy hipMemcpy
#define gpuDeviceGetAttribute hipDeviceGetAttribute
#define gpuFuncSetAttribute hipFuncSetAttribute
#define gpuLaunchKernel hipLaunchKernel
#define gpuOccupancyMaxActiveBlocksPerMultiprocessor hipOccupancyMaxActiveBlocksPerMultiprocessor
#define gpuFuncAttributeMaxDynamicSharedMemorySize hipFuncAttributeMaxDynamicSharedMemorySize
#define gpuDevAttrMultiProcessorCount hipDevAttrMultiProcessorCount

#endif  // __CUDACC__ or __HIPCC__

// `gpu_assert` can be overridden.
#ifndef gpu_assert

#ifdef __HIP_DEVICE_COMPILE__
// HIPCC does not support the use of assert on the GPU side.
#define gpu_assert(COND)
#else
#define gpu_assert(COND) assert(COND)
#endif

#endif  // gpu_assert

#endif  // FLASHINFER_GPU_DEFINES_CUDA_HIP_H_
