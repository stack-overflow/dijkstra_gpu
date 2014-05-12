#pragma once
// STD
#include <string>

// CUDA
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

class gpu
{
public:
    static void malloc(void **mem, size_t size)
    {
        cudaError_t result = cudaMalloc(mem, size);
        if (result != cudaSuccess) throw std::runtime_error("gpu malloc failed, error " + std::to_string(result));
    }

    static void free(void *mem)
    {
        cudaError_t result = cudaFree(mem);
        if (result != cudaSuccess) throw std::runtime_error("gpu free failed, error " + std::to_string(result));
    }

    static void device_synchronize()
    {
        cudaError_t result = cudaDeviceSynchronize();
        if (result != cudaSuccess) throw std::runtime_error("gpu synchronize failed, error " + std::to_string(result));
    }

    template<typename T>
    static void memcpy_cpu_to_gpu(T *dst, T *src, size_t size)
    {
        cudaError_t result = cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
        if (result != cudaSuccess) throw std::runtime_error("memcpy cpu to gpu failed, error " + std::to_string(result));
    }

    template<typename T>
    static void memcpy_gpu_to_cpu(T *dst, T *src, size_t size)
    {
        cudaError_t result = cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
        if (result != cudaSuccess) throw std::runtime_error("memcpy gpu to cpu failed, error " + std::to_string(result));
    }

    static size_t get_overall_num_threads(int threads_per_block, int num_elements_to_process)
    {
        int rest = num_elements_to_process % threads_per_block;
        return (rest == 0) ? (num_elements_to_process) : (num_elements_to_process + threads_per_block - rest);
    }
};