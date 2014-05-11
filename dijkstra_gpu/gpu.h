#pragma once

// CUDA
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

class gpu
{
public:
    static void malloc(void **mem, size_t size)
    {
        if (cudaMalloc(mem, size) != cudaSuccess) throw std::runtime_error("gpu malloc failed");
    }

    static void free(void *mem)
    {
        if (cudaFree(mem) != cudaSuccess) throw std::runtime_error("gpu free failed");
    }

    static void device_synchronize()
    {
        if (cudaDeviceSynchronize() != cudaSuccess) throw std::runtime_error("gpu synchronize failed");
    }

    template<typename T>
    static void memcpy_cpu_to_gpu(T *dst, T *src, size_t size)
    {
        if (cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice) != cudaSuccess) throw std::runtime_error("memcpy cpu to gpu failed");
    }

    template<typename T>
    static void memcpy_gpu_to_cpu(T *dst, T *src, size_t size)
    {
        if (cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost) != cudaSuccess) throw std::runtime_error("memcpy gpu to cpu failed");
    }
};