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

    //template<typename T>
    static void memcpy_cpu_to_gpu(void *d_dst, void *h_src, size_t size)
    {
        cudaError_t result = cudaMemcpy(d_dst, h_src, size, cudaMemcpyHostToDevice);
        if (result != cudaSuccess) throw std::runtime_error("memcpy cpu to gpu failed, error " + std::to_string(result));
    }

    //template<typename T>
    static void memcpy_gpu_to_cpu(void *h_dst, void *d_src, size_t size)
    {
        cudaError_t result = cudaMemcpy(h_dst, d_src, size, cudaMemcpyDeviceToHost);
        if (result != cudaSuccess) throw std::runtime_error("memcpy gpu to cpu failed, error " + std::to_string(result));
    }

    static void memset(void *d_dst, int value, size_t size)
    {
        cudaError_t result = cudaMemset(d_dst, value, size);
        if (result != cudaSuccess) throw std::runtime_error("gpu memset failed, error " + std::to_string(result));
    }

    static void *malloc_cpu(size_t size)
    {
        void *h_ptr;
        cudaError_t result = cudaMallocHost(&h_ptr, size);
        if (result != cudaSuccess) throw std::runtime_error("cudaMallocHost failed, error " + std::to_string(result));
        return h_ptr;
    }

    static void free_cpu(void *host_ptr)
    {
        cudaError_t result = cudaFreeHost(host_ptr);
        if (result != cudaSuccess) throw std::runtime_error("cudaFreeHost failed, error " + std::to_string(result));
    }

    static size_t get_overall_num_threads(int threads_per_block, int num_elements_to_process)
    {
        int rest = num_elements_to_process % threads_per_block;
        return (rest == 0) ? (num_elements_to_process) : (num_elements_to_process + threads_per_block - rest);
    }
};