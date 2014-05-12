#pragma once
// STD
#include <memory>
#include <random>
#include <chrono>
#include <functional>

// CUDA
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// Dijkstra
#include "graph.h"
#include "gpu.h"

class dijkstra_framework
{
private:
    std::unique_ptr<graph_data> m_graph;

    int *d_vertices;
    int *d_edges;
    float *d_weights;

    unsigned char *d_mask;
    float *d_cost;
    float *d_updating_cost;

public:
    void generate_random_graph(int in_vertex_count, int in_neighbour_count);
    void set_graph(std::unique_ptr<graph_data> in_graph) { m_graph = std::move(in_graph); }
    void run_gpu();
    void run_cpu();

private:
    void allocate_gpu_buffers();
    void deallocate_gpu_buffers();
};

__global__ void gpu_init_buffers(unsigned char *mask, float *cost, float *updating_cost, int size)
{
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < size)
    {
        mask[tid] = 0;
        cost[tid] = FLT_MAX;
        updating_cost[tid] = FLT_MAX;
    }
}

void dijkstra_framework::allocate_gpu_buffers()
{
    if (m_graph == nullptr)
    {
        throw std::logic_error("No graph.");
    }

    gpu::malloc(reinterpret_cast<void **>(&d_vertices), sizeof(int) * m_graph->vertices.size());
    gpu::malloc(reinterpret_cast<void **>(&d_edges), sizeof(int) * m_graph->edges.size());
    gpu::malloc(reinterpret_cast<void **>(&d_weights), sizeof(float) * m_graph->weights.size());
    gpu::malloc(reinterpret_cast<void **>(&d_mask), sizeof(unsigned char) * m_graph->vertices.size());
    gpu::malloc(reinterpret_cast<void **>(&d_cost), sizeof(float) * m_graph->vertices.size());
    gpu::malloc(reinterpret_cast<void **>(&d_updating_cost), sizeof(float) * m_graph->vertices.size());

    gpu::memcpy_cpu_to_gpu(d_vertices, m_graph->vertices.data(), m_graph->vertices.size());
    gpu::memcpy_cpu_to_gpu(d_edges, m_graph->edges.data(), m_graph->edges.size());
    gpu::memcpy_cpu_to_gpu(d_weights, m_graph->weights.data(), m_graph->weights.size());
}

void dijkstra_framework::deallocate_gpu_buffers()
{
    gpu::free(d_updating_cost);
    gpu::free(d_cost);
    gpu::free(d_mask);
    gpu::free(d_weights);
    gpu::free(d_edges);
    gpu::free(d_vertices);
}

bool is_mask_empty(unsigned char *mask, int size)
{
    for (int i = 0; i < size; ++i)
    {
        if (mask[i] == 1) return false;
    }

    return true;
}

void dijkstra_framework::run_gpu()
{
    int num_threads         = 128;
    int overall_threads_num = gpu::get_overall_num_threads(num_threads, m_graph->vertices.size());
    int num_blocks          = overall_threads_num / num_threads;

    std::cout << "num_blocks: " << num_blocks << " num_threads: " << num_threads << std::endl;

    allocate_gpu_buffers();
    gpu_init_buffers<<<num_blocks, num_threads>>>(d_mask, d_cost, d_updating_cost, m_graph->vertices.size());

    unsigned char *mask = new unsigned char[m_graph->vertices.size()];

    while (is_mask_empty(mask, m_graph->vertices.size()))
    {
        // Do CUDA things.
    }

    delete [] mask;

    deallocate_gpu_buffers();
}

void dijkstra_framework::generate_random_graph(int in_vertex_count, int in_neighbour_count)
{
    if ((in_neighbour_count < 0) || (in_neighbour_count < 0) || (in_vertex_count < in_neighbour_count))
    {
        throw std::logic_error("Wrong graph size.");
    }

    m_graph = std::unique_ptr<graph_data>(
            new graph_data(in_vertex_count,
                           in_vertex_count * in_neighbour_count));

    unsigned int seed = static_cast<unsigned int>(std::chrono::system_clock::now().time_since_epoch().count());
    std::default_random_engine generator(seed);

    std::uniform_int_distribution<int> distribution_int(0, m_graph->vertices.size());
    auto random_int = std::bind(distribution_int, generator);

    std::uniform_real_distribution<float> distribution_float(0, 1.0f);
    auto random_float = std::bind(distribution_float, generator);

    for (size_t i = 0; i < m_graph->vertices.size(); ++i)
    {
        m_graph->vertices[i] = i * in_neighbour_count;
    }

    for (size_t i = 0; i < m_graph->edges.size(); ++i)
    {
        m_graph->edges[i] = random_int(); // 1 .. vertex_count
        m_graph->weights[i] = random_float(); // 0 .. 1.0f
    }
}