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

    int *h_vertices;
    int *h_edges;
    float *h_weights;

public:
    void generate_random_graph(int in_vertex_count, int in_neighbour_count);
    void set_graph(std::unique_ptr<graph_data> in_graph) { m_graph = std::move(in_graph); }
    void run_gpu();
    void run_cpu();

private:
    void allocate_gpu_buffers();
    void deallocate_gpu_buffers();
};

void dijkstra_framework::allocate_gpu_buffers()
{
    if (m_graph == nullptr)
    {
        throw std::logic_error("No graph.");
    }

    gpu::malloc(reinterpret_cast<void **>(&h_vertices), m_graph->vertices.size());
    gpu::malloc(reinterpret_cast<void **>(&h_edges), m_graph->edges.size());
    gpu::malloc(reinterpret_cast<void **>(&h_weights), m_graph->weights.size());

    gpu::memcpy_cpu_to_gpu(h_vertices, m_graph->vertices.data(), m_graph->vertices.size());
    gpu::memcpy_cpu_to_gpu(h_edges, m_graph->edges.data(), m_graph->edges.size());
    gpu::memcpy_cpu_to_gpu(h_weights, m_graph->weights.data(), m_graph->weights.size());
}

void dijkstra_framework::deallocate_gpu_buffers()
{
    gpu::free(h_weights);
    gpu::free(h_edges);
    gpu::free(h_vertices);
}

void dijkstra_framework::run_gpu()
{
    allocate_gpu_buffers();

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