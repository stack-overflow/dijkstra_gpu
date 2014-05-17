#pragma once
// STD
#include <memory>
#include <random>
#include <chrono>
#include <functional>

#include <cstdio>
#include <cassert>

// CUDA
#include "kernels.cuh"

// Dijkstra
#include "graph.h"
#include "gpu.h"
#include "sample_data.h"


class dijkstra_framework
{
private:
    std::unique_ptr<graph_data> m_graph;
    int   m_source;
    int   m_destination;
    bool  m_computed;
    float *m_result;

    int   *d_vertices;
    int   *d_edges;
    float *d_weights;

    int   *d_mask;
    float *d_cost;
    float *d_updating_cost;

    int   *d_any_changes;

public:
    dijkstra_framework();
    ~dijkstra_framework();

    void create_random_graph(int in_vertex_count, int in_neighbour_count);
    void create_sample_graph();
    void set_graph(std::unique_ptr<graph_data> in_graph) { m_graph = std::move(in_graph); }
    void set_source(int in_source) { m_source = in_source; }
    float *get_result();
    void run_gpu();
    void run_cpu();

private:
    void allocate_gpu_buffers();
    void deallocate_gpu_buffers();
};

dijkstra_framework::dijkstra_framework() :
    m_source(0),
    m_computed(false)
{
}

dijkstra_framework::~dijkstra_framework()
{
    gpu::free_cpu(m_result);
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
    gpu::malloc(reinterpret_cast<void **>(&d_mask), sizeof(int) * m_graph->vertices.size());
    gpu::malloc(reinterpret_cast<void **>(&d_cost), sizeof(float) * m_graph->vertices.size());
    gpu::malloc(reinterpret_cast<void **>(&d_updating_cost), sizeof(float) * m_graph->vertices.size());
    gpu::malloc(reinterpret_cast<void **>(&d_any_changes), sizeof(int) * 1);

    m_result = (float *)gpu::malloc_cpu(sizeof(float) * m_graph->vertices.size());

    gpu::memcpy_cpu_to_gpu(d_vertices, &m_graph->vertices[0], sizeof(int) * m_graph->vertices.size());
    gpu::memcpy_cpu_to_gpu(d_edges, &m_graph->edges[0], sizeof(int) * m_graph->edges.size());
    gpu::memcpy_cpu_to_gpu(d_weights, &m_graph->weights[0], sizeof(float) * m_graph->weights.size());
}

void dijkstra_framework::deallocate_gpu_buffers()
{
    gpu::free(d_any_changes);
    gpu::free(d_updating_cost);
    gpu::free(d_cost);
    gpu::free(d_mask);
    gpu::free(d_weights);
    gpu::free(d_edges);
    gpu::free(d_vertices);
}

void dijkstra_framework::run_gpu()
{
    int num_threads         = 8;
    int overall_threads_num = gpu::get_overall_num_threads(num_threads, m_graph->vertices.size());
    int num_blocks          = overall_threads_num / num_threads;

    std::cout << "num_blocks: " << num_blocks << " num_threads: " << num_threads << std::endl;

    m_computed = false;

    allocate_gpu_buffers();
    gpu_init_buffers<<<num_blocks, num_threads>>>(m_source, d_mask, d_cost, d_updating_cost, m_graph->vertices.size());

    int any_changes = 1;
    int cnt         = 0;

    std::cout << "main loop" << std::endl;
    
    while (any_changes)
    {
        gpu::memset(d_any_changes, 0, sizeof(int));
        gpu_shortest_path<<< num_blocks, num_threads >>>(d_vertices, d_edges, d_weights, d_mask, d_cost, d_updating_cost, m_graph->vertices.size(), m_graph->edges.size());
        gpu_shortest_path2<<< num_blocks, num_threads >>>(d_mask, d_cost, d_updating_cost, d_any_changes, m_graph->vertices.size());
        gpu::memcpy_gpu_to_cpu(&any_changes, d_any_changes, sizeof(int));

        ++cnt;
    }

    gpu::memcpy_gpu_to_cpu(m_result, d_cost, sizeof(float) * m_graph->vertices.size());

    m_computed = true;

    std::cout << "Number of loop cycles: " << cnt << std::endl;

    deallocate_gpu_buffers();
}

float *dijkstra_framework::get_result()
{
    return m_result;
}

void dijkstra_framework::create_random_graph(int in_vertex_count, int in_neighbour_count)
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

    std::uniform_real_distribution<float> distribution_float(0, 10.0f);
    auto random_float = std::bind(distribution_float, generator);

    for (size_t i = 0; i < m_graph->vertices.size(); ++i)
    {
        m_graph->vertices[i] = i * in_neighbour_count;
    }

    for (size_t i = 0; i < m_graph->edges.size(); ++i)
    {
        m_graph->edges[i] = random_int(); // 1 .. vertex_count
        m_graph->weights[i] = random_float(); // 0 .. 10.0f
    }
}

void dijkstra_framework::create_sample_graph()
{
    auto v = sample_data::get_vertices_array();
    auto e = sample_data::get_edges_array();
    auto w = sample_data::get_weights_array();

    m_graph = std::unique_ptr<graph_data>(
        new graph_data(v.second, e.second));

    for (size_t i = 0; i < m_graph->vertices.size(); ++i)
    {
        m_graph->vertices[i] = v.first[i];
    }

    for (size_t i = 0; i < m_graph->edges.size(); ++i)
    {
        m_graph->edges[i] = e.first[i];
        m_graph->weights[i] = w.first[i];
    }
}