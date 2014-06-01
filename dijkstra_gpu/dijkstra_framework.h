#pragma once
// STD
#include <memory>
#include <random>
#include <chrono>
#include <functional>
#include <queue>

#include <cstdio>
#include <cassert>

// CUDA
#include "kernels.cuh"

// Dijkstra
#include "graph.h"
#include "gpu.h"
#include "sample_data.h"

#include "CImg/CImg.h"
using namespace cimg_library;

class dijkstra_framework
{
private:
    std::unique_ptr<graph_data> m_graph;
    unsigned int m_block_size;
    unsigned int m_num_blocks;

    int   m_source;
    int   m_destination;
    float *m_result;
    int   *m_path;
    unsigned int *m_any_changes;

    int   *d_vertices;
    int   *d_edges;
    float *d_weights;

    int   *d_mask;
    float *d_cost;
    float *d_updating_cost;
    int   *d_pred;
    int   *d_updating_pred;

    unsigned int *d_any_changes;

public:
    dijkstra_framework();
    ~dijkstra_framework();

    void create_random_graph(int in_vertex_count, int in_neighbour_count);
    void create_random_graph_metric(int in_vertex_count, int in_neighbour_count, int sizer_x, int size_y);

    // Creates sample graph
    void create_sample_graph();

    void set_graph(std::unique_ptr<graph_data> in_graph) { m_graph = std::move(in_graph); }

    // Set source and destination for Dijkstra algorithm.
    void set_source(int in_source) { m_source = in_source; }
    void set_destination(int in_destination) { m_destination = in_destination; }

    // Maximum blocks size is 1024 threads per block.
    void set_block_size(size_t in_num_threads_per_block) { m_block_size = (in_num_threads_per_block > 1024) ? 1024 : (unsigned int)in_num_threads_per_block; }

    // Maximum number of blocks is 65535.
    void set_block_number(size_t in_num_blocks) { m_num_blocks = (in_num_blocks > 65535) ? 65535 : (unsigned int)in_num_blocks; }

    void clear_graph() { m_graph->clear(); }

    void generate_result_image(const char *filename);

    float *get_result_gpu();
    float *get_result_cpu();

    int *get_result_path_gpu();
    int *get_result_path_cpu();

    void prepare_cpu() { prepare_cpu_buffers(); }
    void cleanup_cpu() { dispose_cpu_buffers(); }

    void prepare_gpu();
    void cleanup_gpu() { dispose_gpu_buffers(); }

    void run_gpu();
    void run_cpu();

private:
    void prepare_gpu_buffers();
    void dispose_gpu_buffers();
    void prepare_cpu_buffers();
    void dispose_cpu_buffers();

    dijkstra_framework(const dijkstra_framework&);
};

dijkstra_framework::dijkstra_framework() :
    m_source(0),
    m_destination(0),
    m_block_size(512)
{
}

dijkstra_framework::~dijkstra_framework()
{
}

void dijkstra_framework::prepare_gpu()
{
    prepare_gpu_buffers();

    *m_any_changes = 1;

    gpu_init_buffers<<<m_num_blocks, m_block_size>>>(m_source, d_mask, d_cost, d_updating_cost, d_pred, d_updating_pred, m_graph->vertices.size());
}

void dijkstra_framework::prepare_gpu_buffers()
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
    gpu::malloc(reinterpret_cast<void **>(&d_pred), sizeof(int) * m_graph->vertices.size());
    gpu::malloc(reinterpret_cast<void **>(&d_updating_pred), sizeof(int) * m_graph->vertices.size());
    gpu::malloc(reinterpret_cast<void **>(&d_any_changes), sizeof(int) * 1);

    m_any_changes = (unsigned int *)gpu::malloc_cpu(sizeof(unsigned int));

    gpu::memcpy_cpu_to_gpu(d_vertices, &m_graph->vertices[0], sizeof(int) * m_graph->vertices.size());
    gpu::memcpy_cpu_to_gpu(d_edges, &m_graph->edges[0], sizeof(int) * m_graph->edges.size());
    gpu::memcpy_cpu_to_gpu(d_weights, &m_graph->weights[0], sizeof(float) * m_graph->weights.size());
}

void dijkstra_framework::dispose_gpu_buffers()
{
    gpu::free_cpu(m_path);
    gpu::free_cpu(m_result);
    gpu::free_cpu(m_any_changes);
    gpu::free(d_any_changes);
    gpu::free(d_updating_pred);
    gpu::free(d_pred);
    gpu::free(d_updating_cost);
    gpu::free(d_cost);
    gpu::free(d_mask);
    gpu::free(d_weights);
    gpu::free(d_edges);
    gpu::free(d_vertices);

    m_result = nullptr;
}

void dijkstra_framework::prepare_cpu_buffers()
{
    if (m_graph == nullptr)
    {
        throw std::logic_error("No graph.");
    }

    m_result = new float[m_graph->vertices.size()];
    m_path = new int[m_graph->vertices.size()];

    std::fill(m_result, m_result + m_graph->vertices.size(), FLT_MAX);
    std::fill(m_path, m_path + m_graph->vertices.size(), -1);
}

void dijkstra_framework::dispose_cpu_buffers()
{
    delete [] m_path;
    delete [] m_result;
}

// GPU Dijkstra algorithm implementation
void dijkstra_framework::run_gpu()
{
    int cnt = 0;
    
    size_t num_blocks = m_num_blocks;
    size_t block_threads = m_block_size;
    size_t data_size = m_graph->vertices.size();

    size_t elems_per_block = (size_t)ceil((float)data_size / (float)num_blocks);
    size_t elems_per_thread = elems_per_block / block_threads;

    printf ("numblocks: %d\nelems_per_block: %d\nelems_per_thread:%d\nproduct: %d\n",
        num_blocks, elems_per_block, elems_per_thread, elems_per_block * num_blocks);

    while ((*m_any_changes) != 0)
    {
        gpu::memset(d_any_changes, 0, sizeof(unsigned int));
        gpu_shortest_path<<< m_num_blocks, m_block_size >>>(d_vertices, d_edges, d_weights, d_mask, d_cost, d_updating_cost, d_pred, d_updating_pred, m_graph->vertices.size(), m_graph->edges.size());
        gpu_shortest_path2<<< m_num_blocks, m_block_size >>>(d_mask, d_cost, d_updating_cost, d_pred, d_updating_pred, d_any_changes, m_graph->vertices.size());

        gpu::memcpy_gpu_to_cpu(m_any_changes, d_any_changes, sizeof(unsigned int));

        ++cnt;
    }

    /*std::cout << "Number of loop cycles: " << cnt << std::endl;*/
}

// Reference CPU Dijkstra algorithm implementation
void dijkstra_framework::run_cpu()
{
    std::priority_queue<std::pair<float, int>,
                        std::vector< std::pair<float, int> >,
                        std::greater< std::pair<float, int> > > pq;

    m_result[m_source] = 0.0f;
    m_path[m_source] = 0;
    pq.push(std::make_pair(0.0f, m_source));

    while (!pq.empty())
    {
        float top_cost = pq.top().first;
        int   top_node = pq.top().second;

        if (top_node >= m_graph->vertices.size()) throw std::logic_error("Overflow!");

        pq.pop();

        size_t neigh_end;
        if ((top_node + 1) < m_graph->vertices.size()) neigh_end = m_graph->vertices[top_node + 1];
        else neigh_end = m_graph->vertices.size();

        //if (top_cost <= m_result[top_node])
        {
            for (size_t i = m_graph->vertices[top_node];
                        i < neigh_end;
                      ++i)
            {
                int nid = m_graph->edges[i];
                float edge_cost = m_graph->weights[i];

                if (m_result[nid] > m_result[top_node] + edge_cost)
                {
                    m_result[nid] = m_result[top_node] + edge_cost;
                    pq.push(std::make_pair(m_result[nid], nid));
                    m_path[nid] = top_node;
                }
            }
        }
    }
}

inline float *dijkstra_framework::get_result_gpu()
{
    m_result = (float *)gpu::malloc_cpu(sizeof(float) * m_graph->vertices.size());
    gpu::memcpy_gpu_to_cpu(m_result, d_cost, sizeof(float) * m_graph->vertices.size());
    return m_result;
}

inline float *dijkstra_framework::get_result_cpu()
{
    return m_result;
}

int *dijkstra_framework::get_result_path_gpu()
{
    m_path = (int *)gpu::malloc_cpu(sizeof(float) * m_graph->vertices.size());
    gpu::memcpy_gpu_to_cpu(m_path, d_pred, sizeof(float) * m_graph->vertices.size());
    return m_path;
}

int *dijkstra_framework::get_result_path_cpu()
{
    return m_path;
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

    std::uniform_int_distribution<int> distribution_int(0, static_cast<int>(m_graph->vertices.size()) - 1);
    auto random_int = std::bind(distribution_int, generator);

    std::uniform_real_distribution<float> distribution_float(0, 10.0f);
    auto random_float = std::bind(distribution_float, generator);

    for (size_t i = 0; i < m_graph->vertices.size(); ++i)
    {
        m_graph->vertices[i] = static_cast<int>(i) * in_neighbour_count;
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

float points_distance(int dX0, int dY0, int dX1, int dY1)
{
    return sqrtf(
        (static_cast<float>(dX1) - static_cast<float>(dX0)) *
        (static_cast<float>(dX1) - static_cast<float>(dX0)) +
        (static_cast<float>(dY1) - static_cast<float>(dY0)) *
        (static_cast<float>(dY1) - static_cast<float>(dY0)));
}
void dijkstra_framework::create_random_graph_metric(int in_vertex_count, int in_neighbour_count, int size_x, int size_y)
{
    if ((in_neighbour_count < 0) || (in_neighbour_count < 0) || (in_vertex_count < in_neighbour_count))
    {
        throw std::logic_error("Wrong graph size.");
    }

    m_graph = std::unique_ptr<graph_data>(
        new graph_data(in_vertex_count,
        in_vertex_count * in_neighbour_count));

    m_graph->metric_vertices.resize(in_vertex_count);

    m_graph->size_x = size_x;
    m_graph->size_y = size_y;

    unsigned int seed = static_cast<unsigned int>(std::chrono::system_clock::now().time_since_epoch().count());
    std::default_random_engine generator(seed);

    std::uniform_int_distribution<int> distribution_int_x(0, size_x - 1);
    auto random_x = std::bind(distribution_int_x,
        std::default_random_engine(
            static_cast<unsigned int>(std::chrono::system_clock::now().time_since_epoch().count())));

    std::uniform_int_distribution<int> distribution_int_y(0, size_y - 1);
    auto random_y = std::bind(distribution_int_y,
        std::default_random_engine());

    std::uniform_int_distribution<int> distribution_int(0, (int)(m_graph->vertices.size()) - 1);
    auto random_int = std::bind(distribution_int, generator);

    std::uniform_real_distribution<float> distribution_float(0, 10.0f);
    auto random_float = std::bind(distribution_float, generator);

    for (size_t i = 0; i < m_graph->vertices.size(); ++i)
    {
        m_graph->vertices[i] = static_cast<int>(i) * in_neighbour_count;
        m_graph->metric_vertices[i].first = rand()%(size_x-1);
        m_graph->metric_vertices[i].second = rand()%(size_y-1);
    }

    for (size_t i = 0; i < m_graph->vertices.size(); ++i)
    {
        int start = m_graph->vertices[i];
        int end   = start + in_neighbour_count;

        for (int j = start; j < end; ++j)
        {
            int rand_neigh;
            do
            {
                rand_neigh = random_int();
            }
            while (rand_neigh == i);
            
            m_graph->edges[j] = rand_neigh;
            m_graph->weights[j] = points_distance(m_graph->metric_vertices[i].first,
                                                  m_graph->metric_vertices[i].second,
                                                  m_graph->metric_vertices[rand_neigh].first,
                                                  m_graph->metric_vertices[rand_neigh].second);
        }
    }
}
//
void dijkstra_framework::generate_result_image(const char *filename)
{
    CImg<unsigned char> img(m_graph->size_x,m_graph->size_y,1,3);        // Define a 640x400 color image with 8 bits per color component.
    img.fill(0);                                 // Set pixel values to 0 (color : black)
    unsigned char purple[] = { 100,0,100 };
    unsigned char red[] = { 255,0,0 };
    unsigned char green[] = { 0,255,0 };


    for (size_t i = 0; i < m_graph->vertices.size(); ++i)
    {
        size_t start = (size_t)m_graph->vertices[i];
        size_t end;

        if ((i + 1) < m_graph->vertices.size()) end = m_graph->vertices[i + 1];
        else end = m_graph->edges.size();

        img.draw_circle(m_graph->metric_vertices[i].first, m_graph->metric_vertices[i].second, 4, green);
        img.draw_text(m_graph->metric_vertices[i].first-12, m_graph->metric_vertices[i].second-12, std::to_string(i).c_str(), green);

        for (size_t j = start; j < end; ++j)
        {
            int nid = m_graph->edges[j];
            img.draw_line(
                m_graph->metric_vertices[i].first,
                m_graph->metric_vertices[i].second,
                m_graph->metric_vertices[nid].first,
                m_graph->metric_vertices[nid].second, purple);
        }

        int it = m_destination;

        while (it != m_source)
        {
            img.draw_line(
                m_graph->metric_vertices[m_path[it]].first,
                m_graph->metric_vertices[m_path[it]].second,
                m_graph->metric_vertices[it].first,
                m_graph->metric_vertices[it].second, red);

            it = m_path[it];
        }

        img.save(filename);
    }

    std::cout << "Weights:" << std::endl;
    for (int i = 0; i < m_graph->weights.size(); ++i)
    {
        std::cout << m_graph->weights[i] << " ";
    }
    std::cout << "----" << std::endl;
}
