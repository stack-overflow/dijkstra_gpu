#pragma once
#include <vector>

struct graph_data
{
    std::vector<int> vertices;
    std::vector<int> edges;
    std::vector<float> weights;

    std::vector< std::pair<int, int> > metric_vertices;
    int size_x;
    int size_y;

    graph_data(size_t in_vertex_count, size_t in_edge_count) :
        vertices(in_vertex_count),
        edges(in_edge_count),
        weights(in_edge_count)
    {}

    void clear()
    {
        std::vector<int>().swap(vertices);
        std::vector<int>().swap(edges);
        std::vector<float>().swap(weights);

        std::vector< std::pair<int, int> >().swap(metric_vertices);
    }
};