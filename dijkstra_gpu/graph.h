#pragma once
#include <vector>

struct graph_data
{
    std::vector<int> vertices;
    std::vector<int> edges;
    std::vector<float> weights;

    graph_data(int in_vertex_count, int in_edge_count) :
        vertices(in_vertex_count),
        edges(in_edge_count),
        weights(in_edge_count)
    {}

    void clear()
    {
        vertices.clear();
        edges.clear();
        weights.clear();
    }

};