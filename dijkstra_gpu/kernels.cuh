#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void gpu_init_buffers(size_t source, int *mask, float *cost, float *updating_cost, int size)
{
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < size)
    {
        mask[tid] = 0;
        cost[tid] = 1764.14f;
        updating_cost[tid] = 1764.14f;
    }

    if (tid == source)
    {
        mask[tid] = 1;
        cost[tid] = 0;
        updating_cost[tid] = 0;
    }
}

__global__ void gpu_shortest_path(
    int *vertices,
    int *edges,
    float *weights,
    int *mask,
    float *costs,
    float *updating_costs,
    int vertex_count,
    int edge_count)
{
    unsigned int tid = threadIdx.x + (blockIdx.x * blockDim.x);
    
    if (tid < vertex_count)
    {
        int edges_end;

        if ((tid + 1) < vertex_count) edges_end = vertices[tid + 1];
        else edges_end = edge_count;

        if (mask[tid] != 0)
        {
            mask[tid] = 0;

            for (int i = vertices[tid]; i < edges_end; ++i)
            {
                int nid = edges[i];
                if (updating_costs[nid] > (costs[tid] + weights[i]))
                {
                    updating_costs[nid] = costs[tid] + weights[i];
                }
            }
        }
    }
}

__global__ void gpu_shortest_path2(
    int *mask,
    float *costs,
    float *updating_costs,
    int *any_changes,
    int vertex_count)
{
    unsigned int tid = threadIdx.x + (blockIdx.x * blockDim.x);

    if (tid < vertex_count)
    {
        if (costs[tid] > updating_costs[tid])
        {
            costs[tid] = updating_costs[tid];
            mask[tid] = 1;
            any_changes[0]++;
        }

        updating_costs[tid] = costs[tid];
    }
}