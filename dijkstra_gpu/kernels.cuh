#include "cuda_runtime.h"
#include "device_launch_parameters.h"

//#define PRINT_NUM

__global__ void gpu_init_buffers(
    size_t source,
    int *mask,
    float *cost,
    float *updating_cost,
    int *pred,
    int *updating_pred,
    size_t size)
{
    //unsigned int tid = threadIdx.x + (blockIdx.x * blockDim.x);

    unsigned int num_blocks = gridDim.x;
    unsigned int block_threads = blockDim.x;
    unsigned int data_size = size;

    int elems_per_block = (int)ceil((float)data_size / (float)num_blocks);
    int elems_per_thread = elems_per_block / block_threads;

    int th_offset = threadIdx.x * elems_per_thread;
    int bl_offset = blockIdx.x * elems_per_block;

#ifdef PRINT_NUM
    printf ("elems_per_block: %d\n \
            elems_per_thread: %d\n", elems_per_block, elems_per_thread);
#endif

    for (int i = bl_offset + th_offset;
             i < bl_offset + th_offset + elems_per_thread &&
             i < size;
             ++i)
    {
        //if (i < size)
        {
            mask[i] = 0;
            cost[i] = FLT_MAX;
            updating_cost[i] = FLT_MAX;
            pred[i] = -1;
            updating_pred[i] = -1;
        }

        if (i == source)
        {
            mask[i] = 1;
            cost[i] = 0;
            updating_cost[i] = 0;
            pred[i] = i;
            updating_pred[i] = i;
        }
    }
}

__global__ void gpu_shortest_path(
    int *vertices,
    int *edges,
    float *weights,
    int *mask,
    float *costs,
    float *updating_costs,
    int *pred,
    int *updating_pred,
    size_t vertex_count,
    size_t edge_count)
{
    //extern __shared__ int shared_vertices[];

    //unsigned int tid = threadIdx.x + (blockIdx.x * blockDim.x);
    unsigned int num_blocks = gridDim.x;
    unsigned int block_threads = blockDim.x;
    unsigned int data_size = vertex_count;

    int elems_per_block = (int)ceil((float)data_size / (float)num_blocks);
    int elems_per_thread = elems_per_block / block_threads;

    int th_offset = threadIdx.x * elems_per_thread;
    int bl_offset = blockIdx.x * elems_per_block;

    for (int i = bl_offset + th_offset;
             i < (bl_offset + th_offset + elems_per_thread) &&
             i < vertex_count;
             ++i)
    {
        //if (i < vertex_count)
        {
            int edges_end;

            if ((i + 1) < vertex_count) edges_end = vertices[i + 1];
            else edges_end = edge_count;

            if (mask[i] != 0)
            {
                mask[i] = 0;

                for (int edge = vertices[i]; edge < edges_end; ++edge)
                {
                    int nid = edges[edge];
                    if (updating_costs[nid] > (costs[i] + weights[edge]))
                    {
                        updating_costs[nid] = costs[i] + weights[edge];
                        updating_pred[nid] = i;
                        mask[i] = 1;
                    }
                }
            }
        }
    }
}

__global__ void gpu_shortest_path2(
    int *mask,
    float *costs,
    float *updating_costs,
    int *pred,
    int *updating_pred,
    unsigned int *any_changes,
    size_t vertex_count)
{
    //unsigned int tid = threadIdx.x + (blockIdx.x * blockDim.x);

    unsigned int num_blocks = gridDim.x;
    unsigned int block_threads = blockDim.x;
    unsigned int data_size = vertex_count;

    int elems_per_block = (int)ceil((float)data_size / (float)num_blocks);
    int elems_per_thread = elems_per_block / block_threads;

    int th_offset = threadIdx.x * elems_per_thread;
    int bl_offset = blockIdx.x * elems_per_block;

#ifdef PRINT_NUM
    printf ("elems_per_block: %d\n \
            elems_per_thread: %d\n", elems_per_block, elems_per_thread);
#endif

    for (int i = bl_offset + th_offset;
             i < bl_offset + th_offset + elems_per_thread &&
             i < vertex_count;
             ++i)
    {
        //if ()
        {
            if (costs[i] > updating_costs[i])
            {
                costs[i] = updating_costs[i];
                pred[i] = updating_pred[i];
                mask[i] = 1;
                any_changes[0]++;
                //atomicInc(&any_changes[0], 1);
            }

            updating_costs[i] = costs[i];
            updating_pred[i] = pred[i];
        }
    }
}
