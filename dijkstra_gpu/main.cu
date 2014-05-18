#include <iostream>

#include "dijkstra_framework.h"

//#define SAMPLE_DATA

int main()
{
    try
    {
        // ------------------
        dijkstra_framework dijkstra;
#ifndef SAMPLE_DATA
        dijkstra.create_random_graph(10000000, 4);
#else
        dijkstra.create_sample_graph();
#endif
        dijkstra.set_source(1024);

        // ---- GPU ----

        dijkstra.prepare_gpu();

        auto gpu_start = std::chrono::steady_clock::now();
        dijkstra.run_gpu();
        auto gpu_end = std::chrono::steady_clock::now();
        auto gpu_time = std::chrono::duration<long double, std::milli>(gpu_end - gpu_start).count();
        std::cout << "GPU version took: " << gpu_time << " ms" << std::endl;

        float *result = dijkstra.get_result_gpu();
        for (int i = 0; i < 7; ++i)
        {
            std::cout << result[i] << std::endl;
        }
        
        dijkstra.cleanup_gpu();

        gpu::device_synchronize();

        // ---- CPU ----

        std::cout << "CPU will prepare now." << std::endl;
        dijkstra.prepare_cpu();
        std::cout << "CPU prepared." << std::endl;
        auto cpu_start = std::chrono::steady_clock::now();
        dijkstra.run_cpu();
        auto cpu_end = std::chrono::steady_clock::now();
        auto cpu_time = std::chrono::duration<long double, std::milli>(cpu_end - cpu_start).count();

        std::cout << "CPU version took: " << cpu_time << " ms" << std::endl;

        result = dijkstra.get_result_cpu();
        for (int i = 0; i < 7; ++i)
        {
            std::cout << result[i] << std::endl;
        }

        dijkstra.cleanup_cpu();
    }
    catch(std::exception e)
    {
        std::cout << "Error occured: " << e.what() << std::endl;
    }

    std::cout << "Still going, nice!" << std::endl;

    return 0;
}
