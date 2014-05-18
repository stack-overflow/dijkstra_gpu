#include <iostream>

#include "dijkstra_framework.h"
#include "timekeeper.h"

//#define SAMPLE_DATA

int main()
{
    try
    {
        // ------------------
        timekeeper timer;

        dijkstra_framework dijkstra;
#ifndef SAMPLE_DATA
        dijkstra.create_random_graph(10000000, 4);
#else
        dijkstra.create_sample_graph();
#endif
        dijkstra.set_source(1024);

        // ---- GPU ----
        std::cout << "GPU will prepare now." << std::endl;
        dijkstra.prepare_gpu();
        std::cout << "GPU prepared." << std::endl;

        auto gpu_time = timer.measure_time([&] {
            dijkstra.run_gpu();
        });

        std::cout << "GPU version took: " << gpu_time << " ms" << std::endl;

        float *result = dijkstra.get_result_gpu();
        for (int i = 0; i < 7; ++i)
        {
            std::cout << result[i] << std::endl;
        }
        
        dijkstra.cleanup_gpu();

        // ---- CPU ----

        std::cout << "CPU will prepare now." << std::endl;
        dijkstra.prepare_cpu();
        std::cout << "CPU prepared." << std::endl;

        auto cpu_time = timer.measure_time([&]() {
            dijkstra.run_cpu();
        });

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
