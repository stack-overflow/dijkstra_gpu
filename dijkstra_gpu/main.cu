#include <iostream>

#include "dijkstra_framework.h"
#include "timekeeper.h"

int main()
{
    try
    {
        // ------------------
        timekeeper timer;

        dijkstra_framework dijkstra;
        dijkstra.create_random_graph(10000000, 4);

        //dijkstra.create_sample_graph();

        dijkstra.set_source(0);

        // ---- GPU ----
        std::cout << "GPU will prepare now.\n";

        auto gpu_prepare_time = timer.measure_time([&]() {
            dijkstra.prepare_gpu();
        });

        std::cout << "GPU prepare time: " << gpu_prepare_time << " ms\n";

        auto gpu_run_time = timer.measure_time([&] {
            dijkstra.run_gpu();
        });

        std::cout << "GPU run time: " << gpu_run_time << " ms\n";

        float *result = dijkstra.get_result_gpu();
        for (int i = 0; i < 7; ++i)
        {
            std::cout << result[i] << std::endl;
        }
        
        auto gpu_cleanup_time = timer.measure_time([&] {
            dijkstra.cleanup_gpu();
        });

        std::cout << "GPU cleanup time: " << gpu_cleanup_time << " ms\n";
        std::cout << "GPU overall time: " << gpu_prepare_time + gpu_run_time + gpu_cleanup_time << " ms" << std::endl;

        // ENTER THE VOID
        
        std::cout << std::endl;

        // EXIT THE VOID

        // ---- CPU ----
        std::cout << "CPU will prepare now." << std::endl;

        auto cpu_prepare_time = timer.measure_time([&]() {
            dijkstra.prepare_cpu();
        });

        std::cout << "CPU prepare time:" << cpu_prepare_time << "ms\n";

        auto cpu_run_time = timer.measure_time([&]() {
            dijkstra.run_cpu();
        });

        std::cout << "CPU run time: " << cpu_run_time << " ms\n";

        result = dijkstra.get_result_cpu();
        for (int i = 0; i < 7; ++i)
        {
            std::cout << result[i] << std::endl;
        }

        auto cpu_cleanup_time = timer.measure_time([&] {
            dijkstra.cleanup_cpu();
        });

        std::cout << "CPU cleanup time: " << cpu_cleanup_time << " ms\n";
        std::cout << "CPU overall time: " << cpu_prepare_time + cpu_run_time + cpu_cleanup_time << " ms" << std::endl;
    }
    catch(std::exception e)
    {
        std::cout << "Error occured: " << e.what() << std::endl;
    }

    std::cout << "Thank you for your cooperation!" << std::endl;

    return 0;
}
