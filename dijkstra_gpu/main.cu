#include <iostream>

#include "dijkstra_framework.h"
#include "timekeeper.h"
#include "runtime_info.h"

int main()
{
    try
    {
        // ------------------
        timekeeper timer;

        dijkstra_framework dijkstra;
        dijkstra.create_random_graph(100000, 4);
        //dijkstra.create_sample_graph();

        runtime_info gpu_times;
        runtime_info cpu_times;

        dijkstra.set_source(0);

        // ---- GPU ----
        std::cout << "GPU will prepare now.\n";

        gpu_times.prepare_time = timer.measure_time([&]() {
            dijkstra.prepare_gpu();
        });

        std::cout << "GPU prepare time: " << gpu_times.prepare_time << " ms\n";

        gpu_times.run_time = timer.measure_time([&] {
            dijkstra.run_gpu();
        });

        std::cout << "GPU run time: " << gpu_times.run_time << " ms\n";

        float *result = dijkstra.get_result_gpu();
        for (int i = 0; i < 7; ++i)
        {
            std::cout << result[i] << std::endl;
        }
        int *path = dijkstra.get_result_path_gpu();
        for (int i = 0; i < 7; ++i)
        {
            std::cout << path[i] << " ";
        }
        std::cout << std::endl;

        gpu_times.cleanup_time = timer.measure_time([&] {
            dijkstra.cleanup_gpu();
        });

        std::cout << "GPU cleanup time: " << gpu_times.cleanup_time << " ms\n";
        std::cout << "GPU overall time: " << gpu_times.prepare_time + gpu_times.run_time + gpu_times.cleanup_time << " ms" << std::endl;

        // ENTER THE VOID
        
        std::cout << "----" << std::endl;

        // EXIT THE VOID

        // ---- CPU ----
        std::cout << "CPU will prepare now." << std::endl;

        cpu_times.prepare_time = timer.measure_time([&]() {
            dijkstra.prepare_cpu();
        });

        std::cout << "CPU prepare time:" << cpu_times.prepare_time << "ms\n";

        cpu_times.run_time = timer.measure_time([&]() {
            dijkstra.run_cpu();
        });

        std::cout << "CPU run time: " << cpu_times.run_time << " ms\n";

        result = dijkstra.get_result_cpu();
        for (int i = 0; i < 7; ++i)
        {
            std::cout << result[i] << std::endl;
        }

        path = dijkstra.get_result_path_cpu();
        for (int i = 0; i < 7; ++i)
        {
            std::cout << path[i] << " ";
        }
        std::cout << std::endl;

        cpu_times.cleanup_time = timer.measure_time([&] {
            dijkstra.cleanup_cpu();
        });

        std::cout << "CPU cleanup time: " << cpu_times.cleanup_time << " ms\n";
        std::cout << "CPU overall time: " << cpu_times.prepare_time + cpu_times.run_time + cpu_times.cleanup_time << " ms" << std::endl;
    }
    catch(std::exception e)
    {
        std::cout << "Error occured: " << e.what() << std::endl;
    }

    std::cout << "Thank you for your cooperation!" << std::endl;

    return 0;
}
