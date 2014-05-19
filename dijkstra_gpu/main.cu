#include <iostream>
#include <sstream>
#include <cstring>
#include <string>

#include "dijkstra_framework.h"
#include "timekeeper.h"
#include "runtime_info.h"

int main(int argc, char *argv[])
{
    int num_vertices = 10000000;
    int num_neighs = 4;
    int num_print_lines = 0;
    int block_size = 512;
    bool print = false;

    if (argc > 1)
    {
        for (int i = 1; i < argc; ++i)
        {
            if (strcmp(argv[i], "-v") == 0)
            {
                if (argc > i + 1)
                {
                    num_vertices = std::stoi(argv[++i]);
                    std::cout << num_vertices << std::endl;
                }
            }
            else if (strcmp(argv[i], "-n") == 0)
            {
                if (argc > i + 1)
                {
                    num_neighs = std::stoi(argv[++i]);
                }
            }
            else if (strcmp(argv[i], "-p") == 0)
            {
                if (argc > i + 1)
                {
                    num_print_lines = std::stoi(argv[++i]);
                }
            }
            else if (strcmp(argv[i], "-t") == 0)
            {
                if (argc > i + 1)
                {
                    block_size = std::stoi(argv[++i]);
                }
            }
        }
    }

    std::cout << "vertices: " << num_vertices << "\n";
    std::cout << "neighbours: " << num_neighs << "\n";
    std::cout << "threads per block: " << block_size << "\n";
    std::cout << "print " << num_print_lines << " first results.\n";

    std::cout << "----" << std::endl;

    try
    {
        // ------------------
        timekeeper timer;

        dijkstra_framework dijkstra;
       // dijkstra.create_random_graph_metric(8, 2, 800, 800);

        dijkstra.create_random_graph(num_vertices, num_neighs);
        //dijkstra.create_sample_graph();

        runtime_info gpu_times;
        runtime_info cpu_times;

        dijkstra.set_source(0);
        dijkstra.set_block_size(block_size);

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
        int *path = dijkstra.get_result_path_gpu();

        for (int i = 0; i < num_print_lines; ++i)
        {
            std::cout << result[i] << std::endl;
        }
            
        for (int i = 0; i < num_print_lines; ++i)
        {
            std::cout << path[i] << " ";
        }
        std::cout << std::endl;

        //dijkstra.generate_result_image("result_map.bmp");

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
        path = dijkstra.get_result_path_cpu();

        for (int i = 0; i < num_print_lines; ++i)
        {
            std::cout << result[i] << std::endl;
        }
            
        for (int i = 0; i < num_print_lines; ++i)
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
