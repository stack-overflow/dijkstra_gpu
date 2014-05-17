#include <iostream>

#include "dijkstra_framework.h"

#define SAMPLE_DATA

int main()
{
    try
    {
        // ------------------
        dijkstra_framework dijkstra;
#ifndef SAMPLE_DATA
        dijkstra.create_random_graph(32, 16);
#else
        dijkstra.create_sample_graph();
#endif
        dijkstra.set_source(0);
        dijkstra.run_gpu();

        float *result = dijkstra.get_result();
        for (int i = 0; i < 7; ++i)
        {
            std::cout << result[i] << std::endl;
        }
    }
    catch(std::exception e)
    {
        std::cout << "Error occured: " << e.what() << std::endl;
    }

    std::cout << "Still going, nice!" << std::endl;

    return 0;
}
