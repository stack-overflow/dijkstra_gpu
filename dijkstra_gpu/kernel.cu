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
        dijkstra.generate_random_graph(1024, 32);
#else
        
#endif
        dijkstra.set_source(0);
        dijkstra.run_gpu();
    }
    catch(std::exception e)
    {
        std::cout << "Error occured: " << e.what() << std::endl;
    }

    std::cout << "Still going, nice!" << std::endl;

    return 0;
}
