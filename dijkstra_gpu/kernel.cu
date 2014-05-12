#include <iostream>

#include "dijkstra_framework.h"

int main()
{
    try
    {
        // ------------------
        dijkstra_framework dijkstra;
        dijkstra.generate_random_graph(1024, 32);

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
