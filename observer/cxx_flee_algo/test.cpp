#include "flee_algorithm.h"
#include <iostream>

int main() {
    // Initialize inputs (similar to the test case)
    int n = 2; // Number of allies
    int m = 2; // Number of enemies
    int rows = 5;
    int cols = 5;

    // Allocate grid maps
    int** grid_maps = new int* [n];
    for (int i = 0; i < n; ++i) {
        grid_maps[i] = new int[rows * cols];
        for (int j = 0; j < rows * cols; ++j) {
            grid_maps[i][j] = 0; // Mark all positions as accessible (default)
        }
    }

    // Example: Add obstacles to Ally 0's grid map
    grid_maps[0][1 * cols + 2] = 1; // Block position (1, 2)
    grid_maps[0][2 * cols + 1] = 0; // Block position (2, 1)
    grid_maps[0][2 * cols + 3] = 1; // Block position (2, 3)
    grid_maps[0][3 * cols + 2] = 1; // Block position (3, 2)

    // Print grid map for Ally 0
    std::cout << "Grid map for Ally 0:" << std::endl;
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            std::cout << grid_maps[0][r * cols + c] << " ";
        }
        std::cout << std::endl;
    }

    // Alive statuses
    bool alive_allies[] = { true, false };
    bool alive_enemies[] = { true, true };

    // Positions
    Position positions_allies[] = { {2, 2}, {0, 0} };
    Position positions_enemies[] = { {0, 4}, {4, 4} };

    // Output array
    Position* flee_positions = new Position[n];

    // Call the function
    compute_flee_positions(
        n, m, rows, cols,
        grid_maps,
        alive_allies,
        alive_enemies,
        positions_allies,
        positions_enemies,
        1, 3,
        1.0f, -1.0f,
        flee_positions
    );

    // Output results
    for (int i = 0; i < n; ++i) {
        std::cout << "Ally " << i << " flee position: ("
            << flee_positions[i].y << ", "
            << flee_positions[i].x << ")" << std::endl;
    }

    // Clean up
    delete[] flee_positions;
    for (int i = 0; i < n; ++i) {
        delete[] grid_maps[i];
    }
    delete[] grid_maps;

    return 0;
}
