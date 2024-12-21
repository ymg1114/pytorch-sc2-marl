#include "flee_algorithm.h"
#include <queue>
#include <cmath>
#include <limits>
#include <iostream>
#include <cstring> // For memset

#if defined(BUILD_DLL) || defined(__GNUC__)
    // Helper function to perform BFS for a single ally
    static void bfs_search(
        const int* grid_map, // Pointer to the grid map data (flattened 1D array)
        int rows,
        int cols,
        Position start_pos,
        int min_search_length,
        int max_search_length,
        Position* accessible_positions, // Output array
        int* num_accessible_positions // Number of accessible positions found
    ) {
        // Initialize visited array
        bool* visited = new bool[rows * cols];
        memset(visited, 0, sizeof(bool) * rows * cols);

        std::queue<std::pair<Position, int>> q; // ((y, x), distance)
        q.push({ start_pos, 0 });
        visited[start_pos.y * cols + start_pos.x] = true;

        int count = 0;

        while (!q.empty()) {
            auto current = q.front();
            q.pop();

            int y = current.first.y;
            int x = current.first.x;
            int dist = current.second;

            if (dist >= min_search_length && dist <= max_search_length) {
                accessible_positions[count++] = { y, x };
            }

            // Stop expanding if max_search_length is reached
            if (dist >= max_search_length) {
                continue;
            }

            // Explore neighboring positions
            Position directions[] = {
                { -1, 0 }, // Up
                { 1, 0 },  // Down
                { 0, -1 }, // Left
                { 0, 1 }   // Right
            };

            for (int i = 0; i < 4; ++i) {
                int newY = y + directions[i].y;
                int newX = x + directions[i].x;

                // Check bounds and if the position is accessible and not visited
                if (newY >= 0 && newY < rows && newX >= 0 && newX < cols &&
                   grid_map[newY * cols + newX] == 0 && !visited[newY * cols + newX]) {
                    visited[newY * cols + newX] = true;
                    q.push({ { newY, newX }, dist + 1 });
                }
            }
        }

        *num_accessible_positions = count;
        delete[] visited;
    }

    void compute_flee_positions(
        int n, // Number of allies
        int m, // Number of enemies
        int rows, // Number of rows in the grid maps
        int cols, // Number of columns in the grid maps
        int** grid_maps, // Array of pointers to grid maps (each grid map is rows * cols)
        bool* alive_allies, // Alive status of allies
        bool* alive_enemies, // Alive status of enemies
        Position* positions_allies, // Positions of allies
        Position* positions_enemies, // Positions of enemies
        int min_search_length,
        int max_search_length,
        float ally_weight,
        float enemy_weight,
        Position* flee_positions // Output flee positions
    ) {
        for (int i = 0; i < n; ++i) {
            // Skip dead allies
            if (!alive_allies[i]) {
                flee_positions[i] = { -9999, -9999 };
                continue;
            }

            // Prepare the grid map for this ally
            int* grid_map = grid_maps[i];

            // Maximum possible positions within the search range
            int max_positions = (2 * max_search_length + 1) * (2 * max_search_length + 1);
            Position* accessible_positions = new Position[max_positions];
            int num_accessible_positions = 0;

            // Perform BFS to find accessible positions
            bfs_search(
                grid_map,
                rows,
                cols,
                positions_allies[i],
                min_search_length,
                max_search_length,
                accessible_positions,
                &num_accessible_positions
            );

            // If no accessible positions found, assign (-9999, -9999)
            if (num_accessible_positions == 0) {
                flee_positions[i] = { -9999, -9999 };
                delete[] accessible_positions;
                continue;
            }

            Position best_position = accessible_positions[0];
            float max_score = -std::numeric_limits<float>::infinity();

            // For each accessible position, compute the total score
            for (int p = 0; p < num_accessible_positions; ++p) {
                Position pos = accessible_positions[p];
                float total_score = 0.0f;

                // Calculate score from other alive allies (excluding self)
                for (int j = 0; j < n; ++j) {
                    if (j != i && alive_allies[j]) {
                        float dx = (float)(pos.x - positions_allies[j].x);
                        float dy = (float)(pos.y - positions_allies[j].y);
                        float distance = sqrtf(dx * dx + dy * dy) + 1e-6f;
                        total_score += ally_weight * (1.0f / distance);
                    }
                }

                // Calculate score from alive enemies
                for (int k = 0; k < m; ++k) {
                    if (alive_enemies[k]) {
                        float dx = (float)(pos.x - positions_enemies[k].x);
                        float dy = (float)(pos.y - positions_enemies[k].y);
                        float distance = sqrtf(dx * dx + dy * dy) + 1e-6f;
                        total_score += enemy_weight * (1.0f / distance);
                    }
                }

                // Update the best position if current total_score is higher
                if (total_score > max_score) {
                    max_score = total_score;
                    best_position = pos;
                }
            }

            // Assign the best flee position
            flee_positions[i] = best_position;

            delete[] accessible_positions;
        }
    }
#endif