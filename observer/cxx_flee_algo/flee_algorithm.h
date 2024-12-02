#ifndef FLEE_ALGORITHM_H
#define FLEE_ALGORITHM_H

#ifdef _WIN32
    #ifdef BUILD_DLL
        #define DLL_API __declspec(dllexport)
    #else
        #define DLL_API __declspec(dllimport)
    #endif
#else
    #ifdef __GNUC__
        #define DLL_API __attribute__((visibility("default")))
    #else
        #define DLL_API
    #endif
#endif



#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int y;
    int x;
} Position;

// Function to compute flee positions
DLL_API void compute_flee_positions(
    int n, // Number of allies
    int m, // Number of enemies
    int rows, // Number of rows in the grid maps
    int cols, // Number of columns in the grid maps
    int** grid_maps, // Flattened grid maps for each ally (size: n * rows * cols)
    bool* alive_allies, // Alive status of allies (size: n)
    bool* alive_enemies, // Alive status of enemies (size: m)
    Position* positions_allies, // Positions of allies (size: n)
    Position* positions_enemies, // Positions of enemies (size: m)
    int min_search_length,
    int max_search_length,
    float ally_weight,
    float enemy_weight,
    Position* flee_positions // Output flee positions (size: n)
);

#ifdef __cplusplus
}
#endif

#endif // FLEE_ALGORITHM_H
