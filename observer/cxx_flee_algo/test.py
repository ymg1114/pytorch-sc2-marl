import sys, os
import ctypes
import random
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from pathlib import Path


def generate_random_test(n_ally, n_enemy, n_rows, n_cols):
    # Generate random alive statuses for allies and enemies
    alive_allies_np = np.array([random.choice([True, False]) for _ in range(n_ally)], dtype=np.bool_)
    alive_enemies_np = np.array([random.choice([True, False]) for _ in range(n_enemy)], dtype=np.bool_)

    # Generate random positions for allies
    positions_allies_np = np.zeros(n_ally, dtype=[("y", np.int32), ("x", np.int32)])
    positions_allies_np["y"] = -1  # Initialize "y" with -1
    positions_allies_np["x"] = -1  # Initialize "x" with -1
    for i in range(n_ally):
        while True:
            y, x = random.randint(0, n_rows - 1), random.randint(0, n_cols - 1)
            if not any((positions_allies_np["y"] == y) & (positions_allies_np["x"] == x)):
                positions_allies_np[i] = (y, x)
                break

    # Generate random positions for enemies
    positions_enemies_np = np.zeros(n_enemy, dtype=[("y", np.int32), ("x", np.int32)])
    positions_enemies_np["y"] = -1  # Initialize "y" with -1
    positions_enemies_np["x"] = -1  # Initialize "x" with -1
    for i in range(n_enemy):
        while True:
            y, x = random.randint(0, n_rows - 1), random.randint(0, n_cols - 1)
            # Check for overlap with allies or already assigned enemy positions
            if not any((positions_allies_np["y"] == y) & (positions_allies_np["x"] == x)) and not any(
                (positions_enemies_np[:i]["y"] == y) & (positions_enemies_np[:i]["x"] == x)
            ):
                positions_enemies_np[i] = (y, x)
                break

    # Create grid_maps and pre-set allies/enemies positions as obstacles
    grid_maps_np = np.zeros((n_ally, n_rows, n_cols), dtype=np.int32)

    # Pre-set allies' and enemies' positions as obstacles
    for ally in positions_allies_np:
        grid_maps_np[:, ally["y"], ally["x"]] = 1  # Mark allies' positions as obstacles

    for enemy in positions_enemies_np:
        grid_maps_np[:, enemy["y"], enemy["x"]] = 1  # Mark enemies' positions as obstacles

    # Generate random obstacles, avoiding existing allies and enemies
    num_obstacles = random.randint(1, (n_rows * n_cols) // 6)
    for _ in range(num_obstacles):
        while True:
            y, x = random.randint(0, n_rows - 1), random.randint(0, n_cols - 1)
            # Ensure obstacles do not overlap with allies or enemies
            if grid_maps_np[0, y, x] == 0:  # Check the first map only (all maps share the same obstacle layout)
                grid_maps_np[:, y, x] = 1  # Set obstacle on all maps
                break

    return grid_maps_np, positions_allies_np, positions_enemies_np, alive_allies_np, alive_enemies_np


def run_test(lib, n_ally, n_enemy, n_rows, n_cols):
    # Generate random data
    grid_maps_np, positions_allies_np, positions_enemies_np, alive_allies_np, alive_enemies_np = generate_random_test(
        n_ally, n_enemy, n_rows, n_cols
    )

    print(f"positions_allies_np: {positions_allies_np}")
    print(f"positions_enemies_np: {positions_enemies_np}")
    print(f"alive_allies_np: {alive_allies_np}")
    print(f"alive_enemies_np: {alive_enemies_np}")

    # Convert grid maps to ctypes
    grid_maps_ctypes = (ctypes.POINTER(ctypes.c_int) * n_ally)()
    for i in range(n_ally):
        grid_maps_ctypes[i] = grid_maps_np[i].ctypes.data_as(ctypes.POINTER(ctypes.c_int))

    # Output array for flee positions
    flee_positions_np = np.zeros(n_ally, dtype=[("y", np.int32), ("x", np.int32)])

    # Call the function
    lib.compute_flee_positions(
        ctypes.c_int(n_ally),
        ctypes.c_int(n_enemy),
        ctypes.c_int(n_rows),
        ctypes.c_int(n_cols),
        grid_maps_ctypes,
        alive_allies_np.ctypes.data_as(ctypes.POINTER(ctypes.c_bool)),
        alive_enemies_np.ctypes.data_as(ctypes.POINTER(ctypes.c_bool)),
        positions_allies_np.ctypes.data_as(ctypes.POINTER(Position)),
        positions_enemies_np.ctypes.data_as(ctypes.POINTER(Position)),
        ctypes.c_int(3),  # min_search_length
        ctypes.c_int(6),  # max_search_length
        ctypes.c_float(1.0),  # ally_weight
        ctypes.c_float(-2.5),  # enemy_weight
        flee_positions_np.ctypes.data_as(ctypes.POINTER(Position)),  # Output array
    )

    print(f"flee_positions_np: {flee_positions_np}")
    print("*" * 100)
    # Return all data for visualization
    return grid_maps_np, positions_allies_np, positions_enemies_np, flee_positions_np, alive_allies_np, alive_enemies_np

# Visualization function
def visualize_grid_maps(
    iter_, grid_maps_np, positions_allies_np, positions_enemies_np, flee_positions_np, alive_allies_np, alive_enemies_np
):
    n_ally = grid_maps_np.shape[0]  # Number of allies (grid maps)
    n_rows, n_cols = grid_maps_np.shape[1:]  # Grid dimensions

    fig, axes = plt.subplots(1, n_ally, figsize=(6 * n_ally, 6), facecolor="white")  # White background
    if n_ally == 1:
        axes = [axes]  # Ensure axes is iterable for a single ally case

    for i, ax in enumerate(axes):
        # Plot the grid map as a grid
        ax.imshow(grid_maps_np[i], cmap="gray", origin="upper", extent=[0, n_cols, 0, n_rows], alpha=0.2)

        # Draw grid lines
        ax.set_xticks(range(n_cols))
        ax.set_yticks(range(n_rows))
        ax.grid(True, which="both", color="black", linewidth=0.5, alpha=0.5)

        # Add obstacles, but exclude alive allies and enemies from being drawn as black
        for y in range(n_rows):
            for x in range(n_cols):
                if grid_maps_np[i, y, x] == 1:
                    # Check if the current cell is an alive ally or enemy
                    is_ally = any(
                        (positions_allies_np["y"] == y)
                        & (positions_allies_np["x"] == x)
                        & alive_allies_np
                    )
                    is_enemy = any(
                        (positions_enemies_np["y"] == y)
                        & (positions_enemies_np["x"] == x)
                        & alive_enemies_np
                    )
                    # Only draw as black if it's not an alive ally or enemy
                    if not (is_ally or is_enemy):
                        ax.add_patch(
                            patches.Rectangle(
                                (x, n_rows - y - 1), 1, 1, color="black", alpha=0.9
                            )
                        )

        # Plot allies (alive only)
        ally_label_added = False
        for j, (ally, alive_a) in enumerate(zip(positions_allies_np, alive_allies_np)):
            if alive_a:  # Only plot alive allies
                ax.plot(
                    ally["x"] + 0.5,
                    n_rows - ally["y"] - 0.5,
                    "bo",
                    label="Ally" if not ally_label_added else "",
                    markersize=20,
                )
                ally_label_added = True  # Ensure "Ally" label is added only once
                if i == j:  # Special case: Highlight the ally on the current grid map
                    ax.plot(
                        ally["x"] + 0.5,
                        n_rows - ally["y"] - 0.5,
                        "y*",
                        markersize=15,
                        label="Self Ally",
                    )

        # Plot enemies (alive only)
        enemy_label_added = False
        for k, (enemy, alive_e) in enumerate(zip(positions_enemies_np, alive_enemies_np)):
            if alive_e:  # Only plot alive enemies
                ax.plot(
                    enemy["x"] + 0.5,
                    n_rows - enemy["y"] - 0.5,
                    "ro",
                    label="Enemy" if not enemy_label_added else "",
                    markersize=20,
                )
                enemy_label_added = True  # Ensure "Enemy" label is added only once

        # Plot flee position (only if valid)
        flee = flee_positions_np[i]
        if flee["y"] != -9999 and flee["x"] != -9999:
            ax.plot(
                flee["x"] + 0.5,
                n_rows - flee["y"] - 0.5,
                "go",
                label="Flee Position",
                markersize=20,
            )

        # Annotations and legends
        ax.set_title(f"Grid Map {i}", fontsize=14)
        ax.set_xticks(range(n_cols))
        ax.set_yticks(range(n_rows))
        ax.set_xlim(0, n_cols)
        ax.set_ylim(0, n_rows)
        ax.set_aspect("equal")
        ax.legend(loc="upper right", fontsize=12, handlelength=2)

    plt.tight_layout()
    # plt.show()
    os.makedirs("observer/cxx_flee_algo/saved", exist_ok=True)
    plt.savefig(f"observer/cxx_flee_algo/saved/output_{iter_}.png")


if __name__ == "__main__":
    # Determine the library name based on the platform
    if sys.platform.startswith("win"):
        lib_name = Path("observer") / "cxx_flee_algo" / "lib_win" / "FleeAlgorithm.dll"
    else:
        assert sys.platform == "linux", f"running platform is '{sys.platform}'"
        lib_name = Path("observer") / "cxx_flee_algo" / "lib_linux" / "libFleeAlgorithm.so"

    # Load the shared library
    lib = ctypes.CDLL(str(lib_name))

    # Define Position struct
    class Position(ctypes.Structure):
        _fields_ = [("y", ctypes.c_int), ("x", ctypes.c_int)]

    # Define the function prototype
    lib.compute_flee_positions.argtypes = [
        ctypes.c_int,  # n_ally
        ctypes.c_int,  # n_enemy
        ctypes.c_int,  # n_rows
        ctypes.c_int,  # n_cols
        ctypes.POINTER(ctypes.POINTER(ctypes.c_int)),  # grid_maps
        ctypes.POINTER(ctypes.c_bool),  # alive_allies
        ctypes.POINTER(ctypes.c_bool),  # alive_enemies
        ctypes.POINTER(Position),  # positions_allies
        ctypes.POINTER(Position),  # positions_enemies
        ctypes.c_int,  # min_search_length
        ctypes.c_int,  # max_search_length
        ctypes.c_float,  # ally_weight
        ctypes.c_float,  # enemy_weight
        ctypes.POINTER(Position),  # flee_positions (output)
    ]

    for i in range(50):
        grid_maps_np, positions_allies_np, positions_enemies_np, flee_positions_np, alive_allies_np, alive_enemies_np = run_test(
            lib, n_ally=6, n_enemy=5, n_rows=15, n_cols=15
        )

        # Call visualization function
        visualize_grid_maps(
            i, grid_maps_np, positions_allies_np, positions_enemies_np, flee_positions_np, alive_allies_np, alive_enemies_np
        )