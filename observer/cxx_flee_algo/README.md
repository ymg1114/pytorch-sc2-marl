# CMake Build Command:

## Linux (GCC/Clang):
```bash
mkdir build_linux
cd build_linux

cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=g++
make
```

## Windows (MSVC):
```bash
mkdir build_windows
cd build_windows

cmake .. -G "Visual Studio 17 2022" -A x64
cmake --build . --config Release
```


# FleeAlgorithm: A Brief Overview:
- A **BFS-based exploration algorithm**.
- Each agent (one of the allies) explores and identifies the optimal **"flee" coordinates (y, x)**.
- The algorithm aims to position **allies closer together** while **avoiding enemy locations as much as possible**, with this behavior adjustable through **hyperparameters**.
- Returns the optimal "flee" target position for the agent.


## Example: `python -m observer.cxx_flee_algo.test`, [test.py](https://github.com/ymg1114/pytorch-sc2-marl/blob/main/observer/cxx_flee_algo/test.py)
- ![output_3](https://github.com/user-attachments/assets/cae0ed24-4813-4b11-9177-7fda1baed373)
- ![output_4](https://github.com/user-attachments/assets/799ee435-a6f5-4943-9793-3ad1afdb6da3)
