# C++ & CUDA Starter Kit for Python Developers

![CUDA Python Starter Kit Thumbnail](https://github.com/ashvardanian/ashvardanian/blob/master/repositories/cuda-python-starter-kit.jpg?raw=true)

One of the most common workflows in high-performance computing is to 1️⃣ prototype algorithms in Python and then 2️⃣ port them to C++ and CUDA.
It's a simple way to prototype and test ideas quickly, but configuring the build tools for such heterogenous code + heterogeneous hardware projects is a pain, often amplified by the error-prone syntax of CMake.
This project provides a pre-configured environment for such workflows...:

1. using only `setup.py` and `requirements-{cpu,gpu}.txt` to manage the build process,
2. supporting OpenMP for parallelism on the CPU, and CUDA for GPU, and
3. including [CCCL](https://github.com/NVIDIA/cccl) libraries, like Thrust, and CUB, to simplify the code.

As an example, the repository implements, tests, and benchmarks only 2 operations - array accumulation and matrix multiplication.
The baseline Python + Numba implementations are placed in `starter_kit_baseline.py`, and the optimized CUDA nd OpenMP implementations are placed in `starter_kit.cu`.
If no CUDA-capable device is found, the file will be treated as a CPU-only C++ implementation.
If VSCode is used, the `tasks.json` file is configured with debuggers for both CPU and GPU code, both in Python and C++.
The `.clang-format` is configured with LLVM base style, adjusted for wider screens, allowing 120 characters per line.

## Installation

I'd recommend forking the repository for your own projects, but you can also clone it directly:

```bash
git clone https://github.com/ashvardanian/cpp-cuda-python-starter-kit.git
cd cpp-cuda-python-starter-kit
```

Once pulled down, you can build the project with:

```bash
git submodule update --init --recursive     # fetch CCCL libraries
pip install -r requirements-gpu.txt         # or requirements-cpu.txt
pip install -e .                            # compile for the current platform
pytest test.py -s -x                        # test until first failure
python bench.py                             # saves charts to disk
```

## Workflow

The project is designed to be as simple as possible, with the following workflow:

1. Fork or download the repository.
2. Implement your baseline algorithm in `starter_kit_baseline.py`.
3. Implement your optimized algorithm in `starter_kit.cu`.

## Reading Materials

Beginner GPGPU:

- High-level concepts: [nvidia.com](https://developer.nvidia.com/blog/even-easier-introduction-cuda/)
- Nvidia CuPy UDFs: [cupy.dev](https://docs.cupy.dev/en/stable/user_guide/kernel.html)
- CUDA in Python with Numba: [numba/nvidia-cuda-tutorial](https://github.com/numba/nvidia-cuda-tutorial)
- C++ STL Parallelism on GPUs: [nvidia.com](https://developer.nvidia.com/blog/accelerating-standard-c-with-gpus-using-stdpar/)

Advanced GPGPU:

- CUDA math intrinsics: [nvidia.com](https://docs.nvidia.com/cuda/cuda-math-api/index.html)
- Troubleshooting Nvidia hardware: [stas00/ml-engineering](https://github.com/stas00/ml-engineering/blob/master/compute/accelerator/nvidia/debug.md)
- Nvidia ISA Generator with SM89 and SM90 codes: [kuterd/nv_isa_solver](https://github.com/kuterd/nv_isa_solver)
- Multi GPU examples: [nvidia/multi-gpu-programming-models](https://github.com/NVIDIA/multi-gpu-programming-models)

Communities:

- CUDA MODE on [Discord](https://discord.com/invite/cudamode)
- r/CUDA on [Reddit](https://www.reddit.com/r/CUDA/)
- NVIDIA Developer Forums on [DevTalk](https://forums.developer.nvidia.com)

