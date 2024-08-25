import numpy as np
import perfplot

from starter_kit_baseline import matmul as matmul_baseline, reduce as reduce_baseline
from starter_kit import (
    reduce_openmp,
    reduce_cuda,
    matmul_openmp,
    matmul_cuda,
    supports_cuda,
)


# Set up the `perfplot` for the reduce operation
def run_perfplot_reduce(dtype=np.float32):
    labels = ["reduce_baseline", "reduce_openmp"]
    kernels = [reduce_baseline, reduce_openmp]
    if supports_cuda():
        kernels.append(reduce_cuda)
        labels.append("reduce_cuda")

    result = perfplot.bench(
        setup=lambda n: (np.random.rand(n) * 100).astype(dtype),
        kernels=kernels,
        labels=labels,
        n_range=[2**i for i in range(10, 20)],
        flops=lambda n: n,
        xlabel="Input Size",
        equality_check=np.allclose,
        target_time_per_measurement=0.1,
    )

    result.save(f"reduce_{dtype.__name__}.png", transparent=True, bbox_inches="tight")


# Set up the `perfplot` for the matrix multiplication operation
def run_perfplot_matmul(dtype=np.float32, tile_sizes: list = [4, 8, 16, 32, 64]):

    labels = []
    kernels = []
    for tile_size in tile_sizes:
        labels.append(f"matmul_baseline_{tile_size}")
        kernels.append(lambda data: matmul_baseline(data, data, tile_size=tile_size))
        labels.append(f"matmul_openmp_{tile_size}")
        kernels.append(lambda data: matmul_openmp(data, data, tile_size=tile_size))
        if supports_cuda():
            labels.append(f"matmul_cuda_{tile_size}")
            kernels.append(lambda data: matmul_cuda(data, data, tile_size=tile_size))

    result = perfplot.bench(
        setup=lambda n: (np.random.rand(n, n) * 100).astype(dtype),
        kernels=kernels,
        labels=labels,
        n_range=[2**i for i in range(6, 11)],
        flops=lambda n: n**3,
        xlabel="Matrix Side",
        equality_check=np.allclose,
        target_time_per_measurement=0.1,
    )

    result.save(f"matmul_{dtype.__name__}.png", transparent=True, bbox_inches="tight")


if __name__ == "__main__":
    run_perfplot_reduce(np.float32)
    run_perfplot_reduce(np.int32)
    run_perfplot_matmul(np.float32)
    run_perfplot_matmul(np.int32)
