import pytest
import numpy as np
from cupy_starter_baseline import matmul as baseline_matmul, reduce as baseline_reduce

# Import the compiled module from the C++/CUDA code
import cupy_starter


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int64, np.uint64])
@pytest.mark.parametrize("backend", ["openmp", "cuda"])
def test_reduce(dtype, backend):
    # Generate random data
    data = (np.random.rand(1024) * 100).astype(dtype)

    # Get the expected result from the baseline implementation
    expected_result = baseline_reduce(data)

    # Get the result from the C++/CUDA implementation
    if backend == "openmp":
        result = cupy_starter.reduce_openmp(data)
    elif backend == "cuda":
        result = cupy_starter.reduce_cuda(data)

    # Compare the results
    np.testing.assert_allclose(result, expected_result, rtol=1e-2)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int64, np.uint64])
@pytest.mark.parametrize("tile_size", [4, 8, 16, 32, 64])
@pytest.mark.parametrize("backend", ["openmp", "cuda"])
def test_matmul(dtype, tile_size, backend):
    # Generate random matrices
    a = (np.random.rand(256, 256) * 100).astype(dtype)
    b = (np.random.rand(256, 256) * 100).astype(dtype)

    # Get the expected result from the baseline implementation
    expected_result = baseline_matmul(a, b)

    # Get the result from the C++/CUDA implementation
    if backend == "openmp":
        result = cupy_starter.matmul_openmp(a, b, tile_size=tile_size)
    elif backend == "cuda":
        result = cupy_starter.matmul_cuda(a, b, tile_size=tile_size)

    # Compare the results
    np.testing.assert_allclose(result, expected_result, rtol=1e-2)


if __name__ == "__main__":
    pytest.main()
