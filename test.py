import pytest
import numpy as np

from starter_kit_baseline import matmul as matmul_baseline, reduce as reduce_baseline
from starter_kit import supports_cuda, reduce_openmp, reduce_cuda, matmul_openmp, matmul_cuda

backends = ["openmp", "cuda"] if supports_cuda() else ["openmp"]

@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int64, np.uint64])
@pytest.mark.parametrize("backend", backends)
def test_reduce(dtype, backend):
    # Generate random data
    data = (np.random.rand(1024) * 100).astype(dtype)

    # Get the expected result from the baseline implementation
    expected_result = reduce_baseline(data)

    # Get the result from the C++/CUDA implementation
    if backend == "openmp":
        result = reduce_openmp(data)
    elif backend == "cuda":
        result = reduce_cuda(data)

    # Compare the results
    np.testing.assert_allclose(result, expected_result, rtol=1e-2)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int64, np.uint64])
@pytest.mark.parametrize("tile_size", [4, 8, 16, 32, 64])
@pytest.mark.parametrize("backend", backends)
def test_matmul(dtype, tile_size, backend):
    # Generate random matrices
    a = (np.random.rand(256, 256) * 100).astype(dtype)
    b = (np.random.rand(256, 256) * 100).astype(dtype)

    # Get the expected result from the baseline implementation
    expected_result = matmul_baseline(a, b)

    # Get the result from the C++/CUDA implementation
    if backend == "openmp":
        result = matmul_openmp(a, b, tile_size=tile_size)
    elif backend == "cuda":
        result = matmul_cuda(a, b, tile_size=tile_size)

    # Compare the results
    np.testing.assert_allclose(result, expected_result, rtol=1e-2)


if __name__ == "__main__":
    pytest.main()
