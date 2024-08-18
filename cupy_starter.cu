/**
 *   @brief  Starter Kit project for CUDA- and OpenMP-accelerated Python libraries.
 *   @author Ash Vardanian
 *   @date   August 10, 2024
 *   @file   cupy_starter.cu
 *   @see    https://github.com/ashvardanian/cuda-python-starter-kit
 */
#include <csignal>   // `std::signal`
#include <cstdint>   // `std::uint32_t`
#include <cstdio>    // `std::printf`
#include <cstdlib>   // `std::rand`
#include <stdexcept> // `std::runtime_error`
#include <thread>    // `std::thread::hardware_concurrency()`
#include <vector>    // `std::vector`

/*
 *  Include the SIMD intrinsics for the target architecture.
 *  Arm: https://developer.arm.com/architectures/instruction-sets/intrinsics
 *  x86: https://www.intel.com/content/www/us/en/docs/intrinsics-guide
 */
#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#endif
#if defined(__AVX2__) || defined(__AVX512F__)
#include <immintrin.h>
#endif

#if defined(__NVCC__)
#include <cuda.h> // `CUtensorMap`
#include <cuda/barrier>
#include <cudaTypedefs.h> // `PFN_cuTensorMapEncodeTiled`
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#endif

/*
 *  If we are only testing the raw kernels, we don't need to link to PyBind.
 *  That accelerates the build process and simplifies the configs.
 */
#if !defined(CUPY_STARTER_TEST)
#include <pybind11/numpy.h> // `array_t`
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
#endif

/*  It's a good idea to specialize kernels for different architectures of GPUs.
 *  - Pascal (6.0) introduced half-precision.
 *  - Volta (7.0) introduced tensor cores.
 *  - Ampere (8.0) introduced TF32.
 *  - Hopper (9.0) introduced FP8. and integer SIMD instructions.
 */
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 300
#define CUPY_STARTER_KEPLER 1
#endif
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 600
#define CUPY_STARTER_PASCAL 1
#endif
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
#define CUPY_STARTER_VOLTA 1
#endif
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
#define CUPY_STARTER_AMPERE 1
#endif
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
#define CUPY_STARTER_HOPPER 1
#endif

using cell_idx_t = std::uint32_t;

enum class backend_t {
    openmp_k,
    cuda_k,
};

/**
 *  @brief Stores the interrupt signal status.
 */
volatile std::sig_atomic_t global_signal_status = 0;

static void signal_handler(int signal) { global_signal_status = signal; }

#pragma region OpenMP and CPU code

template <typename scalar_type>
using reduce_type = std::conditional_t< //
    std::is_floating_point_v<scalar_type>, double,
    std::conditional_t<std::is_signed_v<scalar_type>, std::int64_t, std::uint64_t>>;

/**
 *  @brief Performs a reduction operation on a 1D array using OpenMP for parallelization.
 *
 *  This function reduces a 1D array of elements of type `scalar_type` to a single value
 *  using an OpenMP-enabled parallel reduction. The reduction operation is performed
 *  using multiple threads to sum the elements of the array efficiently.
 *
 *  @tparam scalar_type The data type of the array elements (e.g., float, double).
 *
 *  @param data A pointer to the input array of elements of type `scalar_type`.
 *  @param length The number of elements in the input array.
 *
 *  @return reduce_type<scalar_type> The result of the reduction operation, which is the sum
 *  of all elements in the array.
 */
template <typename scalar_type>
reduce_type<scalar_type> openmp_reduce(scalar_type const* data, std::size_t length) noexcept {
    reduce_type<scalar_type> initial_value = 0;
#pragma omp parallel for reduction(+ : initial_value)
    for (cell_idx_t i = 0; i < length; i++)
        initial_value += data[i];
    // Should be same as `std::accumulate(data.data(), data.data() + length, initial_value)`
    return initial_value;
}

/**
 *  @brief Performs tiled matrix multiplication using OpenMP for parallelization.
 *
 *  This function computes the matrix product of two matrices A and B, storing the result
 *  in matrix C. The multiplication is performed using a tiled approach to optimize cache
 *  usage by copying tiles into stack-allocated arrays. The computation is parallelized
 *  using OpenMP to leverage multiple threads.
 *
 *  @tparam scalar_type The data type of the matrix elements (e.g., float, double).
 *  @tparam tile_size The size of the tiles used for blocking, defaulting to 16.
 *
 *  @param matrix_a Pointer to the input matrix A, stored in row-major order.
 *  @param matrix_b Pointer to the input matrix B, stored in row-major order.
 *  @param matrix_c Pointer to the output matrix C, stored in row-major order.
 *  @param num_rows_a The number of rows in matrix A.
 *  @param num_cols_b The number of columns in matrix B.
 *  @param num_cols_a The number of columns in matrix A, and the number of rows in matrix B.
 *  @param stride_a The stride (leading dimension) of matrix A.
 *  @param stride_b The stride (leading dimension) of matrix B.
 *  @param stride_c The stride (leading dimension) of matrix C.
 *
 *  This function performs the operation:
 *
 *    C = A * B
 *
 *  where A is a (num_rows_a x num_cols_a) matrix, B is a (num_cols_a x num_cols_b) matrix,
 *  and C is a (num_rows_a x num_cols_b) matrix. The computation is broken down into smaller
 *  tile-sized chunks, which are copied into stack-allocated arrays to improve cache efficiency.
 *  The workload is parallelized using OpenMP to distribute the computation across multiple threads.
 */
template <typename scalar_type, cell_idx_t tile_size = 16>                           //
void openmp_matmul(                                                                  //
    scalar_type const* matrix_a, scalar_type const* matrix_b, scalar_type* matrix_c, //
    cell_idx_t num_rows_a, cell_idx_t num_cols_b, cell_idx_t num_cols_a,             //
    cell_idx_t stride_a, cell_idx_t stride_b, cell_idx_t stride_c) noexcept {

#pragma omp parallel for collapse(2)
    for (cell_idx_t i = 0; i < num_rows_a; i += tile_size) {
        for (cell_idx_t j = 0; j < num_cols_b; j += tile_size) {
            scalar_type local_tile_a[tile_size][tile_size];
            scalar_type local_tile_b[tile_size][tile_size];
            scalar_type local_tile_c[tile_size][tile_size];

            // Initialize the local tile to zero
            std::memset(local_tile_c, 0, tile_size * tile_size * sizeof(scalar_type));

            for (cell_idx_t k = 0; k < num_cols_a; k += tile_size) {
                // Load tiles into local memory
                for (cell_idx_t ii = 0; ii < tile_size; ++ii)
                    for (cell_idx_t kk = 0; kk < tile_size; ++kk)
                        local_tile_a[ii][kk] =                           //
                            (i + ii < num_rows_a && k + kk < num_cols_a) //
                                ? matrix_a[(i + ii) * stride_a + (k + kk)]
                                : 0;

                for (cell_idx_t kk = 0; kk < tile_size; ++kk)
                    for (cell_idx_t jj = 0; jj < tile_size; ++jj)
                        local_tile_b[kk][jj] =                           //
                            (k + kk < num_cols_a && j + jj < num_cols_b) //
                                ? matrix_b[(k + kk) * stride_b + (j + jj)]
                                : 0;

                // Perform multiplication on the local tiles
                for (cell_idx_t ii = 0; ii < tile_size; ++ii)
                    for (cell_idx_t jj = 0; jj < tile_size; ++jj)
                        for (cell_idx_t kk = 0; kk < tile_size; ++kk)
                            local_tile_c[ii][jj] += local_tile_a[ii][kk] * local_tile_b[kk][jj];
            }

            // Write the result back to the output matrix
            for (cell_idx_t ii = 0; ii < tile_size; ++ii)
                for (cell_idx_t jj = 0; jj < tile_size; ++jj)
                    if (i + ii < num_rows_a && j + jj < num_cols_b)
                        matrix_c[(i + ii) * stride_c + (j + jj)] += local_tile_c[ii][jj];
        }
    }
}

#pragma endregion OpenMP and CPU code

#pragma region CUDA

#if defined(__NVCC__)

namespace cde = cuda::device::experimental;
using barrier_t = cuda::barrier<cuda::thread_scope_block>;

/**
 *  @brief Performs a reduction operation on a 1D array using CUDA.
 *
 *  This function reduces a contiguous 1D array of elements of type `scalar_type`
 *  to a single value using a CUDA-enabled reduction operation. The reduction
 *  operation is performed on the GPU using the Thrust library, which efficiently
 *  computes the sum (or another binary operation) of all elements in the array.
 *
 *  @tparam scalar_type The data type of the array elements (e.g., float, double).
 *
 *  @param data A pointer to the input array of elements of type `scalar_type`.
 *  @param length The number of elements in the input array.
 *
 *  @return reduce_type<scalar_type> The result of the reduction operation. For
 *  floating-point types, this is typically the sum of all elements in the array.
 */
template <typename scalar_type>
reduce_type<scalar_type> cuda_reduce(scalar_type const* data, std::size_t length) noexcept(false) {
    reduce_type<scalar_type> initial_value = 0;
    thrust::device_vector<scalar_type> device_array(data, data + length);
    return thrust::reduce(thrust::device, device_array.begin(), device_array.end(), initial_value);
}

/**
 *  @brief CUDA kernel for matrix multiplication with support for strided matrices.
 *
 *  This kernel computes the matrix product of two matrices A and B, storing the result in matrix C.
 *  The multiplication is performed in tiles of size `tile_size` to take advantage of shared memory
 *  for optimizing memory access patterns.
 *
 *  @tparam scalar_type The data type of the matrix elements (e.g., float, double).
 *  @tparam tile_size The size of the tiles used for shared memory, defaulting to 16.
 *
 *  @param matrix_a Pointer to the input matrix A, stored in row-major order.
 *  @param matrix_b Pointer to the input matrix B, stored in row-major order.
 *  @param matrix_c Pointer to the output matrix C, stored in row-major order.
 *  @param num_rows_a The number of rows in matrix A.
 *  @param num_cols_b The number of columns in matrix B.
 *  @param num_cols_a The number of columns in matrix A, and the number of rows in matrix B.
 *  @param stride_a The stride (leading dimension) of matrix A.
 *  @param stride_b The stride (leading dimension) of matrix B.
 *  @param stride_c The stride (leading dimension) of matrix C.
 *
 *  This kernel performs the operation:
 *
 *      C = A * B
 *
 *  where A is a (num_rows_a x num_cols_a) matrix, B is a (num_cols_a x num_cols_b) matrix,
 *  and C is a (num_rows_a x num_cols_b) matrix. The computation is broken down into smaller
 *  tile-sized chunks, which are loaded into shared memory to reduce global memory access overhead.
 *
 *  Each thread block computes a tile of the output matrix C by iterating over the corresponding
 *  tiles of matrices A and B. The kernel ensures correct handling of matrix boundaries and
 *  supports strided matrices, where elements of a row are not necessarily contiguous in memory.
 */
template <typename scalar_type, cell_idx_t tile_size = 16>                           //
void cuda_matmul_kernel(                                                             //
    scalar_type const* matrix_a, scalar_type const* matrix_b, scalar_type* matrix_c, //
    cell_idx_t num_rows_a, cell_idx_t num_cols_b, cell_idx_t num_cols_a,             //
    cell_idx_t stride_a, cell_idx_t stride_b, cell_idx_t stride_c) noexcept {

    // Allocate shared memory for matrix_a and matrix_b tiles
    __shared__ scalar_type tile_a[tile_size][tile_size];
    __shared__ scalar_type tile_b[tile_size][tile_size];

    // Calculate the row and column index for this thread in the output matrix matrix_c
    cell_idx_t row = blockIdx.y * tile_size + threadIdx.y;
    cell_idx_t col = blockIdx.x * tile_size + threadIdx.x;

    // Accumulate the result for matrix_c[row][col]
    scalar_type cell_c = 0;

    // Loop over tiles of matrix_a and matrix_b that are multiplied together
    for (cell_idx_t t = 0; t < (num_cols_a + tile_size - 1) / tile_size; ++t) {
        // Load tiles of matrix_a and matrix_b into shared memory with boundary checks
        if (row < num_rows_a && t * tile_size + threadIdx.x < num_cols_a)
            tile_a[threadIdx.y][threadIdx.x] = matrix_a[row * stride_a + t * tile_size + threadIdx.x];
        else
            tile_a[threadIdx.y][threadIdx.x] = 0;

        if (col < num_cols_b && t * tile_size + threadIdx.y < num_cols_a)
            tile_b[threadIdx.y][threadIdx.x] = matrix_b[(t * tile_size + threadIdx.y) * stride_b + col];
        else
            tile_b[threadIdx.y][threadIdx.x] = 0.0f;

        // Synchronize to ensure all data is loaded into shared memory
        __syncthreads();

#pragma unroll
        // Perform the multiplication and accumulate
        for (cell_idx_t k = 0; k < tile_size; ++k) {
            cell_c += tile_a[threadIdx.y][k] * tile_b[k][threadIdx.x];
        }

        // Synchronize to ensure all threads are done with the current tile
        __syncthreads();
    }

    // Write the result back to the output matrix matrix_c with boundary check
    if (row < num_rows_a && col < num_cols_b)
        matrix_c[row * stride_c + col] = cell_c;
}

#endif // defined(__NVCC__)

#pragma endregion CUDA

#pragma region Python bindings
#if !defined(CUPY_STARTER_TEST)

/**
 *  @brief  Router function, that unpacks Python buffers into C++ pointers and calls the appropriate
 *          backend for reductions, like `openmp_reduce` or `cuda_reduce_kernel`.
 */
template <backend_t backend_kind, typename scalar_type>
static py::object python_reduce_typed(py::buffer_info const& buf) noexcept(false) {
    if (buf.ndim != 1 || buf.strides[0] != sizeof(scalar_type))
        throw std::runtime_error("Input should be a contiguous 1D array");
    scalar_type const* ptr = reinterpret_cast<scalar_type const*>(buf.ptr);
    reduce_type<scalar_type> result;

    if constexpr (backend_kind == backend_t::openmp_k) {
        // Explicitly enable dynamic teams, as the amount of compute per thread is not uniform.
        result = openmp_reduce<scalar_type>(ptr, buf.size);
    } else if constexpr (backend_kind == backend_t::cuda_k) {
#if defined(__NVCC__)
        result = cuda_reduce<scalar_type>(ptr, buf.size);
#else
        throw std::runtime_error("CUDA backend not available");
#endif
    } else {
        throw std::runtime_error("Unsupported backend");
    }

    return py::cast(result);
}

/**
 *  @brief  Router function, used to dispatch the right type-specific pre-compiled kernel
 *          using runtime-only type information. Calls `python_reduce_typed`.
 */
template <backend_t backend_kind> static py::object python_reduce(py::array a) noexcept(false) {
    if (py::isinstance<py::array_t<float>>(a))
        return python_reduce_typed<backend_kind, float>(a.request());
    else if (py::isinstance<py::array_t<double>>(a))
        return python_reduce_typed<backend_kind, double>(a.request());
    else if (py::isinstance<py::array_t<std::int8_t>>(a))
        return python_reduce_typed<backend_kind, std::int8_t>(a.request());
    else if (py::isinstance<py::array_t<std::uint8_t>>(a))
        return python_reduce_typed<backend_kind, std::uint8_t>(a.request());
    else if (py::isinstance<py::array_t<std::int16_t>>(a))
        return python_reduce_typed<backend_kind, std::int16_t>(a.request());
    else if (py::isinstance<py::array_t<std::uint16_t>>(a))
        return python_reduce_typed<backend_kind, std::uint16_t>(a.request());
    else if (py::isinstance<py::array_t<std::int32_t>>(a))
        return python_reduce_typed<backend_kind, std::int32_t>(a.request());
    else if (py::isinstance<py::array_t<std::uint32_t>>(a))
        return python_reduce_typed<backend_kind, std::uint32_t>(a.request());
    else if (py::isinstance<py::array_t<std::int64_t>>(a))
        return python_reduce_typed<backend_kind, std::int64_t>(a.request());
    else if (py::isinstance<py::array_t<std::uint64_t>>(a))
        return python_reduce_typed<backend_kind, std::uint64_t>(a.request());

    throw std::runtime_error("Unsupported data type");
    return py::none();
}

/**
 *  @brief  Router function, that unpacks Python buffers into C++ pointers and calls the appropriate
 *          backend for matrix multiplication, like `openmp_matmul` or `cuda_matmul_kernel`.
 */
template <backend_t backend_kind, typename scalar_type>
static py::array python_matmul_typed(py::buffer_info const& buffer_a, py::buffer_info const& buffer_b,
                                     std::size_t tile_size) {

    if (buffer_a.ndim != 2 || buffer_b.ndim != 2)
        throw std::runtime_error("Both tensors must be rank-2");
    if (buffer_a.shape[1] != buffer_b.shape[0])
        throw std::runtime_error("Inner dimensions must match");
    auto ptr_a = reinterpret_cast<scalar_type const*>(buffer_a.ptr);
    auto ptr_b = reinterpret_cast<scalar_type const*>(buffer_b.ptr);
    auto num_rows_a = static_cast<cell_idx_t>(buffer_a.shape[0]);
    auto num_cols_a = static_cast<cell_idx_t>(buffer_a.shape[1]);
    auto num_cols_b = static_cast<cell_idx_t>(buffer_b.shape[1]);
    auto stride_a = static_cast<cell_idx_t>(buffer_a.strides[0] / sizeof(scalar_type));
    auto stride_b = static_cast<cell_idx_t>(buffer_b.strides[0] / sizeof(scalar_type));

    // Allocate NumPy array for the result
    auto tensor_c = py::array_t<scalar_type>({num_rows_a, num_cols_b});
    auto buffer_c = tensor_c.request();
    auto ptr_c = reinterpret_cast<scalar_type*>(buffer_c.ptr);
    auto stride_c = static_cast<cell_idx_t>(buffer_c.strides[0] / sizeof(scalar_type));

    // Call the appropriate kernel based on the backend
    using kernel_t = void (*)(scalar_type const*, scalar_type const*, scalar_type*, cell_idx_t, cell_idx_t, cell_idx_t,
                              cell_idx_t, cell_idx_t, cell_idx_t);

    if constexpr (backend_kind == backend_t::openmp_k) {
        // Explicitly disable dynamic teams, as the amount of compute per thread is uniform.
        kernel_t kernel = nullptr;
        switch (tile_size) {
        case 4: kernel = &openmp_matmul<scalar_type, 4>;
        case 8: kernel = &openmp_matmul<scalar_type, 8>;
        case 16: kernel = &openmp_matmul<scalar_type, 16>;
        case 32: kernel = &openmp_matmul<scalar_type, 32>;
        case 64: kernel = &openmp_matmul<scalar_type, 64>;
        case 128: kernel = &openmp_matmul<scalar_type, 128>;
        default: throw std::runtime_error("Unsupported tile size");
        }
        kernel(ptr_a, ptr_b, ptr_c, num_rows_a, num_cols_b, num_cols_a, stride_a, stride_b, stride_c);

    } else if constexpr (backend_kind == backend_t::cuda_k) {
#if defined(__NVCC__)
        votes_count_t* ptr_c_cuda = nullptr;
        cudaError_t error;
        error = cudaMallocManaged(&ptr_c_cuda, num_rows_a * num_cols_b * sizeof(scalar_type));
        if (error != cudaSuccess)
            throw std::runtime_error("Failed to allocate memory on device");
        cudaMemset(ptr_c_cuda, 0, num_rows_a * num_cols_b * sizeof(scalar_type));

        // Synchronize to ensure all CUDA operations are complete
        error = cudaDeviceSynchronize();
        if (error != cudaSuccess) {
            cudaFree(ptr_c_cuda);
            throw std::runtime_error("CUDA operations did not complete successfully");
        }

        // Launch the CUDA kernel
        dim3 block_size(tile_size, tile_size);
        dim3 grid_size((num_cols_b + tile_size - 1) / tile_size, (num_rows_a + tile_size - 1) / tile_size);
        cuda_matmul_kernel<scalar_type, tile_size><<<grid_size, block_size>>>(
            ptr_a, ptr_b, ptr_c_cuda, num_rows_a, num_cols_b, num_cols_a, stride_a, stride_b, stride_c);

        // Synchronize to ensure all CUDA operations are complete
        error = cudaDeviceSynchronize();
        if (error != cudaSuccess) {
            cudaFree(ptr_c_cuda);
            throw std::runtime_error("CUDA operations did not complete successfully");
        }

        // Copy data from the GPU to the NumPy array
        error = cudaMemcpy(ptr_c, ptr_c_cuda, num_candidates * num_candidates * sizeof(votes_count_t),
                           cudaMemcpyDeviceToHost);
        if (error != cudaSuccess) {
            cudaFree(ptr_c_cuda);
            throw std::runtime_error("Failed to copy data from device to host");
        }

        // Synchronize to ensure all CUDA transfers are complete
        error = cudaDeviceSynchronize();
        if (error != cudaSuccess) {
            cudaFree(ptr_c_cuda);
            throw std::runtime_error("CUDA transfers did not complete successfully");
        }

        // Free the GPU memory
        error = cudaFree(ptr_c_cuda);
        if (error != cudaSuccess)
            throw std::runtime_error("Failed to free memory on device");
#else
        throw std::runtime_error("CUDA backend not available");
#endif
    } else {
        throw std::runtime_error("Unsupported backend");
    }

    return tensor_c;
}

/**
 *  @brief  Router function, used to dispatch the right type-specific pre-compiled kernel
 *          using runtime-only type information. Calls `python_matmul_typed`.
 */
template <backend_t backend_kind>
static py::array python_matmul(py::array a, py::array b, std::size_t tile_size) noexcept(false) {

    if (py::isinstance<py::array_t<float>>(a))
        return python_matmul_typed<backend_kind, float>(a.request(), b.request(), tile_size);
    else if (py::isinstance<py::array_t<double>>(a))
        return python_matmul_typed<backend_kind, double>(a.request(), b.request(), tile_size);
    else if (py::isinstance<py::array_t<std::int8_t>>(a))
        return python_matmul_typed<backend_kind, std::int8_t>(a.request(), b.request(), tile_size);
    else if (py::isinstance<py::array_t<std::uint8_t>>(a))
        return python_matmul_typed<backend_kind, std::uint8_t>(a.request(), b.request(), tile_size);
    else if (py::isinstance<py::array_t<std::int16_t>>(a))
        return python_matmul_typed<backend_kind, std::int16_t>(a.request(), b.request(), tile_size);
    else if (py::isinstance<py::array_t<std::uint16_t>>(a))
        return python_matmul_typed<backend_kind, std::uint16_t>(a.request(), b.request(), tile_size);
    else if (py::isinstance<py::array_t<std::int32_t>>(a))
        return python_matmul_typed<backend_kind, std::int32_t>(a.request(), b.request(), tile_size);
    else if (py::isinstance<py::array_t<std::uint32_t>>(a))
        return python_matmul_typed<backend_kind, std::uint32_t>(a.request(), b.request(), tile_size);
    else if (py::isinstance<py::array_t<std::int64_t>>(a))
        return python_matmul_typed<backend_kind, std::int64_t>(a.request(), b.request(), tile_size);
    else if (py::isinstance<py::array_t<std::uint64_t>>(a))
        return python_matmul_typed<backend_kind, std::uint64_t>(a.request(), b.request(), tile_size);

    throw std::runtime_error("Unsupported data type");
    return py::none();
}

PYBIND11_MODULE(cupy_starter, m) {

    std::signal(SIGINT, signal_handler);

    // Let's show how to wrap `void` functions for basic logging
    m.def("log_gpus", []() {
#if defined(__NVCC__)
        int device_count;
        cudaDeviceProp device_props;
        cudaError_t error = cudaGetDeviceCount(&device_count);
        if (error != cudaSuccess)
            throw std::runtime_error("Failed to get device count");
        for (int i = 0; i < device_count; i++) {
            error = cudaGetDeviceProperties(&device_props, i);
            if (error != cudaSuccess)
                throw std::runtime_error("Failed to get device properties");
            std::printf("Device %d: %s\n", i, device_props.name);
            std::printf("\tSMs: %d\n", device_props.multiProcessorCount);
            std::printf("\tGlobal mem: %.2fGB\n",
                        static_cast<float>(device_props.totalGlobalMem) / (1024 * 1024 * 1024));
            std::printf("\tCUDA Cap: %d.%d\n", device_props.major, device_props.minor);
        }
#else
        throw std::runtime_error("No CUDA devices available\n");
#endif
    });

    // This is how we could have used `thrust::` for higher-level operations
    m.def("reduce_openmp", &python_reduce<backend_t::openmp_k>);
    m.def("matmul_openmp", &python_matmul<backend_t::openmp_k>, py::arg("a"), py::arg("b"), py::kw_only(),
          py::arg("tile_size") = 16);

    m.def("reduce_cuda", &python_reduce<backend_t::cuda_k>);
    m.def("matmul_cuda", &python_matmul<backend_t::cuda_k>, py::arg("a"), py::arg("b"), py::kw_only(),
          py::arg("tile_size") = 16);
}

#endif // !defined(CUPY_STARTER_TEST)
#pragma endregion Python bindings

#if defined(CUPY_STARTER_TEST)

int main() {

    std::size_t num_candidates = 256;
    std::vector<votes_count_t> preferences(num_candidates * num_candidates);
    std::generate(preferences.begin(), preferences.end(),
                  [=]() { return static_cast<votes_count_t>(std::rand() % num_candidates); });

    std::vector<votes_count_t> graph(num_candidates * num_candidates);
    matmul_openmp<64>(preferences.data(), num_candidates, num_candidates, graph.data());

    return 0;
}

#endif // defined(CUPY_STARTER_TEST)