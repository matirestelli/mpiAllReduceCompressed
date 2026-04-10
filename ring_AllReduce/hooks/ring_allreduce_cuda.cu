/*
 * ring_allreduce_cuda.cu
 *
 * PyTorch C++/CUDA extension implementing ring allreduce using
 * CUDA-aware MPI. This is a direct port of the advisor's C++ MPI code
 * with two changes:
 *
 *   1. malloc  → cudaMalloc   (tmpbuf lives on GPU)
 *   2. memcpy  → cudaMemcpy DeviceToDevice
 *   3. MPI_Reduce_local → custom CUDA kernel (CPU cannot read GPU memory)
 *   4. MPI_Allgatherv   → unchanged (CUDA-aware MPI handles GPU pointers)
 *
 * Everything else — MPI_Isend, MPI_Irecv, MPI_Waitall — is identical
 * to the advisor's code because CUDA-aware MPI handles GPU pointers
 * transparently.
 *
 * Build:
 *   cd hooks && python setup.py install
 *
 * Requires:
 *   - CUDA-aware MPI (OpenMPI built with --with-cuda or MPICH with CUDA)
 *   - PyTorch with CUDA support
 */

#include <torch/extension.h>
#include <mpi.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <cstdlib>
#include <cstring>


// ── CUDA kernel: elementwise add (replaces MPI_Reduce_local) ─────────────────
//
// MPI_Reduce_local operates on CPU memory. Since our buffers are on GPU
// we replace it with this simple elementwise add kernel.
//
// Equivalent to: dst[i] += src[i]  for i in [0, n)

__global__ void elementwise_add_kernel(
    float* __restrict__ dst,
    const float* __restrict__ src,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] += src[idx];
    }
}

__global__ void elementwise_add_kernel_double(
    double* __restrict__ dst,
    const double* __restrict__ src,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] += src[idx];
    }
}

static void launch_elementwise_add(
    void* dst, const void* src, int n, torch::ScalarType dtype
) {
    const int threads = 256;
    const int blocks  = (n + threads - 1) / threads;

    if (dtype == torch::kFloat32) {
        elementwise_add_kernel<<<blocks, threads>>>(
            (float*)dst, (const float*)src, n
        );
    } else if (dtype == torch::kFloat64) {
        elementwise_add_kernel_double<<<blocks, threads>>>(
            (double*)dst, (const double*)src, n
        );
    } else {
        throw std::runtime_error(
            "ring_allreduce_cuda: unsupported dtype (only float32/float64)"
        );
    }
    // synchronize the default stream (stream 0) only — cheaper than
    // cudaDeviceSynchronize() which drains ALL streams on the device.
    // We use stream 0 (not a custom stream) because Cray GTL requires
    // MPI GPU transfers to be visible on the default stream.
    cudaStreamSynchronize(0);
}


// ── Helper: map torch dtype → MPI_Datatype ────────────────────────────────────

static MPI_Datatype torch_dtype_to_mpi(torch::ScalarType dtype) {
    switch (dtype) {
        case torch::kFloat32: return MPI_FLOAT;
        case torch::kFloat64: return MPI_DOUBLE;
        case torch::kFloat16: return MPI_SHORT;   // treat as raw bytes
        case torch::kInt32:   return MPI_INT;
        case torch::kInt64:   return MPI_LONG;
        default:
            throw std::runtime_error(
                "ring_allreduce_cuda: unsupported tensor dtype"
            );
    }
}


// ── Core ring allreduce (GPU-aware MPI) ───────────────────────────────────────
//
// Direct port of advisor's MPICH_Allreduce_ring with GPU adaptations.
// sendbuf == recvbuf (in-place, same as MPI_IN_PLACE).

static void ring_allreduce_inplace(
    char*        buf,          // GPU pointer (both input and output)
    int          count,        // number of elements
    MPI_Datatype datatype,     // MPI datatype
    torch::ScalarType dtype,   // torch dtype (for CUDA kernel)
    int          extent,       // bytes per element
    MPI_Comm     comm
) {
    int rank, nranks;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nranks);

    if (nranks == 1) return;

    // ── Compute chunk sizes and displacements (same as advisor's code) ───
    int* cnts   = (int*)malloc(nranks * sizeof(int));
    int* displs = (int*)malloc(nranks * sizeof(int));

    for (int i = 0; i < nranks; i++) cnts[i] = 0;

    int total_count = 0;
    for (int i = 0; i < nranks; i++) {
        cnts[i] = (count + nranks - 1) / nranks;
        if (total_count + cnts[i] > count) {
            cnts[i] = count - total_count;
            break;
        } else {
            total_count += cnts[i];
        }
    }

    displs[0] = 0;
    for (int i = 1; i < nranks; i++)
        displs[i] = displs[i - 1] + cnts[i - 1];

    // ── Allocate tmpbuf on GPU (replaces advisor's malloc) ────────────────
    void* tmpbuf = nullptr;
    cudaError_t cuda_err = cudaMalloc(&tmpbuf, count * extent);
    if (cuda_err != cudaSuccess) {
        free(cnts); free(displs);
        throw std::runtime_error(
            std::string("cudaMalloc failed: ") + cudaGetErrorString(cuda_err)
        );
    }

    int src = (nranks + rank - 1) % nranks;
    int dst = (rank + 1) % nranks;

    // Flush all pending GPU work before MPI touches the buffer.
    // This ensures the gradient values computed by PyTorch's backward
    // pass are fully written to GPU memory before MPI reads them.
    cudaDeviceSynchronize();

    MPI_Request reqs[2];
    int mpi_err;

    // ── Phase 1: Reduce-Scatter ───────────────────────────────────────────
    for (int i = 0; i < nranks - 1; i++) {
        int recv_rank = (nranks + rank - 2 - i) % nranks;
        int send_rank = (nranks + rank - 1 - i) % nranks;

        // post recv first 
        mpi_err = MPI_Irecv(
            tmpbuf,
            cnts[recv_rank], datatype,
            src, i, comm, &reqs[0]
        );
        if (mpi_err != MPI_SUCCESS) goto cleanup;

        mpi_err = MPI_Isend(
            buf + (long)displs[send_rank] * extent,
            cnts[send_rank], datatype,
            dst, i, comm, &reqs[1]
        );
        if (mpi_err != MPI_SUCCESS) goto cleanup;

        MPI_Waitall(2, reqs, MPI_STATUSES_IGNORE);

        // GPU elementwise add (replaces MPI_Reduce_local)
        launch_elementwise_add(
            buf + (long)displs[recv_rank] * extent,
            tmpbuf,
            cnts[recv_rank],
            dtype
        );
    }

    // ── Phase 2: Allgatherv ───────────────────────────────────────────────
    mpi_err = MPI_Allgatherv(
        MPI_IN_PLACE, -1, MPI_DATATYPE_NULL,
        buf, cnts, displs, datatype, comm
    );

cleanup:
    cudaFree(tmpbuf);
    free(cnts);
    free(displs);

    if (mpi_err != MPI_SUCCESS) {
        char err_str[MPI_MAX_ERROR_STRING];
        int err_len;
        MPI_Error_string(mpi_err, err_str, &err_len);
        throw std::runtime_error(
            std::string("MPI error in ring_allreduce: ") + err_str
        );
    }
}


// ── Python-visible entry point ────────────────────────────────────────────────

torch::Tensor ring_allreduce_cuda(torch::Tensor tensor) {
    // Validate
    TORCH_CHECK(tensor.is_cuda(), "ring_allreduce_cuda: tensor must be on GPU");
    TORCH_CHECK(tensor.is_contiguous(), "ring_allreduce_cuda: tensor must be contiguous");

    // Explicitly set the CUDA device to match the tensor's device.
    // Cray GTL requires the active CUDA device to match the memory being
    // passed to MPI — without this, GTL may try to access memory on the
    // wrong device and crash with "illegal memory access".
    int device_id = tensor.device().index();
    cudaSetDevice(device_id);

    char*        buf      = (char*)tensor.data_ptr();
    int          count    = (int)tensor.numel();
    torch::ScalarType dtype = tensor.scalar_type();
    MPI_Datatype mpi_dtype = torch_dtype_to_mpi(dtype);
    int          extent;
    MPI_Type_size(mpi_dtype, &extent);

    ring_allreduce_inplace(buf, count, mpi_dtype, dtype, extent, MPI_COMM_WORLD);

    return tensor;
}


// ── pybind11 module ───────────────────────────────────────────────────────────

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Ring AllReduce via CUDA-aware MPI (GPU tensors, in-place SUM)";
    m.def(
        "ring_allreduce",
        &ring_allreduce_cuda,
        "In-place ring allreduce SUM on a GPU tensor using CUDA-aware MPI"
    );
}xs