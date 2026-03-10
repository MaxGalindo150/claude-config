---
name: cuda-kernel-optimization
description: CUDA kernel development and GPU optimization patterns — memory hierarchy, occupancy tuning, coalescing, shared memory tiling, warp-level ops, and profiling with Nsight Compute. Use when writing or optimizing CUDA C++ kernels.
origin: local
---

# CUDA Kernel Optimization

Deep reference for writing high-performance CUDA kernels in C++. Covers the GPU execution model, memory hierarchy, occupancy, access patterns, and systematic profiling.

## When to Use

- Writing new `.cu`/`.cuh` CUDA kernels from scratch
- Profiling and optimizing existing GPU kernels
- Diagnosing memory bottlenecks, warp divergence, or low occupancy
- Implementing GPU parallel primitives (reductions, scans, matrix ops)
- Choosing between CUDA libraries (cuBLAS, Thrust) vs custom kernels

### When NOT to Use

- CPU-only code or OpenCL/Metal/Vulkan compute (adapt selectively)
- High-level frameworks that abstract CUDA (PyTorch custom ops are adjacent — most concepts apply but the workflow differs)

---

## 1. Execution Model

### Thread Hierarchy

```
Grid
└── Blocks (up to 3D)
    └── Warps (32 threads, scheduled together)
        └── Threads
```

| Unit | Max size | Notes |
|------|----------|-------|
| Thread block | 1024 threads | All dimensions combined |
| Grid | 2³¹-1 per dim (x), 65535 (y/z) | |
| Warp | 32 threads | Lockstep execution unit |

### Key principle

The GPU hides latency through **massive multithreading** — always have enough warps in flight to cover memory latency (~300–700 cycles for global memory).

```cuda
// Minimal kernel shape: choose blockDim first, then gridDim
constexpr int BLOCK = 256;
int grid = (N + BLOCK - 1) / BLOCK;
kernel<<<grid, BLOCK>>>(args...);
```

---

## 2. Memory Hierarchy

| Memory | Scope | Latency | Size | Bandwidth |
|--------|-------|---------|------|-----------|
| Registers | Thread | 1 cycle | ~255 per thread | — |
| Shared memory | Block | ~20–30 cycles | 48–228 KB/SM (arch-dep) | ~20 TB/s |
| L1 cache | SM | ~30 cycles | 32–128 KB (shared with smem) | — |
| L2 cache | Device | ~100–200 cycles | 4–50 MB (arch-dep) | ~4 TB/s |
| Global memory (DRAM) | Device | ~300–700 cycles | GBs | ~900 GB/s (H100) |
| Constant memory | Device (cached) | 1 cycle (hit) | 64 KB | — |
| Texture memory | Device (cached) | Low (2D locality) | — | — |

### Shared Memory Usage Pattern

```cuda
__global__ void tiledMatMul(const float* A, const float* B, float* C,
                             int N) {
    constexpr int TILE = 32;
    __shared__ float tileA[TILE][TILE];
    __shared__ float tileB[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float acc = 0.f;

    for (int t = 0; t < N / TILE; ++t) {
        tileA[threadIdx.y][threadIdx.x] = A[row * N + t * TILE + threadIdx.x];
        tileB[threadIdx.y][threadIdx.x] = B[(t * TILE + threadIdx.y) * N + col];
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; ++k)
            acc += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        __syncthreads(); // prevent overwrite before all threads finish
    }
    C[row * N + col] = acc;
}
```

---

## 3. Memory Access Patterns

### Global Memory Coalescing

Threads in a warp should access **consecutive 128-byte aligned** addresses in a single transaction.

```cuda
// GOOD — coalesced: thread i accesses element i
float val = data[blockIdx.x * blockDim.x + threadIdx.x];

// BAD — strided: each thread jumps by `stride`
float val = data[threadIdx.x * stride]; // multiple transactions
```

### Shared Memory Bank Conflicts

Shared memory has 32 banks (4-byte width). Threads in a warp accessing the same bank (different addresses) serialize.

```cuda
// BAD — column-major access causes bank conflicts
float val = tile[threadIdx.x][threadIdx.y]; // all threads hit bank = threadIdx.x % 32

// FIX — pad the shared array by 1
__shared__ float tile[32][33]; // +1 padding breaks conflict pattern
```

### Unified Memory (CUDA 6+)

```cuda
float* data;
cudaMallocManaged(&data, N * sizeof(float));
// Accessible from host and device; managed by runtime
// Prefetch to avoid page faults during kernel:
cudaMemPrefetchAsync(data, N * sizeof(float), device_id);
```

---

## 4. Occupancy Tuning

**Occupancy** = active warps / max warps per SM. Higher occupancy enables better latency hiding.

Limiters (whichever binds first):
1. **Registers per thread** — spills to local memory if exceeded
2. **Shared memory per block**
3. **Block size** — must be multiple of 32 (warp size)

### Query at runtime

```cuda
int minGridSize, blockSize;
cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, myKernel, 0, 0);

int gridSize = (N + blockSize - 1) / blockSize;
myKernel<<<gridSize, blockSize>>>(args);
```

### Limit register usage

```cuda
// Force compiler to limit registers (may increase spilling — profile first)
__launch_bounds__(256 /*maxThreadsPerBlock*/, 4 /*minBlocksPerSM*/)
__global__ void myKernel(...) { ... }
```

---

## 5. Warp-Level Operations

### Avoid Divergence

```cuda
// BAD — threads in the same warp take different paths → serialized
if (threadIdx.x % 2 == 0) { doA(); } else { doB(); }

// BETTER — divergence across warps is free (different warps run independently)
// Ensure condition is uniform within a warp:
if (threadIdx.x / 32 == 0) { doA(); } // entire warp 0 takes this branch
```

### Warp Intrinsics (sm_70+)

```cuda
// Warp reduction — no shared memory needed
float val = /* per-thread value */;
for (int offset = 16; offset > 0; offset >>= 1)
    val += __shfl_down_sync(0xffffffff, val, offset);
// Thread 0 of each warp holds the warp sum

// Warp vote
bool pred = (val > threshold);
unsigned mask = __ballot_sync(0xffffffff, pred); // bitmask of threads where pred=true
int count = __popc(mask);
```

---

## 6. Common Parallel Primitives

### Parallel Reduction

```cuda
template <int BLOCK_SIZE>
__global__ void reduce(const float* in, float* out, int N) {
    __shared__ float smem[BLOCK_SIZE];
    int tid = threadIdx.x;
    int i   = blockIdx.x * BLOCK_SIZE * 2 + tid;

    smem[tid] = (i < N ? in[i] : 0.f) + (i + BLOCK_SIZE < N ? in[i + BLOCK_SIZE] : 0.f);
    __syncthreads();

    // Sequential addressing — no bank conflicts
    for (int s = BLOCK_SIZE / 2; s > 32; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    // Warp-level unrolling (no sync needed within a warp)
    if (tid < 32) {
        smem[tid] += smem[tid + 32];
        smem[tid] += __shfl_down_sync(0xffffffff, smem[tid], 16);
        smem[tid] += __shfl_down_sync(0xffffffff, smem[tid], 8);
        smem[tid] += __shfl_down_sync(0xffffffff, smem[tid], 4);
        smem[tid] += __shfl_down_sync(0xffffffff, smem[tid], 2);
        smem[tid] += __shfl_down_sync(0xffffffff, smem[tid], 1);
    }
    if (tid == 0) out[blockIdx.x] = smem[0];
}
```

### Prefix Scan (Blelloch)

Use [Thrust](https://thrust.github.io/) for production:
```cpp
#include <thrust/scan.h>
thrust::inclusive_scan(d_in, d_in + N, d_out);
```

---

## 7. Streams and Concurrency

```cuda
cudaStream_t s1, s2;
cudaStreamCreate(&s1);
cudaStreamCreate(&s2);

// Overlap kernel execution with H2D copy
cudaMemcpyAsync(d_buf, h_buf, size, cudaMemcpyHostToDevice, s1);
kernelA<<<grid, block, 0, s1>>>(d_buf);
kernelB<<<grid, block, 0, s2>>>(d_other); // runs concurrently if SM resources allow

cudaStreamSynchronize(s1);
cudaStreamSynchronize(s2);
cudaStreamDestroy(s1);
cudaStreamDestroy(s2);
```

---

## 8. Error Handling

```cuda
// Macro — use in all CUDA API calls
#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = (call);                                           \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d — %s\n",                  \
                    __FILE__, __LINE__, cudaGetErrorString(err));           \
            std::exit(EXIT_FAILURE);                                        \
        }                                                                   \
    } while (0)

// Kernel launch errors are async — always sync and check after launch
myKernel<<<grid, block>>>(args);
CUDA_CHECK(cudaGetLastError());
CUDA_CHECK(cudaDeviceSynchronize()); // remove in production hot paths
```

---

## 9. Profiling Workflow

### Nsight Compute (ncu)

```bash
# Full kernel profile
ncu --set full -o report ./my_app

# Target one kernel
ncu --kernel-name myKernel --set full ./my_app

# Key metrics to check first
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,\
l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum,\
sm__warps_active.avg.pct_of_peak_sustained_active \
./my_app
```

### Key Metrics and What They Tell You

| Metric | Target | If Low → |
|--------|--------|----------|
| SM throughput % | >80% | Low occupancy or memory bound |
| Global load efficiency | >80% | Uncoalesced accesses |
| Shared memory bank conflict rate | 0 | Pad shared arrays |
| Warp execution efficiency | >85% | Warp divergence |
| Achieved occupancy | >50% (app-dep) | Register/smem pressure |
| L2 hit rate | >50% | Poor data reuse, increase tiling |

### Nsight Systems (nsys) — timeline view

```bash
nsys profile --trace=cuda,nvtx ./my_app
nsys-ui report.nsys-rep  # open GUI
```

---

## 10. Optimization Checklist

Work through these in order — fix the biggest bottleneck first (Amdahl's law).

- [ ] **Profile first.** Never optimize blind. Use `ncu` to identify the limiter.
- [ ] **Coalesce global memory.** Structure-of-arrays (SoA) over array-of-structures (AoS).
- [ ] **Use shared memory as a scratchpad.** Tile loops that reuse data.
- [ ] **Eliminate bank conflicts.** Pad shared arrays by 1 element.
- [ ] **Maximize occupancy.** Use `cudaOccupancyMaxPotentialBlockSize`.
- [ ] **Minimize register pressure.** Check `--ptxas-options=-v` in compiler output.
- [ ] **Remove warp divergence.** Restructure branches to be warp-uniform.
- [ ] **Use warp intrinsics.** Replace shared-memory reductions with `__shfl_*` where possible.
- [ ] **Overlap compute and I/O.** Use streams + `cudaMemcpyAsync`.
- [ ] **Prefer libraries.** cuBLAS/cuDNN/Thrust beat hand-rolled for standard ops.
- [ ] **Use `#pragma unroll`.** For small, fixed-bound inner loops.
- [ ] **Avoid dynamic memory allocation in kernels.** No `new`/`malloc` inside `__global__`.

---

## 11. Architecture Quick Reference

| GPU Family | Arch | Compute Cap | Key Feature |
|-----------|------|-------------|-------------|
| Volta | V100 | 7.0 | Tensor Cores (FP16), NVLink |
| Turing | T4/RTX20 | 7.5 | INT8 Tensor Cores, RT cores |
| Ampere | A100/RTX30 | 8.0/8.6 | BF16, TF32, 3rd-gen Tensor Cores |
| Hopper | H100 | 9.0 | FP8, Transformer Engine, NVLink4 |
| Ada Lovelace | RTX40 | 8.9 | FP8, DLSS 3 |

---

## 12. Useful Libraries

| Library | Purpose |
|---------|---------|
| **Thrust** | STL-like parallel algorithms (sort, scan, reduce) |
| **cuBLAS** | Dense linear algebra (GEMM, TRSM…) |
| **cuDNN** | Deep learning primitives (conv, norm, attention) |
| **cuSPARSE** | Sparse matrix operations |
| **cuFFT** | Fast Fourier Transforms |
| **CUTLASS** | Template library for custom GEMM/conv |
| **CUB** | Low-level building blocks (warp/block/device primitives) |

---

## References

- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [Nsight Compute Documentation](https://docs.nvidia.com/nsight-compute/)
- [CUTLASS](https://github.com/NVIDIA/cutlass)
- [CUB](https://nvlabs.github.io/cub/)
