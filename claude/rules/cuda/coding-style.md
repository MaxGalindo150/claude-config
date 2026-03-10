> This file extends [common/coding-style.md](../common/coding-style.md) with CUDA-specific coding standards and idioms.
> For deep optimization patterns and profiling, see the [cuda-kernel-optimization skill](../../skills/cuda-kernel-optimization/SKILL.md).

# CUDA Coding Style

Standards for CUDA C++ source files (`.cu`, `.cuh`). These rules override or extend the common style where GPU idioms differ from general C++.

---

## File Organization

```
project/
├── include/
│   └── mylib/
│       ├── mykernel.cuh      # device declarations, __device__ helpers
│       └── mykernel.h        # host-facing API (no CUDA types)
└── src/
    ├── mykernel.cu           # kernel definitions + launch wrappers
    └── host_code.cpp         # pure host code, no .cu extension
```

- **`.cuh`** — headers with `__device__`/`__global__` declarations or device-only templates.
- **`.cu`** — translation units compiled by `nvcc`. Keep as few as needed.
- **`.cpp`** — host-only code; do not force `nvcc` to compile it.
- Never expose raw CUDA types (`dim3`, `cudaStream_t`) in public headers consumed by non-CUDA translation units.

---

## Naming Conventions

| Entity | Convention | Example |
|--------|-----------|---------|
| Kernels (`__global__`) | `verbNounKernel` | `reduceFloatKernel` |
| Device helpers (`__device__`) | camelCase prefixed `d_` or plain | `warpReduce`, `d_clamp` |
| Host launch wrappers | same name without `Kernel` suffix | `reduceFloat(...)` |
| Device pointers | `d_` prefix | `d_input`, `d_output` |
| Host pinned buffers | `h_` prefix | `h_input` |
| Shared memory arrays | `s_` prefix | `s_tile` |
| Tile/block size constants | `BLOCK_SIZE`, `TILE_M` | all caps |

---

## Error Handling (mandatory)

Every CUDA API call must be wrapped. No silent failures.

```cuda
// cuda_check.h — include in every .cu file
#pragma once
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t _err = (call);                                           \
        if (_err != cudaSuccess) {                                           \
            fprintf(stderr, "[CUDA] %s:%d error: %s\n",                     \
                    __FILE__, __LINE__, cudaGetErrorString(_err));           \
            std::exit(EXIT_FAILURE);                                         \
        }                                                                    \
    } while (0)

// After every kernel launch:
#define CUDA_LAUNCH_CHECK()                                                  \
    do {                                                                     \
        CUDA_CHECK(cudaGetLastError());                                      \
    } while (0)
```

```cuda
// Usage
CUDA_CHECK(cudaMalloc(&d_buf, size));
myKernel<<<grid, block>>>(d_buf);
CUDA_LAUNCH_CHECK();
// cudaDeviceSynchronize() only during debug/test — remove from hot paths
```

---

## Memory Management

- Prefer **RAII wrappers** over raw `cudaMalloc`/`cudaFree`:

```cuda
struct DeviceBuffer {
    void* ptr = nullptr;
    explicit DeviceBuffer(size_t bytes) { CUDA_CHECK(cudaMalloc(&ptr, bytes)); }
    ~DeviceBuffer() { if (ptr) cudaFree(ptr); }
    // Non-copyable
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};
```

- For production use, prefer `thrust::device_vector<T>` which handles all of the above.
- Never call `cudaMalloc` inside a kernel (`__global__` or `__device__`).
- Always pair `cudaMallocHost` with `cudaFreeHost` (not `free()`).

---

## Kernel Design Rules

1. **No dynamic allocation inside kernels.** No `new`, `malloc`, `printf` in hot paths.
2. **Explicit `__syncthreads()` placement.** Every shared-memory write that is read by another thread must be followed by a sync before the read.
3. **`__syncthreads()` in uniform control flow only.** Never inside a divergent branch.
4. **Use `__launch_bounds__` for kernels with tight register budgets.**
5. **Mark read-only pointer arguments `const __restrict__`.** Enables compiler to use `__ldg` (texture cache path on sm_35+).

```cuda
__global__ void myKernel(const float* __restrict__ in,
                          float* __restrict__ out,
                          int N) { ... }
```

6. **Prefer `float` over `double` unless precision is required.** FP64 throughput is 1/32× on consumer GPUs vs FP32.

---

## Launch Configuration

- Block size must be a **multiple of 32** (warp size). Prefer 128, 256, or 512.
- Use `cudaOccupancyMaxPotentialBlockSize` for tuning, not hardcoded constants, unless benchmarked.
- Always guard against out-of-bounds:

```cuda
__global__ void kernel(float* data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return; // guard
    data[idx] = process(data[idx]);
}
```

---

## Host/Device Code Separation

- Business logic lives in `.cpp`/`.h` — no CUDA headers.
- GPU operations live in `.cu`/`.cuh` — exposed through a plain C++ API.
- Never `#include <cuda_runtime.h>` from a `.cpp` or `.h` file unless absolutely necessary. Use a thin wrapper instead.

```cpp
// mykernel.h — pure C++ public API
void launchMyKernel(float* d_out, const float* d_in, int N, cudaStream_t stream);
```

---

## Compiler Flags (CMakeLists.txt)

```cmake
find_package(CUDAToolkit REQUIRED)

target_compile_options(mytarget PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --use_fast_math           # fused ops, fast intrinsics (check precision!)
        -Xptxas -v               # print register/smem usage
        --generate-line-info     # enable Nsight source correlation
        -arch=sm_86              # target SM — set to your minimum supported arch
    >
)
```

For multi-arch (distributable binaries):
```cmake
set_target_properties(mytarget PROPERTIES
    CUDA_ARCHITECTURES "70;80;86;90"
)
```

---

## CMake Project Layout

```cmake
cmake_minimum_required(VERSION 3.25)
project(myproject LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

add_library(mylib STATIC src/mykernel.cu)
target_include_directories(mylib PUBLIC include)
target_link_libraries(mylib PUBLIC CUDA::cudart)
```

---

## Testing CUDA Code

- Unit-test **host-side launch wrappers** with GoogleTest; use small, reproducible inputs.
- Compare GPU output against a CPU reference implementation.
- Use `cudaMemcpy` → host array → `EXPECT_NEAR` (account for FP rounding).
- For performance regression tests, measure kernel time with `cudaEvent_t`:

```cuda
cudaEvent_t start, stop;
CUDA_CHECK(cudaEventCreate(&start));
CUDA_CHECK(cudaEventCreate(&stop));

CUDA_CHECK(cudaEventRecord(start));
myKernel<<<grid, block>>>(args);
CUDA_CHECK(cudaEventRecord(stop));
CUDA_CHECK(cudaEventSynchronize(stop));

float ms = 0.f;
CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
```

---

## Common Pitfalls

| Pitfall | Fix |
|---------|-----|
| Missing `__syncthreads()` after shared mem write | Add sync before any cross-thread read |
| Sync inside divergent branch | Restructure: sync must be reached by all threads |
| Forgetting `cudaDeviceSynchronize()` before timing | Use `cudaEvent_t` instead |
| AoS layout → uncoalesced reads | Refactor to SoA |
| `cudaFree` on host pointer | Track ownership; use RAII |
| No out-of-bounds guard | Always check `idx < N` |
| `double` on consumer GPU | Use `float` unless precision justified |
| Launching with `blockDim` not a multiple of 32 | Round up to next multiple of warp size |
