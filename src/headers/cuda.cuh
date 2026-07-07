#include <cstdio>
#include <cuda_runtime.h>
#include <array>
#include <iostream>
#include <cassert>

#include "common.hpp"

// constants
__constant__ float gpu_env_dt;
__constant__ float gpu_max_x;
__constant__ float gpu_max_y;

// bad every thread in warp will reach diff index everything will be serialized
// __constant__ float particle_radius[population_size * env_num_particles];

// error checking
static const char *_cudaGetErrorEnum(cudaError_t error) {
    return cudaGetErrorName(error);
}
template <typename T>
void check(T result, char const *const func, const char *const file,
           int const line) {
  if (result) {
    fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line,
            static_cast<unsigned int>(result), _cudaGetErrorEnum(result), func);
    exit(EXIT_FAILURE);
  }
}

#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)

// kernels
const int block_sz = env_num_particles;
const int grid_sz = population_size;

__global__ void big_kernel(GPU_unified_mem gpu_mem);

__global__ void first_half_step(GPU_unified_mem gpu_mem);

__global__ void second_half_step(GPU_unified_mem gpu_mem);
__device__ void handle_boundary(GPU_unified_mem gpu_mem, int idx);
