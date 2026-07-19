#include <cstdio>
#include <cuda_runtime.h>
#include <array>
#include <iostream>
#include <cassert>

#include <cub/cub.cuh>
#include "common.hpp"

// constants
//* constexpr and __constant__ together doesnt make sense 
//* it handle by compiler why store in runtime
//__constant__ constexpr float gpu_env_dt;

constexpr float gpu_env_dt = env_dt;
constexpr float gpu_max_x = max_x;
constexpr float gpu_max_y = max_y;
constexpr float gpu_restitution = restitution;
constexpr float gpu_coefficient_friction = coefficient_friction;
constexpr float gpu_min_rel_vel_for_friction = min_rel_vel_for_friction;

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

__device__ void handle_particle_particle_collision(
    float* s_particles_pos_x, float* s_particles_pos_y,
    float* s_particles_vel_x, float* s_particles_vel_y,
    float* s_particles_radius,
    float m1, float m2,
    int p1, int p2);

__device__ void handle_boundary(
    float* s_particles_pos_x, float* s_particles_pos_y,
    float* s_particles_vel_x, float* s_particles_vel_y,
    float* s_particles_radius, int idx);

__global__ void first_half_step(GPU_unified_mem gpu_mem);

__global__ void second_half_step(GPU_unified_mem gpu_mem);
__device__ void handle_boundary(GPU_unified_mem gpu_mem, int idx);

__device__ __forceinline__ float angle_to_deg(float ax, float ay, float bx, float by);

__device__ void stage2_then_stage0(
    GPU_unified_mem const& gpu_mem,
    float* s_particles_pos_x, float* s_particles_pos_y,
    float* s_particles_vel_x, float* s_particles_vel_y,
    float* s_particles_mass,
    float* s_springs_nat_len, float* s_springs_const,
    const float* __restrict__ s_springs_p1, const float* __restrict__ s_springs_p2,
    int idx, int env, int t);