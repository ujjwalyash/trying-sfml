#include "headers/cuda.cuh"

GPU_unified_mem gpu_mem{};

void cudaSynchronize(){
    checkCudaErrors(cudaDeviceSynchronize());
}

//* memory management
void allocate_cuda_memory(){
    
	int vector_length = population_size * env_num_particles;
    for(float* & ptr: gpu_mem.ptrs_particle){
        checkCudaErrors(cudaMallocManaged(&ptr, vector_length*sizeof(float)));
        // std::cout << "allocated " << ptr << '\n';
    }

    vector_length = population_size * (env_num_springs + env_num_muscles);
    for(float* & ptr: gpu_mem.ptrs_spring){
        checkCudaErrors(cudaMallocManaged(&ptr, vector_length*sizeof(float)));
        // std::cout << "allocated " << ptr << '\n';
    }
    
    // allocate constant memory
    cudaMemcpyToSymbol(gpu_env_dt, &env_dt, sizeof(float));
    cudaMemcpyToSymbol(gpu_max_x, &max_x, sizeof(float));
    cudaMemcpyToSymbol(gpu_max_y, &max_y, sizeof(float));

    checkCudaErrors(cudaDeviceSynchronize());
}    

void free_cuda_memory(){
    for(float* ptr: gpu_mem.ptrs_particle){
        checkCudaErrors(cudaFree(ptr));
        // std::cout << "freed " << ptr << '\n';
    }
    for(float* ptr: gpu_mem.ptrs_spring){
        checkCudaErrors(cudaFree(ptr));
        // std::cout << "freed " << ptr << '\n';
    }
}

//* kernels
// cannot execute procedural code or statement logic (like macros, loops, or function calls) out in the open file space
// assert(block_sz <= 1024);

void launch_big_kernel(){

    big_kernel<<<population_size, env_num_springs + env_num_muscles>>>(gpu_mem);
    
    checkCudaErrors(cudaGetLastError());
}

__global__ void big_kernel(GPU_unified_mem gpu_mem){

    // num_parts * env_num
    int global_offset = env_num_particles * blockIdx.x;

    // coords within thread block
    int idx = threadIdx.x;
    int warp_id = threadIdx.x / 32;

    constexpr int num_threads = env_num_muscles + env_num_springs;
    static_assert(num_threads >= 2 * env_num_particles, "num_threads < 2 * env_num_particles");

    constexpr int NUM_ARRAYS = 12;

    // decalre everyhting in shared mem
    __shared__ float s_particles_vel_x[env_num_particles];
    __shared__ float s_particles_vel_y[env_num_particles];
    __shared__ float s_particles_pos_x[env_num_particles];
    __shared__ float s_particles_pos_y[env_num_particles];
    __shared__ float s_particles_spring_acc_x[env_num_particles];
    __shared__ float s_particles_spring_acc_y[env_num_particles];
    __shared__ float s_particles_const_acc_x[env_num_particles];
    __shared__ float s_particles_const_acc_y[env_num_particles];
    __shared__ float s_particles_net_acc_x[env_num_particles];
    __shared__ float s_particles_net_acc_y[env_num_particles];
    __shared__ float s_particles_radius[env_num_particles];
    __shared__ float s_particles_mass[env_num_particles];
    
    
    float* global_ptrs[NUM_ARRAYS] = {
        gpu_mem.particles_vel_x(), gpu_mem.particles_vel_y(),
        gpu_mem.particles_pos_x(), gpu_mem.particles_pos_y(),
        gpu_mem.particles_spring_acc_x(), gpu_mem.particles_spring_acc_y(),
        gpu_mem.particles_const_acc_x(), gpu_mem.particles_const_acc_y(),
        gpu_mem.particles_net_acc_x(), gpu_mem.particles_net_acc_y(),
        gpu_mem.particles_radius(), gpu_mem.particles_mass()
    };

    float* shared_ptrs[NUM_ARRAYS] = {
        s_particles_vel_x, s_particles_vel_y,
        s_particles_pos_x, s_particles_pos_y,
        s_particles_spring_acc_x, s_particles_spring_acc_y,
        s_particles_const_acc_x, s_particles_const_acc_y,
        s_particles_net_acc_x, s_particles_net_acc_y,
        s_particles_radius, s_particles_mass
    };
    
    // load everything needed to shared mem
    // first 5 (32*5 = 160 > 148(num_parts)) warps load the first half of things
    static_assert(5 * 32 >= env_num_particles);
    if(warp_id < 5){
        if(idx < env_num_particles){
            #pragma unroll // Tells compiler to flatten this loop for max performance
            for (int i = 0; i < NUM_ARRAYS/2; i++) {
                shared_ptrs[i][idx] = global_ptrs[i][idx + global_offset];
            }
        }
    }
    else if(warp_id < 10){
        int offset = 5 * 32;
        if(idx - offset < env_num_particles){
            #pragma unroll // Tells compiler to flatten this loop for max performance
            for (int i = NUM_ARRAYS/2; i < NUM_ARRAYS; i++) {
                shared_ptrs[i][idx - offset] = global_ptrs[i][idx - offset + global_offset];
            }
        }
    }

    __syncthreads();

    for(int i = 0; i < env_num_particles; i++){
        #pragma unroll // Tells compiler to flatten this loop for max performance
        for (int k = 0; k < NUM_ARRAYS; k++) {
            assert(shared_ptrs[k][i] == global_ptrs[k][i + global_offset]);
        }
    }
    // some threads pass by this and in the dummy test modify shared memeory causing assert to fails
    __syncthreads();

    //dummy test
    if(idx < env_num_particles){
        s_particles_vel_x[idx] += 0.5f * s_particles_net_acc_x[idx] * gpu_env_dt;
        s_particles_vel_y[idx] += 0.5f * s_particles_net_acc_y[idx] * gpu_env_dt;
        
        // m_curr_pos += m_vel*env_dt;
        s_particles_pos_x[idx] += s_particles_vel_x[idx] * gpu_env_dt;
        s_particles_pos_y[idx] += s_particles_vel_y[idx] * gpu_env_dt;
    }

    // total num_springs + num_muscles threads in a block
    // update all springs



    // write back everything
    if(warp_id < 5){
        if(idx < env_num_particles){
            gpu_mem.particles_vel_x()[idx + global_offset] = s_particles_vel_x[idx];
            gpu_mem.particles_vel_y()[idx + global_offset] = s_particles_vel_y[idx];

            gpu_mem.particles_pos_x()[idx + global_offset] = s_particles_pos_x[idx];
            gpu_mem.particles_pos_y()[idx + global_offset] = s_particles_pos_y[idx];

            gpu_mem.particles_spring_acc_x()[idx + global_offset] = s_particles_spring_acc_x[idx];
            gpu_mem.particles_spring_acc_y()[idx + global_offset] = s_particles_spring_acc_y[idx];
        }
    }
    else if(warp_id < 10){

        int offset = 5 * 32;
        if(idx - offset < env_num_particles){

            gpu_mem.particles_const_acc_x()[idx - offset + global_offset] = s_particles_const_acc_x[idx - offset];
            gpu_mem.particles_const_acc_y()[idx - offset + global_offset] = s_particles_const_acc_y[idx - offset];
            
            gpu_mem.particles_net_acc_x()[idx - offset + global_offset] = s_particles_net_acc_x[idx - offset];
            gpu_mem.particles_net_acc_y()[idx - offset + global_offset] = s_particles_net_acc_y[idx - offset];
            
            gpu_mem.particles_radius()[idx - offset  + global_offset] = s_particles_radius[idx - offset];
        }
    }

}

__device__ void calculate_spring_force(float * ){

}

void launch_first_half_step(){

    
    first_half_step<<<grid_sz, block_sz>>>(gpu_mem);

    checkCudaErrors(cudaGetLastError());
}

__global__ void first_half_step(GPU_unified_mem gpu_mem){

    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    assert(gpu_env_dt == 1.f/(60 * 16));

    // m_vel += + 0.5f * m_acc * env_dt;
    if(idx < grid_sz*block_sz){
        gpu_mem.particles_vel_x()[idx] += 0.5f * gpu_mem.particles_net_acc_x()[idx] * gpu_env_dt;
        gpu_mem.particles_vel_y()[idx] += 0.5f * gpu_mem.particles_net_acc_y()[idx] * gpu_env_dt;
        
        // m_curr_pos += m_vel*env_dt;
        gpu_mem.particles_pos_x()[idx] += gpu_mem.particles_vel_x()[idx] * gpu_env_dt;
        gpu_mem.particles_pos_y()[idx] += gpu_mem.particles_vel_y()[idx] * gpu_env_dt;
    }

}

void launch_second_half_step(){

    
    second_half_step<<<grid_sz, block_sz>>>(gpu_mem);

    checkCudaErrors(cudaGetLastError());
}

__global__ void second_half_step(GPU_unified_mem gpu_mem){

    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    assert(gpu_env_dt == 1.f/(60 * 16));

    // m_acc = gravity + m_buoyancy_acc + m_spring_acc;
    //! assignemnt not +=
    if(idx < grid_sz*block_sz){

        gpu_mem.particles_net_acc_x()[idx] = gpu_mem.particles_spring_acc_x()[idx] + gpu_mem.particles_const_acc_x()[idx];
        gpu_mem.particles_net_acc_y()[idx] = gpu_mem.particles_spring_acc_y()[idx] + gpu_mem.particles_const_acc_y()[idx];
        
        gpu_mem.particles_spring_acc_x()[idx] = 0;
        gpu_mem.particles_spring_acc_y()[idx] = 0;
        
        // m_vel += 0.5f * m_acc * env_dt;
        gpu_mem.particles_vel_x()[idx] += 0.5f * gpu_mem.particles_net_acc_x()[idx] * gpu_env_dt;
        gpu_mem.particles_vel_y()[idx] += 0.5f * gpu_mem.particles_net_acc_y()[idx] * gpu_env_dt;
        
        handle_boundary(gpu_mem, idx);
    }
}

__device__ void handle_boundary(GPU_unified_mem gpu_mem, int idx){
    float radius = gpu_mem.particles_radius()[idx];
    if(gpu_mem.particles_pos_x()[idx] + radius > gpu_max_x){
        // reflect(gpu_max_x, 0, 1);
        // m_curr_pos.x = 2*gpu_max_x - (m_curr_pos.x + radius) - radius;        
        gpu_mem.particles_pos_x()[idx] = 2*gpu_max_x - (gpu_mem.particles_pos_x()[idx] + radius) - radius;        
        // m_vel.x = -m_vel.x;
        gpu_mem.particles_vel_x()[idx] = -gpu_mem.particles_vel_x()[idx];
    }
    else if(gpu_mem.particles_pos_x()[idx] - gpu_mem.particles_radius()[idx] < 0){
        // reflect(0, 0, -1);
        // m_curr_pos.x = 2*0 - (m_curr_pos.x - radius) + radius;        
        gpu_mem.particles_pos_x()[idx] = -(gpu_mem.particles_pos_x()[idx] - radius) + radius;        
        // m_vel.x = -m_vel.x;
        gpu_mem.particles_vel_x()[idx] = -gpu_mem.particles_vel_x()[idx];
    }
    
    if(gpu_mem.particles_pos_y()[idx] + radius > gpu_max_y){
        // reflect(gpu_max_y, 1, 1);
        // m_curr_pos.y = 2*gpu_max_y - (m_curr_pos.y + radius) - radius;        
        gpu_mem.particles_pos_y()[idx] = 2*gpu_max_y - (gpu_mem.particles_pos_y()[idx] + radius) - radius;        
        // m_vel.x = -m_vel.x;
        gpu_mem.particles_vel_y()[idx] = -gpu_mem.particles_vel_y()[idx];
    }
    else if(gpu_mem.particles_pos_y()[idx] - radius < 0){
        // reflect(0, 1, -1);
        // m_curr_pos.y = 2*0 - (m_curr_pos.y - radius) + radius;        
        gpu_mem.particles_pos_y()[idx] = -(gpu_mem.particles_pos_y()[idx] - radius) + radius;        
        // m_vel.y = -m_vel.y;
        gpu_mem.particles_vel_y()[idx] = -gpu_mem.particles_vel_y()[idx];
    }
}

// __global__ void reset(GPU_unified_mem gpu_mem){

// }