#include "headers/cuda.cuh"
#include "headers/helper_math.cuh"

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
    
    for(float* & ptr: gpu_mem.ptrs_env){
        checkCudaErrors(cudaMallocManaged(&ptr, population_size*sizeof(float)));
    }
    checkCudaErrors(cudaMallocManaged(&gpu_mem.env_sensing_pts,        population_size*4*sizeof(float)));
    checkCudaErrors(cudaMallocManaged(&gpu_mem.env_muscle_activation,  population_size*env_num_muscles*sizeof(float)));
    checkCudaErrors(cudaMallocManaged(&gpu_mem.env_nn_data,            population_size*GPU_unified_mem::nn_floats_per_env*sizeof(float)));
    checkCudaErrors(cudaMallocManaged(&gpu_mem.muscle_rest_nat_len,      population_size*env_num_muscles*sizeof(float)));
    checkCudaErrors(cudaMallocManaged(&gpu_mem.muscle_rest_spring_const, population_size*env_num_muscles*sizeof(float)));
    checkCudaErrors(cudaMallocManaged(&gpu_mem.muscle_contraction_limit, population_size*env_num_muscles*sizeof(float)));
    checkCudaErrors(cudaMallocManaged(&gpu_mem.muscle_max_const_scaling, population_size*env_num_muscles*sizeof(float)));

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
    for(float* & ptr: gpu_mem.ptrs_env){
        checkCudaErrors(cudaFree(ptr));
    }
    checkCudaErrors(cudaFree(gpu_mem.env_sensing_pts));
    checkCudaErrors(cudaFree(gpu_mem.env_muscle_activation));
    checkCudaErrors(cudaFree(gpu_mem.env_nn_data));
    checkCudaErrors(cudaFree(gpu_mem.muscle_rest_nat_len));
    checkCudaErrors(cudaFree(gpu_mem.muscle_rest_spring_const));
    checkCudaErrors(cudaFree(gpu_mem.muscle_contraction_limit));
    checkCudaErrors(cudaFree(gpu_mem.muscle_max_const_scaling));
}

//* kernels
// cannot execute procedural code or statement logic (like macros, loops, or function calls) out in the open file space
// assert(block_sz <= 1024);

void launch_big_kernel(){

    big_kernel<<<population_size, env_num_springs + env_num_muscles>>>(gpu_mem);
    
    checkCudaErrors(cudaGetLastError());
}

__global__ void big_kernel(GPU_unified_mem gpu_mem){
    
    // coords within thread block
    int idx = threadIdx.x;
    int warp_id = threadIdx.x / 32;
    
    constexpr int num_threads = env_num_muscles + env_num_springs;
    static_assert(num_threads >= 2 * env_num_particles, "num_threads < 2 * env_num_particles");
    
    constexpr int NUM_ARRAYS_PAR = 12;
    constexpr int NUM_ARRAYS_SPR = 5;
    
    // num_parts * env_num
    int particle_offset = env_num_particles * blockIdx.x;
    int spring_offset = num_threads * blockIdx.x;

    // decalre everyhting in shared mem
    __shared__ float s_particles_vel_x[env_num_particles];
    __shared__ float s_particles_vel_y[env_num_particles];
    __shared__ float s_particles_pos_x[env_num_particles];
    __shared__ float s_particles_pos_y[env_num_particles];
    __shared__ float s_particles_net_acc_x[env_num_particles];
    __shared__ float s_particles_net_acc_y[env_num_particles];
    __shared__ float s_particles_spring_acc_x[env_num_particles];
    __shared__ float s_particles_spring_acc_y[env_num_particles];
    
    // const float* __restrict__ s_particles_const_acc_x = gpu_mem.particles_const_acc_x();
    // const float* __restrict__ s_particles_const_acc_y = gpu_mem.particles_const_acc_y();
    // const float* __restrict__ s_particles_mass = gpu_mem.particles_mass();
    // const float* __restrict__ s_particles_radius = gpu_mem.particles_radius();
    __shared__ float s_particles_const_acc_x[env_num_particles];
    __shared__ float s_particles_const_acc_y[env_num_particles];
    __shared__ float s_particles_mass[env_num_particles];
    __shared__ float s_particles_radius[env_num_particles];
    
    __shared__ float s_springs_nat_len[num_threads];
    __shared__ float s_springs_const[num_threads];
    __shared__ float s_springs_damping_factor[num_threads];
    __shared__ float s_springs_viscous_factor[num_threads];
    __shared__ float s_springs_moi_along_com[num_threads];
   
    const float* __restrict__ s_springs_outside_body = gpu_mem.springs_outside_body() + spring_offset;
    const float* __restrict__ s_springs_radius = gpu_mem.springs_radius() + spring_offset;
    const float* __restrict__ s_springs_p1 = gpu_mem.springs_p1() + spring_offset;
    const float* __restrict__ s_springs_p2 = gpu_mem.springs_p2() + spring_offset;

    typedef cub::BlockRadixSort<float, num_threads, 1, int> BlockSort;
    __shared__ typename BlockSort::TempStorage temp_storage;
    __shared__ int sorted_inds[env_num_particles];
    
    float* global_ptrs[NUM_ARRAYS_PAR + NUM_ARRAYS_SPR] = {
        gpu_mem.particles_vel_x(), gpu_mem.particles_vel_y(),
        gpu_mem.particles_pos_x(), gpu_mem.particles_pos_y(),
        gpu_mem.particles_net_acc_x(), gpu_mem.particles_net_acc_y(),
        gpu_mem.particles_spring_acc_x(), gpu_mem.particles_spring_acc_y(),
        
        gpu_mem.particles_const_acc_x(), gpu_mem.particles_const_acc_y(),
        gpu_mem.particles_radius(), 
        gpu_mem.particles_mass(),


        gpu_mem.springs_nat_len(), 
        gpu_mem.springs_const(),
        gpu_mem.springs_damping_factor(),
        gpu_mem.springs_viscous_factor(),
        gpu_mem.springs_moi_along_com(),
        
        // gpu_mem.springs_outside_body(),
        // gpu_mem.springs_radius(),
        // gpu_mem.springs_p1() ,gpu_mem.springs_p2()
    };

    float* shared_ptrs[NUM_ARRAYS_PAR + NUM_ARRAYS_SPR] = {
        s_particles_vel_x, s_particles_vel_y,
        s_particles_pos_x, s_particles_pos_y,
        s_particles_net_acc_x, s_particles_net_acc_y,
        s_particles_spring_acc_x, s_particles_spring_acc_y,

        s_particles_const_acc_x, s_particles_const_acc_y,
        s_particles_radius,
        s_particles_mass,

        s_springs_nat_len,
        s_springs_const,
        s_springs_damping_factor,
        s_springs_viscous_factor, 
        s_springs_moi_along_com, 
        // s_springs_outside_body,
        // s_springs_radius,
        // s_springs_p1, s_springs_p2
    };
    
    //* load everything needed to shared mem
    // first 5 (32*5 = 160 > 148(num_parts)) warps load the first half of things
    //TODO: change to load 4 floats at a time
    static_assert(5 * 32 >= env_num_particles);

    // load particle data
    if(warp_id < 5){
        if(idx < env_num_particles){
            // #pragma unroll // Tells compiler to flatten this loop for max performance
            for (int i = 0; i < NUM_ARRAYS_PAR/2; i++) {
                shared_ptrs[i][idx] = global_ptrs[i][idx + particle_offset];
            }
        }
    }
    else if(warp_id < 10){
        int offset = 5 * 32;
        if(idx - offset < env_num_particles){
            // #pragma unroll // Tells compiler to flatten this loop for max performance
            for (int i = NUM_ARRAYS_PAR/2; i < NUM_ARRAYS_PAR; i++) {
                shared_ptrs[i][idx - offset] = global_ptrs[i][idx - offset + particle_offset];
            }
        }
    }

    // load spring data
    // #pragma unroll // Tells compiler to flatten this loop for max performance
    for (int i = 0; i < NUM_ARRAYS_SPR; i++) {
        shared_ptrs[i+NUM_ARRAYS_PAR][idx] = global_ptrs[i+NUM_ARRAYS_PAR][idx + spring_offset];
    }

    // no need bc in next step if idx < num_part then this idx would have already loaded
    // whats needed in first half step
    bool ep_end = 0;

    for(int step = 0; step < env_max_steps_per_episode; step++){
        
        stage2_then_stage0(
            gpu_mem,
            s_particles_pos_x, s_particles_pos_y, s_particles_mass,
            s_springs_nat_len, s_springs_const,
            s_springs_p1, s_springs_p2,
            idx, blockIdx.x, ep_end);
        
        // bad if someone ends then that threads wont be present to barriers below
        // while it itself is waiting for a barrier -- deadlock
        // if you uncomment then add a barrier just before storing to global starts
        // if(ep_end == 1){
        //     break;
        // } 

        for(int cycle = 0; cycle < env_num_frames_per_creature_action; cycle++){
                    
            for(int iter = 0; iter < env_num_iterations_per_frame; iter++){

                //* first half step
                if(idx < env_num_particles){
                    s_particles_vel_x[idx] += 0.5f * s_particles_net_acc_x[idx] * gpu_env_dt;
                    s_particles_vel_y[idx] += 0.5f * s_particles_net_acc_y[idx] * gpu_env_dt;
                    
                    // m_curr_pos += m_vel*env_dt;
                    s_particles_pos_x[idx] += s_particles_vel_x[idx] * gpu_env_dt;
                    s_particles_pos_y[idx] += s_particles_vel_y[idx] * gpu_env_dt;
                }
                __syncthreads();



                // total num_springs + num_muscles threads in a block
                //TODO: test with a thread of each particle and it calc force for springs attched to it
                //TODO: it will elim atomicwrites but some threads may do a lot of work since time is max(all threads) bad
                //* calculate all 3 spring forces for all springs here for idx only
                float2 p1_spring_acc = make_float2(0, 0);
                float2 p2_spring_acc = make_float2(0, 0);
                int p1 = s_springs_p1[idx];
                int p2 = s_springs_p2[idx];
                float m1 = s_particles_mass[p1];
                float m2 = s_particles_mass[p2];
                float2 rel_pos = make_float2(s_particles_pos_x[p2] - s_particles_pos_x[p1],
                                                    s_particles_pos_y[p2] - s_particles_pos_y[p1]);
                
                //* Spring::calculate_spring_force()
                float deformation = length(rel_pos) - s_springs_nat_len[idx];
                rel_pos = normalize(rel_pos);

                //! really big spring const multiplied to small deformation --> bad
                p1_spring_acc += (deformation * ( rel_pos)) * (s_springs_const[idx] / m1);
                p2_spring_acc += (deformation * (-rel_pos)) * (s_springs_const[idx] / m2);
                
                //* Spring::calculate_damping_force()
                float2 vel_vec = make_float2(s_particles_vel_x[p2] - s_particles_vel_x[p1],
                                                s_particles_vel_y[p2] - s_particles_vel_y[p1]);
                vel_vec /= (m1+m2);
                // project vel_vel along spring(rel_pos)
                // rel_pos is normalised already
                vel_vec = dot(vel_vec, rel_pos) * rel_pos;
                    
                p1_spring_acc += -(s_springs_damping_factor[idx]) * (m2*(-vel_vec));
                p2_spring_acc += -(s_springs_damping_factor[idx]) * (m1*(vel_vec));

                //* Spring::calculate_viscous_force()
                if(s_springs_outside_body[idx]){
                    //* TRUST that nvcc will inline all this 
                    float2 v1 = make_float2(s_particles_vel_x[p1], s_particles_vel_y[p1]);
                    float2 v2 = make_float2(s_particles_vel_x[p2], s_particles_vel_y[p2]);
                    float2 vel_com = (m1 * v1 + m2 * v2) / (m1 + m2);
                    float2 vel_com_along_spring = dot(vel_com, rel_pos) * rel_pos;

                    float2 perp_spring = make_float2(-rel_pos.y, rel_pos.x);

                    float2 perp_viscous_acc = make_float2(0.f, 0.f);
                    if(dot(vel_com, -perp_spring) > 0.f){
                        perp_viscous_acc = -(fminf(1.f / gpu_env_dt, 1.f * s_springs_viscous_factor[idx])) * (vel_com - vel_com_along_spring);
                    }

                    float2 parallel_viscous_acc = -(fminf(1.f / gpu_env_dt, 0.25f * s_springs_viscous_factor[idx])) * vel_com_along_spring;

                    float total_mass = m1 + m2;
                    float angular_acc = length(parallel_viscous_acc) * s_springs_radius[idx] / s_springs_moi_along_com[idx];

                    float2 dir_perp_spring = make_float2(0.f, 0.f);
                    if(angular_acc != 0.f){
                        float2 perp_of_parallel = make_float2(-parallel_viscous_acc.y, parallel_viscous_acc.x);
                        dir_perp_spring = normalize(perp_of_parallel);
                    }

                    p1_spring_acc += perp_viscous_acc + parallel_viscous_acc + dir_perp_spring * (angular_acc * m2 / total_mass * s_springs_nat_len[idx]);
                    p2_spring_acc += perp_viscous_acc + parallel_viscous_acc - dir_perp_spring * (angular_acc * m1 / total_mass * s_springs_nat_len[idx]);
                }

                // atomic wrtie back for accs
                atomicAdd(&s_particles_spring_acc_x[p1], p1_spring_acc.x);
                atomicAdd(&s_particles_spring_acc_y[p1], p1_spring_acc.y);
                
                atomicAdd(&s_particles_spring_acc_x[p2], p2_spring_acc.x);
                atomicAdd(&s_particles_spring_acc_y[p2], p2_spring_acc.y);

                // no need to sync here bc collision handling wont need the accs

                //* collision handle

                // sort by x
                float dx = s_particles_pos_x[(int)gpu_mem.env_sperm_center_idx()[blockIdx.x]] - s_particles_pos_x[(int)gpu_mem.env_ball_center_idx()[blockIdx.x]];
                float dy = s_particles_pos_y[(int)gpu_mem.env_sperm_center_idx()[blockIdx.x]] - s_particles_pos_y[(int)gpu_mem.env_ball_center_idx()[blockIdx.x]];
                if(sqrtf(dx*dx + dy*dy) < (sperm_length/2 + ball_radius) && iter % iter_per_collision_check == 0){

                    float thread_key[1];
                    int thread_value[1];
                    if(idx < env_num_particles){
                        thread_key[0] = s_particles_pos_x[idx] - s_particles_radius[idx];
                        thread_value[0] = idx; 
                    }
                    else{
                        thread_key[0] = max_x + 1;
                        thread_value[0] = -1; 
                    }
                    
                    // all threads in block must enter together -- can't be in a divergent if branch
                    // the keys are sorted and values are moved with them thus the final thread value is sorted[idx]
                    BlockSort(temp_storage).Sort(thread_key, thread_value);
                    
                    if(idx < env_num_particles){
                        sorted_inds[idx] = thread_value[0];
                    }
                    __syncthreads();   
                    
                    // each idx will handle ALL collision of particle idx with particles to idx's right
                    if(idx < env_num_particles){
                        int p1 = sorted_inds[idx];

                        for(int k = idx+1; k < env_num_particles; k++){
                            int p2 = sorted_inds[k];
                            if(p2 < 0) break;

                            if(s_particles_pos_x[p2] - s_particles_radius[p2] > s_particles_pos_x[p1] + s_particles_radius[p1]){
                                break;
                            }
                            handle_particle_particle_collision(
                                s_particles_pos_x, s_particles_pos_y,
                                s_particles_vel_x, s_particles_vel_y,
                                s_particles_radius,
                                s_particles_mass[p1], s_particles_mass[p2],
                                p1, p2);
                        }
                    }
                }

                __syncthreads();
                // //* second half step

                if(idx < env_num_particles){
                    // m_acc = gravity + m_buoyancy_acc + m_spring_acc;
                    //! assignment not +=
                    s_particles_net_acc_x[idx] = s_particles_spring_acc_x[idx] + s_particles_const_acc_x[idx];
                    s_particles_net_acc_y[idx] = s_particles_spring_acc_y[idx] + s_particles_const_acc_y[idx];
                    
                    s_particles_spring_acc_x[idx] = 0.f;
                    s_particles_spring_acc_y[idx] = 0.f;

                    // m_vel += 0.5f * m_acc * env_dt;
                    s_particles_vel_x[idx] += 0.5f * s_particles_net_acc_x[idx] * gpu_env_dt;
                    s_particles_vel_y[idx] += 0.5f * s_particles_net_acc_y[idx] * gpu_env_dt;

                    if(iter % iter_per_boundary_check == 0){
                        handle_boundary(
                            s_particles_pos_x, s_particles_pos_y,
                            s_particles_vel_x, s_particles_vel_y,
                            s_particles_radius, idx);
                    }
                }

                __syncthreads();
                
            }
        }
    }

    //* write back shared to global 

    // some env may end early
    // __syncthreads();

    // store particle data
    if(warp_id < 5){
        if(idx < env_num_particles){
            // #pragma unroll // Tells compiler to flatten this loop for max performance
            for (int i = 0; i < NUM_ARRAYS_PAR/2; i++) {
                global_ptrs[i][idx + particle_offset] = shared_ptrs[i][idx];
            }
        }
    }
    else if(warp_id < 10){
        int offset = 5 * 32;
        if(idx - offset < env_num_particles){
            // #pragma unroll // Tells compiler to flatten this loop for max performance
            for (int i = NUM_ARRAYS_PAR/2; i < NUM_ARRAYS_PAR; i++) {
                global_ptrs[i][idx - offset + particle_offset] = shared_ptrs[i][idx - offset];
            }
        }
    }

    // sotre spring data
    // #pragma unroll // Tells compiler to flatten this loop for max performance
    for (int i = 0; i < NUM_ARRAYS_SPR; i++) {
        global_ptrs[i+NUM_ARRAYS_PAR][idx + spring_offset] = shared_ptrs[i+NUM_ARRAYS_PAR][idx];
    }
}

__device__ void handle_particle_particle_collision(
    float* s_particles_pos_x, float* s_particles_pos_y,
    float* s_particles_vel_x, float* s_particles_vel_y,
    float* s_particles_radius,
    float m1, float m2,
    int p1, int p2)
{
    float2 pos1 = make_float2(s_particles_pos_x[p1], s_particles_pos_y[p1]);
    float2 pos2 = make_float2(s_particles_pos_x[p2], s_particles_pos_y[p2]);
    float2 line_of_contact = pos2 - pos1;

    float r1 = s_particles_radius[p1];
    float r2 = s_particles_radius[p2];

    float dist = length(line_of_contact);
    float overlap = (r1 + r2) - dist;
    if (overlap <= 0.f) return;

    float2 v1 = make_float2(s_particles_vel_x[p1], s_particles_vel_y[p1]);
    float2 v2 = make_float2(s_particles_vel_x[p2], s_particles_vel_y[p2]);

    // projectedOnto(b) == (dot(a,b)/dot(b,b)) * b
    float2 v1_along = (dot(v1, line_of_contact) / dot(line_of_contact, line_of_contact)) * line_of_contact;
    float2 v2_along = (dot(v2, line_of_contact) / dot(line_of_contact, line_of_contact)) * line_of_contact;

    // already moving apart
    if(dot(line_of_contact, v2_along - v1_along) >= 0.f) return;

    float2 v1_perp = v1 - v1_along;
    float2 v2_perp = v2 - v2_along;

    // wrt p1
    float2 rel_vel_perp = v2_perp - v1_perp;
    float rel_vel_perp_len = length(rel_vel_perp);

    float c1 = (m1 - m2 * gpu_restitution) / (m1 + m2);
    float c2 = (1.f + gpu_restitution) * (m2 / (m1 + m2));

    float2 impulse_along = ((c1 * v1_along + c2 * v2_along) - v1_along) * m1;

    float magnitude_impulse_friction = fminf(
        gpu_coefficient_friction * length(impulse_along),
        fmaxf(rel_vel_perp_len, gpu_min_rel_vel_for_friction) * fminf(m1, m2));

    float2 impulse_perp = magnitude_impulse_friction * rel_vel_perp /
        (rel_vel_perp_len != 0.f ? rel_vel_perp_len : 1.f);

    float2 delta_v1 =  (impulse_along + impulse_perp) / m1;
    float2 delta_v2 = -1.f * (impulse_along + impulse_perp) / m2;

    // multiple threads can touch the same particle (idx=p1 for one pair,
    // idx=i=p2 for another pair running concurrently), so writes must be atomic
    atomicAdd(&s_particles_vel_x[p1], delta_v1.x);
    atomicAdd(&s_particles_vel_y[p1], delta_v1.y);

    atomicAdd(&s_particles_vel_x[p2], delta_v2.x);
    atomicAdd(&s_particles_vel_y[p2], delta_v2.y);
}

__device__ void handle_boundary(
    float* s_particles_pos_x, float* s_particles_pos_y,
    float* s_particles_vel_x, float* s_particles_vel_y,
    float* s_particles_radius, int idx)
{
    float radius = s_particles_radius[idx];

    if (s_particles_pos_x[idx] + radius > gpu_max_x) {
        // m_curr_pos.x = 2*gpu_max_x - (m_curr_pos.x + radius) - radius;
        s_particles_pos_x[idx] = 2.f * gpu_max_x - (s_particles_pos_x[idx] + radius) - radius;
        s_particles_vel_x[idx] = -s_particles_vel_x[idx];
    }
    else if (s_particles_pos_x[idx] - radius < 0.f) {
        // m_curr_pos.x = 2*0 - (m_curr_pos.x - radius) + radius;
        s_particles_pos_x[idx] = -(s_particles_pos_x[idx] - radius) + radius;
        s_particles_vel_x[idx] = -s_particles_vel_x[idx];
    }

    if (s_particles_pos_y[idx] + radius > gpu_max_y) {
        s_particles_pos_y[idx] = 2.f * gpu_max_y - (s_particles_pos_y[idx] + radius) - radius;
        s_particles_vel_y[idx] = -s_particles_vel_y[idx];
    }
    else if (s_particles_pos_y[idx] - radius < 0.f) {
        s_particles_pos_y[idx] = -(s_particles_pos_y[idx] - radius) + radius;
        s_particles_vel_y[idx] = -s_particles_vel_y[idx];
    }
}

__device__ void stage2_then_stage0(
    GPU_unified_mem const& gpu_mem,
    float* s_particles_pos_x, float* s_particles_pos_y,
    float* s_particles_mass,
    float* s_springs_nat_len, float* s_springs_const,
    const float* __restrict__ s_springs_p1, const float* __restrict__ s_springs_p2,
    int idx, int env, bool& ep_end)
{
    int ball_center_idx = (int)gpu_mem.env_ball_center_idx()[env];
    const float* sp = gpu_mem.env_sensing_pts + env*4;

    //* stage_2

    // calc reward
    float ball_x = s_particles_pos_x[ball_center_idx];
    float ball_y = s_particles_pos_y[ball_center_idx];
    float goal_x = gpu_mem.env_goal_pos_x()[env];
    float goal_y = gpu_mem.env_goal_pos_y()[env];

    // warp 0 does this 
    if(idx == 0){

        int apex_idx = (int)sp[0]; // Creature::get_apex_tip_index() == m_sensing_points[0]
        float dx_c = s_particles_pos_x[apex_idx] - ball_x;
        float dy_c = s_particles_pos_y[apex_idx] - ball_y;
        float creature_ball_dist = sqrtf(dx_c*dx_c + dy_c*dy_c);

        float dx_g = ball_x - goal_x, dy_g = ball_y - goal_y;
        float goal_ball_dist = sqrtf(dx_g*dx_g + dy_g*dy_g);

        float reward = -creature_ball_dist/100.f - goal_ball_dist/300.f - 1.f;

        constexpr float goal_radius = 10.f;
        if(goal_ball_dist < goal_radius){
            reward += 1000.f;
            ep_end = 1;
            // gpu_mem.env_episode_end()[env] = 1.f;
        }

        gpu_mem.env_reward()[env] += reward;
        // gpu_mem.env_num_steps_done()[env] += 1.f;
        // if(gpu_mem.env_num_steps_done()[env] > (float)env_max_steps_per_episode)
        //     gpu_mem.env_episode_end()[env] = 1.f;
    }

    //* stage 0

    // get obs
    __shared__ float s_obs[env_observation_size];
    // warp 1 does this 
    if(idx == 32){

        float vals[8];
        float mn = 10000.f, mx = 0.f;
        for(int i = 0; i < 4; i++){
            int p = (int)sp[i];
            float dx = ball_x - s_particles_pos_x[p];
            float dy = ball_y - s_particles_pos_y[p];
            vals[i] = sqrtf(dx*dx + dy*dy);
            mn = fminf(mn, vals[i]); 
            mx = fmaxf(mx, vals[i]);
        }
        float dem = mx - mn;
        for(int i = 0; i < 4; i++) s_obs[i] = 2.f*(vals[i]-mn)/dem - 1.f;

        mn = 10000.f; mx = 0.f;
        for(int i = 4; i < 8; i++){
            int p = (int)sp[i-4];
            float dx = goal_x - s_particles_pos_x[p];
            float dy = goal_y - s_particles_pos_y[p];
            vals[i] = sqrtf(dx*dx + dy*dy);
            mn = fminf(mn, vals[i]);
            mx = fmaxf(mx, vals[i]);
        }
        dem = mx - mn;
        for(int i = 4; i < 8; i++) s_obs[i] = 2.f*(vals[i]-mn)/dem - 1.f;

        int p0 = (int)sp[0], p3 = (int)sp[3];
        s_obs[8]  = 2.f*(s_particles_pos_x[p0]/max_x) - 1.f;
        s_obs[9]  = 2.f*(s_particles_pos_y[p0]/max_y) - 1.f;
        s_obs[10] = 2.f*(s_particles_pos_x[p3]/max_x) - 1.f;
        s_obs[11] = 2.f*(s_particles_pos_y[p3]/max_y) - 1.f;
    }
    
    __syncthreads();
    
    //forward pass
    constexpr int nn_in = GPU_unified_mem::nn_in, nn_hidden = GPU_unified_mem::nn_hidden, nn_out = GPU_unified_mem::nn_out;

    __shared__ float s_layer_in[nn_in];
    __shared__ float s_hidden[nn_hidden];
    __shared__ float s_new_activation[nn_out];

    float* prev_activation = gpu_mem.env_muscle_activation + env*env_num_muscles;

    if(idx < env_num_muscles)             s_layer_in[idx] = 2.f*prev_activation[idx] - 1.f;
    if(idx >= env_num_muscles && idx < nn_in) s_layer_in[idx] = s_obs[idx - env_num_muscles];
    __syncthreads();

    const float* W0 = gpu_mem.nn_W0(env); // row-major [nn_in x nn_hidden]
    const float* b0 = gpu_mem.nn_b0(env);
    if(idx < nn_hidden){
        float acc = b0[idx];
        #pragma unroll
        for(int i = 0; i < nn_in; i++) acc += s_layer_in[i] * W0[i*nn_hidden + idx];
        s_hidden[idx] = fmaxf(acc, 0.f); // relu
    }
    __syncthreads();

    const float* W1 = gpu_mem.nn_W1(env); // [nn_hidden x nn_out], no bias (matches CPU forward)
    const float* b1 = gpu_mem.nn_b1(env);
    if(idx < nn_out){
        float acc = b1[idx];
        #pragma unroll
        for(int i = 0; i < nn_hidden; i++) acc += s_hidden[i] * W1[i*nn_out + idx];
        // sigmoid approx: 0.5*(1 + x/(1+|x|))
        s_new_activation[idx] = 0.5f*(1.f + acc/(1.f + fabsf(acc)));
    }
    __syncthreads();

    if(idx < env_num_muscles) prev_activation[idx] = s_new_activation[idx];

    //Muscle::handle_nerve_signal()
    // muscle m lives at spring-array slot (env_num_springs + m)
    if(idx >= env_num_springs && idx < env_num_springs + env_num_muscles){
        int m = idx - env_num_springs;
        float activation = s_new_activation[m];

        float rest_len   = gpu_mem.muscle_rest_nat_len[env*env_num_muscles + m];
        float rest_k     = gpu_mem.muscle_rest_spring_const[env*env_num_muscles + m];
        float contr_lim  = gpu_mem.muscle_contraction_limit[env*env_num_muscles + m];
        float k_scaling  = gpu_mem.muscle_max_const_scaling[env*env_num_muscles + m];
        // float radius     = s_springs_radius[idx];

        // int p1 = (int)s_springs_p1[idx], p2 = (int)s_springs_p2[idx];
        // float m1 = s_particles_mass[p1], m2 = s_particles_mass[p2];
        // float total_mass = m1 + m2;

        float nat_len = rest_len * (1.f - activation*contr_lim);
        float k       = rest_k * (1.f + activation*k_scaling);

        // float damping = (viscosity * nat_len / ((logf(nat_len/radius) + 0.5f) * total_mass)) * gpu_env_dt;
        // damping = fminf(1.f, damping);

        // float d = (m1 - m2) * nat_len / (2.f*total_mass);
        // float moi = nat_len*nat_len/12.f + d*d;

        s_springs_nat_len[idx]         = nat_len;
        s_springs_const[idx]           = k;
        // s_springs_damping_factor[idx]  = damping;
        // s_springs_moi_along_com[idx]   = moi;
    }
    __syncthreads(); // all threads must see updated muscle spring params before the physics loop starts
}

/*

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
    // assignemnt not +=
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

*/