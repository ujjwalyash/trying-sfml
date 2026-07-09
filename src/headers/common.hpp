#include <SFML/System/Vector2.hpp>
#include <array>

// physics params
inline constexpr float max_y = 1080;
inline constexpr float max_x = 1920;

// since i keep both balanced out anyways just make both 0
inline constexpr sf::Vector2f gravity = {0, 10};
inline constexpr float buoyancy_const = 4.f/3.f * 3.141f * 10.f; // DO NOT MAKE DENSITY 1000

inline constexpr float restitution = 0.8f;

// if we have small particles then no need for this but for large circles its needed bc we have no rotation two spheres will just sit on each other
inline constexpr float min_rel_vel_for_friction = 0.f;
inline constexpr float coefficient_friction = 0.5;

// if two particles in a line move along that line then viscous force on back particle must be smaller but here its same as the front one
// this completely destroys the concept of streamlined bodies 
// // TODO: make viscous force more accurate ie springs(lines) face it too -- no need to do this for collisions though wont be too hard
inline constexpr float viscosity = 4.f * 3.141f * (1e-3) * 10; //// (not any more)*10 bc viscosity is only applied at point masses which have small radius so we scale it

// env params
inline constexpr int env_fps = 60;
inline constexpr int env_num_frames_per_creature_action = (float)env_fps/10; // every 100ms 
inline constexpr int env_num_iterations_per_frame = 16;
inline constexpr float env_dt = 1.f/(env_fps*env_num_iterations_per_frame);
inline constexpr int env_max_steps_per_episode = 500;
inline constexpr int env_observation_size = 12;

inline constexpr int env_num_particles = 148;
inline constexpr int env_num_springs = 343;
inline constexpr int env_num_muscles = 8;

inline constexpr int population_size = 16 * 7;
inline constexpr int num_generations = 500;
inline constexpr bool load_old_gen = true;

#ifdef __CUDACC__
    #define CUDA_HOST_DEVICE __host__ __device__ inline
#else
    #define CUDA_HOST_DEVICE inline
#endif

struct GPU_unified_mem{

    float* ptrs_particle[12];
	
    // naming them
    CUDA_HOST_DEVICE float* particles_vel_x() const { return ptrs_particle[0]; }
    CUDA_HOST_DEVICE float* particles_vel_y() const { return ptrs_particle[1]; }
    
    CUDA_HOST_DEVICE float* particles_pos_x() const { return ptrs_particle[2]; }
    CUDA_HOST_DEVICE float* particles_pos_y() const { return ptrs_particle[3]; }
	
    CUDA_HOST_DEVICE float* particles_net_acc_x() const { return ptrs_particle[4]; }
    CUDA_HOST_DEVICE float* particles_net_acc_y() const { return ptrs_particle[5]; }
	
    CUDA_HOST_DEVICE float* particles_spring_acc_x() const { return ptrs_particle[6]; }
    CUDA_HOST_DEVICE float* particles_spring_acc_y() const { return ptrs_particle[7]; }
	
    CUDA_HOST_DEVICE float* particles_const_acc_x() const { return ptrs_particle[8]; }
    CUDA_HOST_DEVICE float* particles_const_acc_y() const { return ptrs_particle[9]; }
    
    CUDA_HOST_DEVICE float* particles_radius() const { return ptrs_particle[11]; }
    CUDA_HOST_DEVICE float* particles_mass() const { return ptrs_particle[10]; }
	
    float* ptrs_spring[9];
    
    CUDA_HOST_DEVICE float* springs_nat_len() const { return ptrs_spring[0]; }
    CUDA_HOST_DEVICE float* springs_const() const { return ptrs_spring[1]; }
    CUDA_HOST_DEVICE float* springs_damping_factor() const { return ptrs_spring[2]; }
    CUDA_HOST_DEVICE float* springs_viscous_factor() const { return ptrs_spring[3]; }
    CUDA_HOST_DEVICE float* springs_moi_along_com() const { return ptrs_spring[4]; }
    CUDA_HOST_DEVICE float* springs_outside_body() const { return ptrs_spring[5]; }
    CUDA_HOST_DEVICE float* springs_radius() const { return ptrs_spring[6]; }
    
    // the local_idx is used for these
    CUDA_HOST_DEVICE float* springs_p1() const { return ptrs_spring[7]; }
    CUDA_HOST_DEVICE float* springs_p2() const { return ptrs_spring[8]; }

      // one-per-env scalars
    float* ptrs_env[8];
    CUDA_HOST_DEVICE float* env_reward()          const { return ptrs_env[0]; }
    CUDA_HOST_DEVICE float* env_ball_pos_x()      const { return ptrs_env[1]; }
    CUDA_HOST_DEVICE float* env_ball_pos_y()      const { return ptrs_env[2]; }
    CUDA_HOST_DEVICE float* env_goal_pos_x()      const { return ptrs_env[3]; }
    CUDA_HOST_DEVICE float* env_goal_pos_y()      const { return ptrs_env[4]; }
    CUDA_HOST_DEVICE float* env_num_steps_done()  const { return ptrs_env[5]; }
    CUDA_HOST_DEVICE float* env_episode_end()     const { return ptrs_env[6]; }
    CUDA_HOST_DEVICE float* env_goal_center_idx() const { return ptrs_env[7]; }

    // 4 sensing points per env (only sensing_points[0..3] are ever used)
    float* env_sensing_pts; // size population_size * 4

    // recurrent muscle activation fed back into the net each step
    float* env_muscle_activation; // size population_size * env_num_muscles

    // neural net weights, laid out per-env contiguous block:
    // [ W0 (20*12=240) | b0 (12) | W1 (12*8=96) ]  = 348 floats/env
    static constexpr int nn_in     = env_num_muscles + env_observation_size; // 20
    static constexpr int nn_hidden = 12;
    static constexpr int nn_out    = env_num_muscles; // 8
    static constexpr int nn_floats_per_env = nn_in*nn_hidden + nn_hidden + nn_hidden*nn_out + nn_out; // 348
    float* env_nn_data; // size population_size * nn_floats_per_env

    CUDA_HOST_DEVICE float* nn_W0(int env) const { return env_nn_data + env*nn_floats_per_env; }
    CUDA_HOST_DEVICE float* nn_b0(int env) const { return nn_W0(env) + nn_in*nn_hidden; }
    CUDA_HOST_DEVICE float* nn_W1(int env) const { return nn_b0(env) + nn_hidden; }
    CUDA_HOST_DEVICE float* nn_b1(int env) const { return nn_W1(env) + nn_out; }

    // muscle "rest" constants -- set once at construction, never mutated by the kernel
    // (mirrors Muscle's m_rest_length / m_rest_spring_constant / m_contraction_limit / m_max_spring_constant_scaling)
    float* muscle_rest_nat_len;          // size population_size * env_num_muscles
    float* muscle_rest_spring_const;     // size population_size * env_num_muscles
    float* muscle_contraction_limit;     // size population_size * env_num_muscles
    float* muscle_max_const_scaling;     // size population_size * env_num_muscles
	
};

extern GPU_unified_mem gpu_mem;

// cpu interface
void allocate_cuda_memory();
void free_cuda_memory();

// kernels
void launch_big_kernel();
void launch_first_half_step();
void launch_second_half_step();

void cudaSynchronize();
