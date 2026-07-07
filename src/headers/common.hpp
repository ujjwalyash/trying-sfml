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
//// const int by deafult have internal linkage -- others have a extern declaration in cpp but this has
//// in cuda so problems ig
inline constexpr int population_size = 112;

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
	
    CUDA_HOST_DEVICE float* particles_spring_acc_x() const { return ptrs_particle[4]; }
    CUDA_HOST_DEVICE float* particles_spring_acc_y() const { return ptrs_particle[5]; }
	
    CUDA_HOST_DEVICE float* particles_const_acc_x() const { return ptrs_particle[6]; }
    CUDA_HOST_DEVICE float* particles_const_acc_y() const { return ptrs_particle[7]; }
	
    CUDA_HOST_DEVICE float* particles_net_acc_x() const { return ptrs_particle[8]; }
    CUDA_HOST_DEVICE float* particles_net_acc_y() const { return ptrs_particle[9]; }
    
    CUDA_HOST_DEVICE float* particles_radius() const { return ptrs_particle[10]; }
    CUDA_HOST_DEVICE float* particles_mass() const { return ptrs_particle[11]; }
	
    float* ptrs_spring[12];

    CUDA_HOST_DEVICE float* springs_nat_len() const { return ptrs_spring[0]; }
    CUDA_HOST_DEVICE float* springs_spr_const() const { return ptrs_spring[1]; }
	
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
