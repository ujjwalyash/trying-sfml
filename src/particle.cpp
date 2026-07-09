#include "headers/particle.hpp"
#include <SFML/System/Vector2.hpp>
#include <cmath>
#include <iostream>

Particle::Particle(int env_id, int id, float radius, float mass, sf::Vector2<float> curr_pos, sf::Vector2<float> vel, structure type, bool cpu_only)
    :m_env_id(env_id),
     m_id(id),
     m_global_id(env_id * env_num_particles + id),
     m_cpu_only(cpu_only),
     m_original_vel(vel),
     m_original_curr_pos(curr_pos),
     m_vel(vel),
     m_curr_pos(curr_pos),
     m_spring_acc({0, 0}),
     m_radius(radius),
     m_cube_radius(radius*radius*radius),
     m_mass(mass),
     m_sqrt_mass(sqrt(mass)),
    //  m_damping_factor(fmin(1, (viscosity*m_radius*env_dt)/(m_mass))),
     m_type(type)
{
    sf::Vector2f buoyancy_acc = -buoyancy_const*(m_cube_radius)*gravity/m_mass;
    m_buoyancy_acc = buoyancy_acc;
    m_acc = buoyancy_acc + gravity;

    if(env_id >= 0){

        gpu_mem.particles_pos_x()[m_global_id] = curr_pos.x;
        gpu_mem.particles_pos_y()[m_global_id] = curr_pos.y;
        
        gpu_mem.particles_vel_x()[m_global_id] = vel.x;
        gpu_mem.particles_vel_y()[m_global_id] = vel.y;
        
        gpu_mem.particles_const_acc_x()[m_global_id] = gravity.x;
        gpu_mem.particles_const_acc_y()[m_global_id] = gravity.y;
        
        gpu_mem.particles_const_acc_x()[m_global_id] += buoyancy_acc.x;
        gpu_mem.particles_const_acc_y()[m_global_id] += buoyancy_acc.y;
        
        gpu_mem.particles_net_acc_x()[m_global_id] = 0;
        gpu_mem.particles_net_acc_y()[m_global_id] = 0;
        
        gpu_mem.particles_spring_acc_x()[m_global_id] = 0;
        gpu_mem.particles_spring_acc_y()[m_global_id] = 0;
        
        gpu_mem.particles_radius()[m_global_id] = m_radius;
        gpu_mem.particles_mass()[m_global_id] = m_mass;
        
    }

    // std::cout << m_env_id << " " << m_id << ": " << m_global_id << '\n';
    
}

void Particle::set_pos_vel(sf::Vector2<float> curr_pos, sf::Vector2<float> vel){
    m_curr_pos = curr_pos;
    m_vel = vel;
}

void Particle::shift_pos(sf::Vector2<float> shift){
    m_curr_pos += shift;

    if(m_env_id >= 0){

        gpu_mem.particles_pos_x()[m_global_id] += shift.x;
        gpu_mem.particles_pos_y()[m_global_id] += shift.y;
    }
}


void Particle::check(){
    
    // for debugging
    auto approx_eq = [&](float a, float b){
        float perct_error = 1e-2; // pct    
        float diff = fabs(a - b);
        bool result = (diff < (2000)) || (diff/fmax(fabs(a), fabs(b)) < perct_error);
        // result = result * (fabs(a) < 1e-6 || fabs(b/a) < 5);
    
        if(!result){
            std::fprintf(stderr, "\nAssert failed: %f -- %f\n env_id, local_id: %d, %d \n", a, b, m_env_id, m_id);
            // std::fprintf(stderr, "\nAssert failed: %f -- %f\n env_id, local_id: %d, %d \n", a, b, m_env_id, m_id);
        }
    
        // return true;
        return result;
    };
    // float cpu_val = m_spring_acc.y;
    // float gpu_val = gpu_mem.particles_spring_acc_y()[m_global_id];

    // if (!approx_eq(cpu_val, gpu_val)) {
    //     // std::cout << "Divergence at Iteration: " << iteration_count << "\n"
    //     std::cout  << "Particle ID: " << m_global_id << "\n"
    //               << "CPU Spring Acc: " << cpu_val << "\n"
    //               << "GPU Spring Acc: " << gpu_val << "\n"
    //               << "Current Pos (CPU): " << m_curr_pos.x << ", " << m_curr_pos.y << "\n"
    //               << "Current Pos (GPU): " << gpu_mem.particles_pos_x()[m_global_id] << "\n";
    //     assert(false);
    // }

    assert(approx_eq(m_radius, gpu_mem.particles_radius()[m_global_id]));
    
    // // approx_eq will not be int gdb backtrace bc it returned a false val without error
    // // the actual path to termination starts after approx_eq returns and its frame is removed
    assert(approx_eq(m_buoyancy_acc.x + gravity.x, gpu_mem.particles_const_acc_x()[m_global_id]));
    assert(approx_eq(m_buoyancy_acc.y + gravity.y, gpu_mem.particles_const_acc_y()[m_global_id]));

    assert(approx_eq(m_spring_acc.x, gpu_mem.particles_spring_acc_x()[m_global_id]));
    assert(approx_eq(m_spring_acc.y, gpu_mem.particles_spring_acc_y()[m_global_id]));
    
    assert(approx_eq(m_acc.x, gpu_mem.particles_net_acc_x()[m_global_id]));
    assert(approx_eq(m_acc.y, gpu_mem.particles_net_acc_y()[m_global_id]));
    
    assert(approx_eq(m_vel.x, gpu_mem.particles_vel_x()[m_global_id]));
    // int a = m_vel.y;
    // int b = gpu_mem.particles_vel_y()[m_global_id];
    assert(approx_eq(m_vel.y, gpu_mem.particles_vel_y()[m_global_id]));
    
    assert(approx_eq(m_curr_pos.x, gpu_mem.particles_pos_x()[m_global_id]));
    assert(approx_eq(m_curr_pos.y, gpu_mem.particles_pos_y()[m_global_id]));
}

sf::Vector2<float> Particle::get_curr_pos() const{
    return m_curr_pos;
}
sf::Vector2<float> Particle::get_curr_pos_gpu() const{
    return {gpu_mem.particles_pos_x()[m_global_id],
                gpu_mem.particles_pos_y()[m_global_id]};
}

sf::Vector2<float> Particle::get_vel() const{
    return m_vel;
}
sf::Vector2<float> Particle::get_vel_gpu() const{
    return {gpu_mem.particles_vel_x()[m_global_id],
                gpu_mem.particles_vel_y()[m_global_id]};
}

float Particle::get_radius() const{
    return m_radius;
    // return gpu_mem.particles_radius()[m_global_id];
}

float Particle::get_mass() const{ return m_mass; }
float Particle::get_sqrt_mass() const{ return m_sqrt_mass; }

int Particle::get_global_id() const{
    return m_global_id;
}
int Particle::get_local_id() const{
    return m_id;
}

void Particle::set_acc(sf::Vector2<float> acc){
    m_acc = acc;

    if(m_env_id >= 0){
        gpu_mem.particles_net_acc_x()[m_global_id] = acc.x;
        gpu_mem.particles_net_acc_y()[m_global_id] = acc.y;

        // check();
    }

}

float Particle::calculate_total_energy(){
    // float speed = m_vel.length();
    // return 0.5f*m_mass*(speed*speed) 
    //     + m_mass*(m_acc.y+gravity.y)*(max_y-m_curr_pos.y) + m_mass*(m_acc.x+gravity.x)*(max_y-m_curr_pos.x);
    return 0;
}

void Particle::add_spring_acc(sf::Vector2f spring_acc){
    m_spring_acc += spring_acc;

    // if(m_env_id >= 0){
    //     gpu_mem.particles_spring_acc_x()[m_global_id] += spring_acc.x;
    //     gpu_mem.particles_spring_acc_y()[m_global_id] += spring_acc.y;
     
        // do not check here bc in gpu all springs have written thier acc
        // but in cpu not all springs are done
        // check();
    // }

}

float Particle::get_bounding_box_wall(direction dir){
    switch (dir) {
        
        case direction::left:
            return m_curr_pos.x - m_radius;
        
        case direction::right:
            return m_curr_pos.x + m_radius;
    }
}

void Particle::first_half_step(){

    // this is wasteful remove this -- done
    // float damping_factor = fmax(0.f, 1-(viscosity*m_radius*env_dt)/(m_mass));

    m_vel += + 0.5f * m_acc * env_dt;
    m_curr_pos += m_vel*env_dt;
}

void Particle::second_half_step(){
    
    // m_acc = gravity + m_buoyancy_acc + m_spring_acc - m_damping_factor*m_vel;
    m_acc = gravity + m_buoyancy_acc + m_spring_acc;
    m_spring_acc.x = 0; m_spring_acc.y = 0;
    
    m_vel = m_vel + 0.5f * m_acc * env_dt;

    // ? is it needed to check this so often -- it has many if conditions -- jumps -- slow
    // regular checks imp since if particle goes to0 far inside the wall then when finally refelcted it may skip a particles which should have collided
    // ? but how regular
    handle_boundary();
}

void Particle::handle_boundary(){
    if(m_curr_pos.x + m_radius > max_x)
        reflect(max_x, 0, 1);
    else if(m_curr_pos.x - m_radius < 0)
        reflect(0, 0, -1);
    
    if(m_curr_pos.y + m_radius > max_y)
        reflect(max_y, 1, 1);
    else if(m_curr_pos.y - m_radius < 0)
        reflect(0, 1, -1);
}

// sign indicates weather we need to add/subtract radius to get offending point on the circle
void Particle::reflect(float wall, int axis, int sign){
    float signed_radius = sign*m_radius;
    if(axis == 0){
        m_curr_pos.x = 2*wall - (m_curr_pos.x + signed_radius) - signed_radius;        
        m_vel.x = -m_vel.x;
    }
    else if(axis == 1){
        m_curr_pos.y = 2*wall - (m_curr_pos.y + signed_radius) - signed_radius;
        m_vel.y = -m_vel.y;
    }
}

void Particle::reset(){
    m_vel = m_original_vel;
    m_curr_pos = m_original_curr_pos;
    m_acc = m_buoyancy_acc + gravity;
    
    if(m_env_id >= 0){

        gpu_mem.particles_pos_x()[m_global_id] = m_original_curr_pos.x;
        gpu_mem.particles_pos_y()[m_global_id] = m_original_curr_pos.y;
        
        gpu_mem.particles_vel_x()[m_global_id] = m_original_vel.x;
        gpu_mem.particles_vel_y()[m_global_id] = m_original_vel.y;
    }
    set_acc(m_buoyancy_acc + gravity);
}