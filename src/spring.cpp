#include "headers/spring.hpp"
#include <SFML/Graphics/RectangleShape.hpp>
#include <SFML/Graphics/Vertex.hpp>
#include <SFML/System/Vector2.hpp>
#include <cassert>
#include <cmath>
#include <iostream>

Spring::Spring(Particle& p1, Particle& p2, int env_id, int idx, float len, float spring_const, bool outside_body)

    :m_p1(p1), m_p2(p2),
     m_present_outside_body(outside_body),
     m_mass(m_p1.get_mass()+m_p2.get_mass()),
     m_radius(0.5f*(m_p1.get_radius()+m_p2.get_radius())),
     m_natural_length(len), m_spring_constant(spring_const),
     m_id(idx),
     m_env_id(env_id)
{
    // this is too late refs cant exist without initialization hence
    // initialization needs to be done before constructor body starts
    // m_p1 = p1;
    // m_p2 = p2;
    assert(len > p1.get_radius() + p2.get_radius());

    m_viscous_factor = (viscosity * m_natural_length / ((log(m_natural_length/m_radius) + 0.5) * m_mass));

    float d = (m_p1.get_mass()-m_p2.get_mass())*m_natural_length / (2*m_mass);
    // dont multiply by mass since damping factor already has mass
    m_moment_interia_along_com = (m_natural_length*m_natural_length/12 + d*d) * 0.1f;
    
    float m1 = m_p1.get_mass();
    float m2 = m_p2.get_mass();
    m_damping_factor = sqrtf(m_spring_constant * (m1+m2)/(m1*m2));

    // std::cout << "spring env_id, id: " << env_id << " " <<  idx << '\n';

    if(env_id >= 0){
        constexpr int total_springs_per_env = env_num_springs + env_num_muscles;
        gpu_mem.springs_nat_len()[env_id*total_springs_per_env + idx] = m_natural_length;
        assert(fabs(m_natural_length - (m_p1.get_curr_pos() - m_p2.get_curr_pos()).length()) < 0.1);
        gpu_mem.springs_const()[env_id*total_springs_per_env + idx] = m_spring_constant;
        gpu_mem.springs_damping_factor()[env_id*total_springs_per_env + idx] = m_damping_factor;
        gpu_mem.springs_viscous_factor()[env_id*total_springs_per_env + idx] = m_viscous_factor;
        gpu_mem.springs_moi_along_com()[env_id*total_springs_per_env + idx] = m_moment_interia_along_com;
        gpu_mem.springs_outside_body()[env_id*total_springs_per_env + idx] = m_present_outside_body;
        gpu_mem.springs_radius()[env_id*total_springs_per_env + idx] = m_radius;

        gpu_mem.springs_p1()[env_id*total_springs_per_env + idx] = m_p1.get_local_id();
        gpu_mem.springs_p2()[env_id*total_springs_per_env + idx] = m_p2.get_local_id();
    }
}

std::array<sf::Vertex, 2> Spring::get_line(){

    sf::Color color = sf::Color::White;
    
    // sf::Vector2f spring_vector = m_p2.get_curr_pos() - m_p1.get_curr_pos();
    // sf::Vector2f vel_com = (m_p2.get_vel() * m_p2.get_mass() + m_p1.get_vel() * m_p1.get_mass()) / (m_p1.get_mass() + m_p2.get_mass());
    // if(m_present_outside_body){
    //     // this flipped bc y axis is downward .perpendicualar rotates 90 deg clockwise not anti clock
    //     if(vel_com.dot(-spring_vector.perpendicular()) > 0){
    //     // if(vel_com.dot(spring_vector.perpendicular()) > 0){
    //         color = sf::Color::Red;
    //     }
    //     else{
    //         color = sf::Color::Green;
    //     }
    // }

    std::array<sf::Vertex, 2> line =
    {
        sf::Vertex{m_p1.get_curr_pos(),color},
        sf::Vertex{m_p2.get_curr_pos(),color}
    };

    return line;
}
std::array<sf::Vertex, 2> Spring::get_line_gpu(){

    sf::Color color = sf::Color::White;
    sf::Vector2f spring_vector = m_p2.get_curr_pos_gpu() - m_p1.get_curr_pos_gpu();
    sf::Vector2f vel_com = (m_p2.get_vel_gpu() * m_p2.get_mass() + m_p1.get_vel_gpu() * m_p1.get_mass()) / (m_p1.get_mass() + m_p2.get_mass());

    if(m_present_outside_body){
        // this flipped bc y axis is downward .perpendicualar rotates 90 deg clockwise not anti clock
        if(vel_com.dot(-spring_vector.perpendicular()) > 0){
        // if(vel_com.dot(spring_vector.perpendicular()) > 0){
            color = sf::Color::Red;
        }
        else{
            color = sf::Color::Green;
        }
    }

    std::array<sf::Vertex, 2> line =
    {
        sf::Vertex{m_p1.get_curr_pos_gpu(),color},
        sf::Vertex{m_p2.get_curr_pos_gpu(),color}
    };

    return line;
}

float Spring::calculate_total_energy(){
    sf::Vector2f rel_pos = m_p2.get_curr_pos() - m_p1.get_curr_pos();
    float rel_pos_length = rel_pos.length();
    float deformation = rel_pos_length - m_natural_length;

    return 0.5f * m_spring_constant * (deformation * deformation);
}

float Spring::get_bounding_box_wall(direction dir){
    switch (dir) {
        
        case direction::left:
            return fmin(m_p1.get_bounding_box_wall(dir), m_p2.get_bounding_box_wall(dir));
            
        case direction::right:
            return fmax(m_p1.get_bounding_box_wall(dir), m_p2.get_bounding_box_wall(dir));    
    }
}


void Spring::calculate_spring_force(){

    // ! this direction is important -- p1 to p2 not the other way
    sf::Vector2f rel_pos = m_p2.get_curr_pos() - m_p1.get_curr_pos();

    float rel_pos_length = rel_pos.length();
    float deformation = rel_pos_length - m_natural_length;

    float m1 = m_p1.get_mass();
    float m2 = m_p2.get_mass();

    calculate_damping_force(rel_pos, m1, m2);

    m_p1.add_spring_acc(deformation * (rel_pos/rel_pos_length) * (m_spring_constant/m1));
    m_p2.add_spring_acc(deformation * (-rel_pos/rel_pos_length) * (m_spring_constant/m2));
}

void Spring::calculate_damping_force(sf::Vector2f vec_along_spring, float m1, float m2){

    sf::Vector2f v1_along_spring = m_p1.get_vel().projectedOnto(vec_along_spring);
    sf::Vector2f v2_along_spring = m_p2.get_vel().projectedOnto(-vec_along_spring);

    sf::Vector2f vel_com = (m1*m_p1.get_vel() + m2*m_p2.get_vel())/(m1+m2);
    sf::Vector2f vel_com_along_spring = vel_com.projectedOnto(vec_along_spring);

    if(m_present_outside_body){
    //     calculate_viscous_force(vel_com, vec_along_spring, vel_com_along_spring, m1, m2);
        calculate_viscous_force(vec_along_spring);
    }

    // sf::Vector2f vel = ((m_p2.get_vel() - m_p1.get_vel())/(m1+m2)).projectedOnto(vec_along_spring);
    // m_p1.add_spring_acc(-m_damping_factor * m2 * (-vel));
    // m_p2.add_spring_acc(-m_damping_factor * m1 * (vel));
    m_p1.add_spring_acc(-1.f*m_damping_factor * (v1_along_spring-vel_com_along_spring));
    m_p2.add_spring_acc(-1.f*m_damping_factor * (v2_along_spring-vel_com_along_spring));
}

void Spring::calculate_viscous_force(sf::Vector2f const& vec_along_spring){
    
    //! wtf is this brother if the vel com is 0 but the spring is rotating then no force???
    //! let me explain: since a spring is small length we ignore its rotation smth lile that
    // sf::Vector2f perp_viscous_acc = {0, 0};
    // if(vel_com.dot(-vec_along_spring.perpendicular()) > 0){
    //     ////apply perp force too
        
    //     // perp to len
        
    // }
    // additional 0.5f bc only one side will be exposed to the outside
    // sf::Vector2f perp_viscous_acc = -(fminf(1.f/env_dt, 1.f * m_viscous_factor)) * (vel_com - vel_com_along_spring);
    // sf::Vector2f parallel_viscous_acc = -(fminf(1.f/env_dt, 0.25f * m_viscous_factor)) * vel_com_along_spring;
    
    // float angular_acc = parallel_viscous_acc.length() * m_radius / m_moment_interia_along_com;
    
    // sf::Vector2f dir_perp_spring = {0, 0};
    // if(angular_acc != 0){
    //     dir_perp_spring = parallel_viscous_acc.perpendicular().normalized();
    // }
    
    // m_p1.add_spring_acc(perp_viscous_acc + 1.f*parallel_viscous_acc + dir_perp_spring * (angular_acc * m2/m_mass * m_natural_length));
    // m_p2.add_spring_acc(perp_viscous_acc + 1.f*parallel_viscous_acc - dir_perp_spring * (angular_acc * m1/m_mass * m_natural_length));
    // m_p1.add_spring_acc(perp_viscous_acc + 1.f*parallel_viscous_acc);
    // m_p2.add_spring_acc(perp_viscous_acc + 1.f*parallel_viscous_acc);
    
    //* //////////////////////
    // 1. Ensure we have a normalized direction vector for accurate projection
    float len = vec_along_spring.length();
    sf::Vector2f dir_parallel = (len > 0.0001f) ? (vec_along_spring / len) : sf::Vector2f(1, 0);

    // 2. Anisotropic drag multipliers
    // Slender body theory dictates perpendicular drag is significantly higher
    float c_parallel = 0.25f * m_viscous_factor;
    float c_perp = 1.0f * m_viscous_factor; 

    // Helper lambda to apply independent drag to a particle
    auto apply_anisotropic_drag = [&](Particle& p) {
        sf::Vector2f v = p.get_vel();
        
        // Decompose the particle's velocity
        float v_dot_dir = v.x * dir_parallel.x + v.y * dir_parallel.y;
        sf::Vector2f v_parallel = dir_parallel * v_dot_dir;
        sf::Vector2f v_perp = v - v_parallel;
        
        // Calculate drag acceleration.
        // Because m_viscous_factor is already scaled by 1/m_total, 
        // applying it directly as an acceleration scales correctly for both masses.
        sf::Vector2f acc_parallel = -fminf(1.f/env_dt, c_parallel) * v_parallel;
        sf::Vector2f acc_perp = -fminf(1.f/env_dt, c_perp) * v_perp;
        
        p.add_spring_acc(acc_parallel + acc_perp);
    };

    // 3. Apply the force to both ends independently
    apply_anisotropic_drag(m_p1);
    apply_anisotropic_drag(m_p2);

    // 1. Ensure we have a normalized direction vector for accurate projection
    // float len = vec_along_spring.length();
    // sf::Vector2f dir_parallel = (len > 0.0001f) ? (vec_along_spring / len) : sf::Vector2f(1, 0);

    // // 2. Anisotropic drag multipliers
    // // Slender body theory: perpendicular drag is significantly higher than parallel drag
    // float c_parallel = 0.25f * m_viscous_factor;
    // float c_perp = 1.0f * m_viscous_factor; 

    // // 3. Decompose the Center of Mass velocity, NOT individual particle velocity
    // float v_com_dot_dir = vel_com.x * dir_parallel.x + vel_com.y * dir_parallel.y;
    // sf::Vector2f v_com_parallel = dir_parallel * v_com_dot_dir;
    // sf::Vector2f v_com_perp = vel_com - v_com_parallel;

    // // 4. Calculate drag acceleration based on the body's movement through fluid
    // sf::Vector2f acc_parallel = -fminf(1.f/env_dt, c_parallel) * v_com_parallel;
    // sf::Vector2f acc_perp = -fminf(1.f/env_dt, c_perp) * v_com_perp;
    // sf::Vector2f total_viscous_acc = acc_parallel + acc_perp;

    // // 5. Apply the shared fluid resistance acceleration directly to both ends
    // m_p1.add_spring_acc(total_viscous_acc);
    // m_p2.add_spring_acc(total_viscous_acc);
}

void handle_all_springs(std::vector<Spring> &springs){

    for(Spring& spring: springs){
        spring.calculate_spring_force();
    }
}

void Spring::check(){
    
    // for debugging
    auto approx_eq = [&](float a, float b){
        float perct_error = 1e-4; // pct    
        float diff = fabs(a - b);
        bool result = (diff < perct_error) || (diff/fmax(fabs(a), fabs(b)) < perct_error);
        // result = result * (fabs(a) < 1e-6 || fabs(b/a) < 5);
    
        if(!result){
            std::fprintf(stderr, "\nAssert failed: %f -- %f\n env_id, local_id: %d, %d \n", a, b, m_env_id, m_id);
            // std::fprintf(stderr, "\nAssert failed: %f -- %f\n env_id, local_id: %d, %d \n", a, b, m_env_id, m_id);
        }
    
        // return true;
        return result;
    };

    int x = env_num_muscles + env_num_springs;
    assert(approx_eq(m_natural_length, gpu_mem.springs_nat_len()[m_env_id * x + m_id]));
}