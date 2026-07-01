#include "headers/spring.hpp"
#include <SFML/Graphics/RectangleShape.hpp>
#include <SFML/Graphics/Vertex.hpp>
#include <SFML/System/Vector2.hpp>
#include <cassert>
#include <cmath>
// #include <iostream>

Spring::Spring(Particle& p1, Particle& p2, float len, float spring_const, bool outside_body)

    :m_p1(p1), m_p2(p2),
     m_present_outside_body(outside_body),
     m_mass(m_p1.get_mass()+m_p2.get_mass()),
     m_radius(0.5f*(m_p1.get_radius()+m_p2.get_radius())),
     m_natural_length(len), m_spring_constant(spring_const)
{
    // this is too late refs cant exist without initialization hence
    // initialization needs to be done before constructor body starts
    // m_p1 = p1;
    // m_p2 = p2;
    assert(len > p1.get_radius() + p2.get_radius());

    m_viscous_factor = (viscosity * m_natural_length / ((log(m_natural_length/m_radius) + 0.5) * m_mass));

    float d = (m_p1.get_mass()-m_p2.get_mass())*m_natural_length / (2*m_mass);
    // dont multiply by mass since damping factor already has mass
    m_moment_interia_along_com = (m_natural_length*m_natural_length/12 + d*d);
    
    float m1 = m_p1.get_mass();
    float m2 = m_p2.get_mass();
    m_damping_factor = sqrtf(m_spring_constant * (m1+m2)/(m1*m2));
}

std::array<sf::Vertex, 2> Spring::get_line(){

    sf::Color color = sf::Color::White;
    sf::Vector2f spring_vector = m_p2.get_curr_pos() - m_p1.get_curr_pos();
    sf::Vector2f vel_com = (m_p2.get_vel() * m_p2.get_mass() + m_p1.get_vel() * m_p1.get_mass()) / (m_p1.get_mass() + m_p2.get_mass());

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
        sf::Vertex{m_p1.get_curr_pos(),color},
        sf::Vertex{m_p2.get_curr_pos(),color}
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

    calculate_damping_force(rel_pos, m1, m2, m_p1.get_sqrt_mass(), m_p2.get_sqrt_mass());

    m_p1.add_spring_acc(deformation * (rel_pos/rel_pos_length) * (m_spring_constant/m1));
    m_p2.add_spring_acc(deformation * (-rel_pos/rel_pos_length) * (m_spring_constant/m2));
}

void Spring::calculate_damping_force(sf::Vector2f vec_along_spring, float m1, float m2, float sqrt_m1, float sqrt_m2){

    sf::Vector2f v1_along_spring = m_p1.get_vel().projectedOnto(vec_along_spring);
    sf::Vector2f v2_along_spring = m_p2.get_vel().projectedOnto(-vec_along_spring);

    sf::Vector2f vel_com = (m1*m_p1.get_vel() + m2*m_p2.get_vel())/(m1+m2);
    sf::Vector2f vel_com_along_spring = vel_com.projectedOnto(vec_along_spring);

    if(m_present_outside_body){
        calculate_viscous_force(vel_com, vec_along_spring, vel_com_along_spring, m1, m2);
    }

    // change to using reduced mass later -- that is faster
    // why didnt i jsut store this >?????
    // float spring_const_for_p1 = m_spring_constant * (m1+m2)/m2;
    // float spring_const_for_p2 = m_spring_constant * (m1+m2)/m1;

    // keep this underdamped do not change -1.f to -2.f
    // using fsqrt here is bad this function is called for every spring every step -- the most active func
    m_p1.add_spring_acc(-1.f*m_damping_factor * (v1_along_spring-vel_com_along_spring));
    m_p2.add_spring_acc(-1.f*m_damping_factor * (v2_along_spring-vel_com_along_spring));
}

void Spring::calculate_viscous_force(sf::Vector2f const& vel_com, sf::Vector2f const& vec_along_spring, sf::Vector2f const& vel_com_along_spring, float m1, float m2){
    
    sf::Vector2f perp_viscous_acc = {0, 0};
    if(vel_com.dot(-vec_along_spring.perpendicular()) > 0){
        ////apply perp force too
        
        // perp to len
        perp_viscous_acc = -(fminf(1.f/env_dt, 1.f * m_viscous_factor)) * (vel_com - vel_com_along_spring);
        
    }
    // additional 0.5f bc only one side will be exposed to the outside
    sf::Vector2f parallel_viscous_acc = -(fminf(1.f/env_dt, 0.25f * m_viscous_factor)) * vel_com_along_spring;
    
    float angular_acc = parallel_viscous_acc.length() * m_radius / m_moment_interia_along_com;
    
    sf::Vector2f dir_perp_spring = {0, 0};
    if(angular_acc != 0){
        dir_perp_spring = parallel_viscous_acc.perpendicular().normalized();
    }
    
    m_p1.add_spring_acc(perp_viscous_acc + parallel_viscous_acc + dir_perp_spring * (angular_acc * m2/m_mass * m_natural_length));
    m_p2.add_spring_acc(perp_viscous_acc + parallel_viscous_acc - dir_perp_spring * (angular_acc * m1/m_mass * m_natural_length));
    
}

void handle_all_springs(std::vector<Spring> &springs){

    for(Spring& spring: springs){
        spring.calculate_spring_force();
    }
}