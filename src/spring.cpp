#include "headers/spring.hpp"
#include <SFML/Graphics/RectangleShape.hpp>
#include <SFML/Graphics/Vertex.hpp>
#include <SFML/System/Vector2.hpp>
#include <cassert>
#include <cmath>
// #include <iostream>

Spring::Spring(Particle& p1, Particle& p2, float len, float spring_const)

    : m_p1(p1), m_p2(p2),
    m_natural_length(len), m_spring_constant(spring_const),
    m_sqrt_spring_constant(sqrt(m_spring_constant))
{
    // this is too late refs cant exist without initialization hence
    // initialization needs to be done before constructor body starts
    // m_p1 = p1;
    // m_p2 = p2;
    
    assert(len > p1.get_radius() + p2.get_radius());
}

std::array<sf::Vertex, 2> Spring::get_line(){
    std::array<sf::Vertex, 2> line =
    {
        sf::Vertex{m_p1.get_curr_pos()},
        sf::Vertex{m_p2.get_curr_pos()}
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

    sf::Vector2f vel_com = (m1*v1_along_spring + m2*v2_along_spring)/(m1+m2);

    // change to using reduced mass later -- that is faster
    float spring_const_for_p1 = m_spring_constant * (m1+m2)/m2;
    float spring_const_for_p2 = m_spring_constant * (m1+m2)/m1;

    // keep this underdamped do not change -1.f to -2.f
    m_p1.add_spring_acc(-1.f*fsqrt(spring_const_for_p1)/sqrt_m1 * (v1_along_spring-vel_com));
    m_p2.add_spring_acc(-1.f*fsqrt(spring_const_for_p2)/sqrt_m2 * (v2_along_spring-vel_com));
}

// void Spring::calculate_viscous_force(){
    
// }

void handle_all_springs(std::vector<Spring> &springs){

    for(Spring& spring: springs){
        spring.calculate_spring_force();
    }
}