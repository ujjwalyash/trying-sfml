#include "spring.hpp"
#include <SFML/Graphics/RectangleShape.hpp>
#include <SFML/Graphics/Vertex.hpp>
#include <cassert>
// #include <iostream>

Constraint::Constraint(Particle& p1, Particle& p2, float len): m_p1(p1), m_p2(p2){

    // this is too late refs cant exist without initialization hence
    // initialization needs to be done before constructor body starts
    // m_p1 = p1;
    // m_p2 = p2;
    
    assert(len > p1.get_radius() + p2.get_radius());
    m_natural_length = len;
    m_spring_constant = 1000;
}

std::array<sf::Vertex, 2> Constraint::get_line(){
    std::array<sf::Vertex, 2> line =
    {
        sf::Vertex{m_p1.get_curr_pos()},
        sf::Vertex{m_p2.get_curr_pos()}
    };

    return line;
}

float Constraint::calculate_total_energy(){
    sf::Vector2f rel_pos = m_p2.get_curr_pos() - m_p1.get_curr_pos();
    float rel_pos_length = rel_pos.length();
    float deformation = rel_pos_length - m_natural_length;

    return 0.5f * m_spring_constant * (deformation * deformation);
}

void Constraint::calculate_deformation(){

    sf::Vector2f rel_pos = m_p2.get_curr_pos() - m_p1.get_curr_pos();
    float rel_pos_length = rel_pos.length();
    float deformation = rel_pos_length - m_natural_length;

    float m1 = m_p1.get_mass();
    float m2 = m_p2.get_mass();

    m_p1.add_spring_acc(deformation * (rel_pos/rel_pos_length) * (m_spring_constant/m1));
    m_p2.add_spring_acc(deformation * (-rel_pos/rel_pos_length) * (m_spring_constant/m2));
}

void handle_all_springs(std::vector<Constraint> &springs){

    for(Constraint spring: springs){
        spring.calculate_deformation();
    }
}