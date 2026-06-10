#include "constraint.hpp"
#include <SFML/Graphics/RectangleShape.hpp>
#include <SFML/Graphics/Vertex.hpp>
// #include <iostream>

Constraint::Constraint(Particle& p1, Particle& p2, float len): m_p1(p1), m_p2(p2){

    // this is too late refs cant exist without initialization hence
    // needs to be done before constructor body starts
    // m_p1 = p1;
    // m_p2 = p2;
 
    m_natural_length = len;
}

std::array<sf::Vertex, 2> Constraint::get_line(){
    std::array<sf::Vertex, 2> line =
    {
        sf::Vertex{m_p1.get_curr_pos()},
        sf::Vertex{m_p2.get_curr_pos()}
    };

    return line;
}

// returns the total (by p1 + by p2) length to be moved by p1 and p2
void Constraint::calculate_deformation(){

    sf::Vector2f rel_pos = m_p2.get_curr_pos() - m_p1.get_curr_pos();
    float rel_pos_length = rel_pos.length();
    float deformation = rel_pos_length - m_natural_length;

    float m1 = m_p1.get_mass();
    float m2 = m_p2.get_mass();

    m_p1.add_constraint_update(deformation * (rel_pos/rel_pos_length) * (m2/(m1+m2)));
    m_p2.add_constraint_update(deformation * (-rel_pos/rel_pos_length) * (m1/(m1+m2)));
}

void Constraint::apply_deformation_to_particles(){
    m_p1.handle_constraint_update();
    m_p2.handle_constraint_update();
}

void handle_all_constraints(std::vector<Constraint> &constraints){

    for(Constraint constraint: constraints){
        constraint.calculate_deformation();
    }

    for(Constraint constraint: constraints){
        constraint.apply_deformation_to_particles();
    }

}