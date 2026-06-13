#include "headers/muscle.hpp"

Muscle::Muscle(Particle& p1, Particle& p2, float len, float spring_const
                    , float contraction_limit, float max_spring_constant_scaling)
    : Spring(p1, p2, len, spring_const),
    m_activation(0.f),
    m_rest_length(len),
    m_contraction_limit(contraction_limit),
    m_rest_spring_constant(spring_const),
    m_max_spring_constant_scaling(max_spring_constant_scaling)
{}

void Muscle::set_activation(float acti){
    m_activation = acti;
}

void Muscle::handle_nerve_signal(){
    m_natural_length = m_rest_length*(1 - m_activation*m_contraction_limit);
    m_spring_constant = m_rest_spring_constant*(1+m_activation*m_max_spring_constant_scaling);
}


void handle_all_muscles(std::vector<Muscle> &muscles, float dt){
    int num_muscles = muscles.size();
    for(int i = 0; i < num_muscles; i++){
        muscles[i].handle_nerve_signal();
        muscles[i].calculate_spring_force(dt);
    }
}