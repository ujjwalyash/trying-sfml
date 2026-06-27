#include "headers/muscle.hpp"

Muscle::Muscle(Particle& p1, Particle& p2, float len, float spring_const
                    , bool outside_body, float contraction_limit, float max_spring_constant_scaling)
    : Spring(p1, p2, len, spring_const, outside_body),
    m_activation(0.f),
    m_rest_length(len),
    m_contraction_limit(contraction_limit),
    m_rest_spring_constant(spring_const),
    m_max_spring_constant_scaling(max_spring_constant_scaling),
    m_rest_damping_factor(m_damping_factor)
{}

void Muscle::set_activation(float acti){
    m_activation = acti;
}

void Muscle::handle_nerve_signal(){
    m_natural_length = m_rest_length*(1 - m_activation*m_contraction_limit);
    m_spring_constant = m_rest_spring_constant*(1+m_activation*m_max_spring_constant_scaling);

    m_damping_factor = (viscosity * m_natural_length / ((log(m_natural_length/m_radius) + 0.5) * m_mass)) * env_dt;
    m_damping_factor = fmin(1, m_damping_factor);

    float d = (m_p1.get_mass()-m_p2.get_mass())*m_natural_length / (2*m_mass);
    m_moment_interia_along_com = (m_natural_length*m_natural_length/12 + d*d);
}

void Muscle::reset(){
    m_natural_length = m_rest_length;
    m_spring_constant = m_rest_spring_constant;
    m_damping_factor = m_rest_damping_factor;

    // could have stored rest MOI as well but that would need 30 sec extra
    float d = (m_p1.get_mass()-m_p2.get_mass())*m_natural_length / (2*m_mass);
    m_moment_interia_along_com = (m_natural_length*m_natural_length/12 + d*d);
}

void handle_all_muscle_contraction(std::vector<Muscle> &muscles){
    int num_muscles = muscles.size();
    for(int i = 0; i < num_muscles; i++){
        muscles[i].handle_nerve_signal();
    }
}

void handle_all_muscle_forces(std::vector<Muscle> &muscles){
    int num_muscles = muscles.size();
    for(int i = 0; i < num_muscles; i++){
        muscles[i].calculate_spring_force();
    }
}