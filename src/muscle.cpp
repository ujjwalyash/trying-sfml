#include "headers/muscle.hpp"

Muscle::Muscle(Particle& p1, Particle& p2, int env_id, int idx, float len, float spring_const
                    , bool outside_body, float contraction_limit, float max_spring_constant_scaling)
    : Spring(p1, p2, env_id, idx, len, spring_const, outside_body),
    m_activation(0.f),
    m_rest_length(len),
    m_contraction_limit(contraction_limit),
    m_rest_spring_constant(spring_const),
    m_max_spring_constant_scaling(max_spring_constant_scaling),
    m_rest_damping_factor(m_damping_factor)
{
    if(m_env_id >= 0){
        int m = idx - env_num_springs;
        gpu_mem.muscle_rest_nat_len[env_id * env_num_muscles + m] = m_rest_length;
        gpu_mem.muscle_rest_spring_const[env_id * env_num_muscles + m] = m_rest_spring_constant;
        gpu_mem.muscle_contraction_limit[env_id * env_num_muscles + m] = m_contraction_limit;
        gpu_mem.muscle_max_const_scaling[env_id * env_num_muscles + m] = m_max_spring_constant_scaling;
    }
}

void Muscle::set_activation(float acti){
    m_activation = acti;
}

void Muscle::handle_nerve_signal(){
    m_natural_length = m_rest_length*(1 - m_activation*m_contraction_limit);
    m_spring_constant = m_rest_spring_constant*(1+m_activation*m_max_spring_constant_scaling);

    // m_damping_factor = (viscosity * m_natural_length / ((log(m_natural_length/m_radius) + 0.5) * m_mass)) * env_dt;
    // m_damping_factor = fmin(1, m_damping_factor);

    // TODO add change to viscous factor too -- dont

    // float d = (m_p1.get_mass()-m_p2.get_mass())*m_natural_length / (2*m_mass);
    // m_moment_interia_along_com = (m_natural_length*m_natural_length/12 + d*d);

    // if(m_env_id >= 0){
    //     constexpr int total_springs_per_env = env_num_springs + env_num_muscles;
    //     gpu_mem.springs_nat_len()[m_env_id*total_springs_per_env + m_id] = m_natural_length;
    //     gpu_mem.springs_const()[m_env_id*total_springs_per_env + m_id] = m_spring_constant;

    //     gpu_mem.springs_damping_factor()[m_env_id*total_springs_per_env + m_id] = m_damping_factor;
    //     gpu_mem.springs_moi_along_com()[m_env_id*total_springs_per_env + m_id] = m_moment_interia_along_com;
    // }
}

void Muscle::reset(){
    m_natural_length = m_rest_length;
    m_spring_constant = m_rest_spring_constant;
    // m_damping_factor = m_rest_damping_factor;

    // could have stored rest MOI as well but that would need 30 sec extra
    // float d = (m_p1.get_mass()-m_p2.get_mass())*m_natural_length / (2*m_mass);
    // m_moment_interia_along_com = (m_natural_length*m_natural_length/12 + d*d);

    if(m_env_id >= 0){
        constexpr int total_springs_per_env = env_num_springs + env_num_muscles;
        gpu_mem.springs_nat_len()[m_env_id*total_springs_per_env + m_id] = m_natural_length;
        gpu_mem.springs_const()[m_env_id*total_springs_per_env + m_id] = m_spring_constant;

        // gpu_mem.springs_damping_factor()[m_env_id*total_springs_per_env + m_id] = m_damping_factor;
        // gpu_mem.springs_moi_along_com()[m_env_id*total_springs_per_env + m_id] = m_moment_interia_along_com;
    }
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