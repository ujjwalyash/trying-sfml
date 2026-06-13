#pragma once
#include "particle.hpp"
#include "spring.hpp"

class Muscle : public Spring
{
    private:
        float m_activation; // in [0, 1]
        
        float m_rest_length;
        float m_contraction_limit;

        float m_rest_spring_constant;
        float m_max_spring_constant_scaling;

    public:

        Muscle(Particle& p1, Particle& p2, float len, float spring_const, float contraction_limit=0.4, float max_spring_constant_scaling=10);
        
        void handle_nerve_signal();
        void set_activation(float acti);
        
};

void handle_all_muscles(std::vector<Muscle> &muscles, float dt);