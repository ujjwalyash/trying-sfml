#pragma once
#include "particle.hpp"

class Spring{

    private:

        // bad -- suppose we add new particles to the particles vector which causes it to expand
        //        then these refernces will be invalidated -- hold integer indicies instead
        Particle& m_p1;
        Particle& m_p2;

    protected: // class Muscle needs to vary these hence not private

        float m_natural_length;
        float m_spring_constant;
        float m_sqrt_spring_constant;

    public:

        Spring(Particle& p1, Particle& p2, float len, float spring_const);

        std::array<sf::Vertex, 2> get_line();

        float calculate_total_energy();

        void calculate_spring_force(float dt);
    
    private:
    
        void calculate_damping_force(sf::Vector2f vec_along_spring, float m1, float m2, float sqrt_m1, float sqrt_m2, float dt);
};

void handle_all_springs(std::vector<Spring> &springs, float dt);