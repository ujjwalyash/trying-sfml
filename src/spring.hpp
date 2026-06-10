#pragma once
#include "particle.hpp"

class Constraint{

    private:

        Particle& m_p1;
        Particle& m_p2;

        float m_natural_length;
        float m_spring_constant;
        float m_damping_constant;

    public:
        Constraint(Particle& p1, Particle& p2, float len, float spring_const);

        std::array<sf::Vertex, 2> get_line();

        float calculate_total_energy();

        void calculate_spring_force(float dt);
        void calculate_damping_force(sf::Vector2f vec_along_spring, float m1, float m2, float sqrt_m1, float sqrt_m2, float dt);
        void apply_deformation_to_particles();
};

void handle_all_springs(std::vector<Constraint> &springs, float dt);