#pragma once
#include "particle.hpp"

class Constraint{

    private:

        Particle& m_p1;
        Particle& m_p2;

        float m_natural_length;
        float m_spring_constant;

    public:
        Constraint(Particle& p1, Particle& p2, float len);

        std::array<sf::Vertex, 2> get_line();

        float calculate_total_energy();

        void calculate_deformation();
        void apply_deformation_to_particles();
};

void handle_all_springs(std::vector<Constraint> &springs);