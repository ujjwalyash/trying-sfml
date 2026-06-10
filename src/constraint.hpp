#pragma once
#include "particle.hpp"

class Constraint{

    private:
        Particle& m_p1;
        Particle& m_p2;

        float m_natural_length;

    public:
        Constraint(Particle& p1, Particle& p2, float len);

        std::array<sf::Vertex, 2> get_line();

        void calculate_deformation();
        void apply_deformation_to_particles();
};

void handle_all_constraints(std::vector<Constraint> &constraints);