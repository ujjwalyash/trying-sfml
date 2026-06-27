#pragma once
#include "particle.hpp"

class Spring{

    protected: // class Muscle needs to access these hence not private

        // bad -- suppose we add new particles to the particles vector which causes it to expand
        //        then these refernces will be invalidated -- hold integer indicies instead
        Particle& m_p1;
        Particle& m_p2;

        // springs is from p1 to p2 and the outside is on left side of vector from p1 to p2
        bool m_present_outside_body;
        
        float m_mass;
        float m_radius;
        float m_natural_length;
        float m_spring_constant;
        float m_sqrt_spring_constant;
        
        float m_damping_factor;
        float m_moment_interia_along_com;
        
    public:

        Spring(Particle& p1, Particle& p2, float len, float spring_const, bool outside_body);

        std::array<sf::Vertex, 2> get_line();

        float calculate_total_energy();

        void calculate_spring_force();

        float get_bounding_box_wall(direction dir);
    
    private:
    
        void calculate_damping_force(sf::Vector2f vec_along_spring, float m1, float m2, float sqrt_m1, float sqrt_m2);
        void calculate_viscous_force(sf::Vector2f const& vel_com, sf::Vector2f const& vec_along_spring, sf::Vector2f const& vel_com_along_spring, float m1, float m2);
};

void handle_all_springs(std::vector<Spring> &springs);