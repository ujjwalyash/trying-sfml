#pragma once
#include <SFML/Graphics/CircleShape.hpp>
#include <SFML/System/Vector2.hpp>

class Particle{
    private:
        sf::Vector2f m_old_pos;
        sf::Vector2f m_curr_pos;

        sf::Vector2f m_acc;

        sf::CircleShape m_body_shape{};
        float m_radius;
        float m_mass;
        
    public:
        Particle();

        void set_pos(sf::Vector2<float> old_pos, sf::Vector2<float> curr_pos);
        void set_acc(sf::Vector2<float> acc);
        
        sf::Vector2<float> get_curr_pos() const;
        sf::Vector2<float> get_old_pos() const;
        float get_radius() const;
        float get_mass() const;

        void step(float dt);
        void handle_boundary_constraint();
        void bounce(float wall, int axis, sf::Vector2f new_pos);
        sf::CircleShape& get_shape();
};

void handle_two_body_collision(Particle& p1, Particle& p2);
void handle_all_collisions(std::vector<Particle>& particles);