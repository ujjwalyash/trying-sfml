#pragma once
#include <SFML/Graphics/CircleShape.hpp>
#include <SFML/System/Vector2.hpp>

class Particle{
    private:

        int m_id;

        sf::Vector2f m_old_pos;
        sf::Vector2f m_curr_pos;
        
        sf::Vector2f m_spring_acc;
        sf::Vector2f m_acc;

        // this circle shape is 344 bytes ?? try to do something for this -- maybe do not hold the shapes and create just one circle and modify
                                                                                    // it according to points before rendering
        sf::CircleShape m_body_shape{};
        float m_radius;
        float m_mass;
        
    public:
        Particle(int id, float radius, float mass);

        void set_pos(sf::Vector2<float> old_pos, sf::Vector2<float> curr_pos);
        void set_acc(sf::Vector2<float> acc);
        
        sf::Vector2<float> get_curr_pos() const;
        sf::Vector2<float> get_old_pos() const;
        float get_radius() const;
        float get_mass() const;
        int get_id() const;
        float calculate_total_energy(float dt);

        void add_spring_acc(sf::Vector2f movement);
        
        void step(float dt);
        void handle_boundary_spring();
        void reflect(float wall, int axis, int sign);
        sf::CircleShape& get_shape();
};

void handle_two_body_collision(Particle& p1, Particle& p2);
void handle_all_collisions(std::vector<Particle>& particles);