#pragma once
#include <SFML/Graphics/CircleShape.hpp>
#include <SFML/System/Vector2.hpp>

// params
extern const float max_y;
extern const float max_x;

extern const sf::Vector2f gravity;
extern const float buoyancy_const; // DO NOT MAKE DENSITY 1000

extern const float restitution;
extern const float coefficient_friction;
extern const float viscosity; // *10 bc viscosity is only applied at point masses which have small radius so we scale it


enum class structure{
    creature,
    ball
};

class Particle{
    private:

        int m_id;

        sf::Vector2f m_original_old_pos;
        sf::Vector2f m_original_curr_pos;

        sf::Vector2f m_old_pos;
        sf::Vector2f m_curr_pos;
        
        sf::Vector2f m_spring_acc;
        sf::Vector2f m_buoyancy_acc;
        sf::Vector2f m_acc;

        // this circle shape is 344 bytes ?? try to do something for this -- maybe do not hold the shapes and create just one circle and modify
                                                                                    // it according to points before rendering
        // removed
        // sf::CircleShape m_body_shape{};
        
        float m_radius;
        float m_cube_radius;
        float m_mass;
        float m_sqrt_mass;
        
        // maybe make radius, mass public const too 
        // but it causes default copy constructor to fail
    public:
        const structure m_type;

    public:
        Particle(int id, float radius, float mass, sf::Vector2<float> old_pos, sf::Vector2<float> curr_pos, structure type);

        void set_pos(sf::Vector2<float> old_pos, sf::Vector2<float> curr_pos);
        void shift_pos(sf::Vector2<float> shift);
        void set_acc(sf::Vector2<float> acc);
        
        sf::Vector2<float> get_curr_pos() const;
        sf::Vector2<float> get_old_pos() const;
        float get_radius() const;
        float get_mass() const;
        float get_sqrt_mass() const;
        int get_id() const;
        float calculate_total_energy(float dt);

        void add_spring_acc(sf::Vector2f spring_acc);
        
        void step(float dt);
        void handle_boundary();
        void reflect(float wall, int axis, int sign);
        void reset();

        // why pass by non const reference -- if you want this just make the shape public
        // change it to const reference in return value
        // const sf::CircleShape& get_shape();
};

void handle_two_body_collision(Particle& p1, Particle& p2);
void handle_all_collisions(std::vector<Particle>& particles);