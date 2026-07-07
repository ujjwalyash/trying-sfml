#pragma once
#include <SFML/Graphics/CircleShape.hpp>
#include <SFML/System/Vector2.hpp>

#include "common.hpp"

enum class direction{left, right};
// struct direction{
//     public:
//         static const int left = 0, right = 1;
// };

enum class structure{
    creature,
    ball,
    cluster
};

class Particle{
    private:

        int m_env_id;
        int m_id;
        int m_global_id;

        bool m_cpu_only;

        sf::Vector2f m_original_vel;
        sf::Vector2f m_original_curr_pos;

        sf::Vector2f m_vel;
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
        // used to determine if particle belongs to ball -- bc only ball particles are needed to be randomly shift when environment is reset 
        const structure m_type;

    public:
        Particle(int env_id, int id, float radius, float mass, sf::Vector2<float> curr_pos, sf::Vector2<float> vel, structure type, bool cpu_only = false);

        void set_pos_vel(sf::Vector2<float> curr_pos, sf::Vector2<float> vel);
        void shift_pos(sf::Vector2<float> shift);
        void set_acc(sf::Vector2<float> acc);
        
        sf::Vector2<float> get_curr_pos() const;
        sf::Vector2<float> get_vel() const;
        float get_radius() const;
        float get_mass() const;
        float get_sqrt_mass() const;
        int get_global_id() const;
        float calculate_total_energy();

        void add_spring_acc(sf::Vector2f spring_acc);
        
        void first_half_step();
        void second_half_step();

        void handle_boundary();
        void reflect(float wall, int axis, int sign);

        void reset();
        void check();

        float get_bounding_box_wall(direction dir);

        // why pass by non const reference -- if you want this just make the shape public
        // change it to const reference in return value
        // const sf::CircleShape& get_shape();
};
