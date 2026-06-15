#include "particle.hpp"
#include "spring.hpp"
#include "creature.hpp"
#include "muscle.hpp"
#include <SFML/Graphics/CircleShape.hpp>
#include <SFML/Graphics/Color.hpp>
#include <SFML/Graphics/RenderWindow.hpp>
#include <SFML/Graphics/Font.hpp>
#include <SFML/Graphics/Text.hpp>
#include <SFML/System/Clock.hpp>

class Environment
{

    private:
        // simulation params
        int m_fps = 60;
        int m_num_iterations = 16;
        float m_dt = 1.f/(m_fps*m_num_iterations);
        
        int m_num_particles = 0; 
        std::vector<Particle> m_particles;
        int m_num_springs = 0;
        std::vector<Spring> m_springs;
        
        // Creature m_creature;
        int m_num_muscles = 0;
        std::vector<Muscle> m_muscles;


    public:

        Environment();

        // proceed forward with num_iterations step each with time_interval m_dt
        // also calculates rewards, end_condition, new_target pos
        // we have a set of fixed sequenece of target_pos generated randomly for each
        // generation and step will switch targets too as required
        void step();

        // the state of a environment is just the positions of all the particles in it
        // also return all muscles to natural length, and natual spring constant
        // TODO: add reset functions to particles and muscles
        // TODO: also make every thing generate in the center in creature.cpp
        void reset();

        // draws everything on window, does not clear, display
        // TODO: remove shapes from particles and just create one shape in render 
        //       and keep updating its pos
        void render();
};