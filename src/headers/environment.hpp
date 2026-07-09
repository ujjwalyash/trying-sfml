#pragma once
#include "particle.hpp"
#include "spring.hpp"
#include "creature.hpp"
#include "muscle.hpp"
#include "collision_handling.hpp"
#include <SFML/Graphics/CircleShape.hpp>
#include <SFML/Graphics/Color.hpp>
#include <SFML/Graphics/RenderWindow.hpp>
#include <SFML/Graphics/Text.hpp>

class Environment
{
    private:
        
        int m_id;

        int m_num_particles = 0; 
        int m_num_springs = 0;
        std::vector<Particle> m_particles;
        std::vector<Spring> m_springs;
        
        int m_num_muscles = 0;
        std::vector<Muscle> m_muscles;
        Creature m_creature;
        
        sf::Vector2f m_original_ball_pos;
        sf::Vector2f m_ball_pos;
        sf::Vector2f m_goal_pos;
        int m_goal_center_index;
        float m_goal_radius;
        bool m_has_touched_ball = false;
        
        float m_reward = 0; 
        int m_num_steps_done = 0;
        
    public:
        bool m_episode_end = 0;

        // Environment(int id, sf::Vector2f ball_pos, sf::Vector2f goal_pos);
        Environment(int id, sf::Vector2f ball_pos, sf::Vector2f goal_pos);

        // the state of a environment is just the positions of all the particles in it
        // also return all muscles to natural length, and natual spring constant
        // randomize ball and goal pos -- doing this will also include rotations for sperm
        void reset(sf::Vector2f ball_pos, sf::Vector2f goal_pos);
        
        // draws everything on window, does not clear, display
        void render(sf::RenderWindow& window);
        void render_gpu(sf::RenderWindow& window);
        
        // step till m_episode_end
        // void run_episode();
        
        // proceed forward with num_iterations step each with time_interval m_dt
        // also calculates rewards, end_condition, new_target pos
        // we have a set of fixed sequenece of target_pos generated randomly for each
        // generation and step will switch targets too as required
        // EACH STEP DOES 0.1s not 1/60 = 0.016;
        void step_stage_0();
        void step_stage_1();
        void step_stage_2();

        void step(sf::RenderWindow& window, sf::Text& text, bool render_between_cycle = false);

        // returns the reward for one step
        // checks for end condition
        float calculate_reward();
        float calculate_reward_gpu();

        float get_curr_reward();
        void reset_reward();
        Creature const& get_creature() const;

        void crossover(Environment const& par_1, Environment const& par_2);
        void mutate(float mutation_rate);
        void save(int id, float total_reard);

        void copy_brain(Environment const& env);

        void check();
};