#include "headers/environment.hpp"

Environment::Environment(sf::Vector2f ball_pos, sf::Vector2f goal_pos)
    // m_creature has no default constructor hence need to do this
    // ensure all function parameter are defined earlier in the CLASS DECLARATION
    {
        // CREATE THE BALL BEFROE THE CREATURE
        // if you do afterwords and the vector moves to a bigger size
        // all the springs which hold references will be invalidated
        
    m_original_ball_pos = ball_pos;
    m_goal_pos = goal_pos;
    
    // after creating football in sperm creation springs vector is resized, the springs break
    // temp fix for now
    m_particles.reserve(200);
    m_springs.reserve(300);
    m_muscles.reserve(10);
    
    m_goal_center_index = create_football(m_num_particles, m_particles, m_num_springs, m_springs, m_dt);
    for(int i = 0; i < m_num_particles; i++){
        m_particles[i].shift_pos(m_original_ball_pos);
    }
    m_creature = create_creature_muscle_sperm(m_num_particles, m_particles, m_num_springs, m_springs, m_num_muscles, m_muscles, m_dt);
    
    m_goal_radius = 10;
}

void Environment::reset(sf::Vector2f ball_pos, sf::Vector2f goal_pos){

    m_original_ball_pos = ball_pos;
    m_goal_pos = goal_pos;

    for(int i = 0; i < m_num_particles; i++){
        m_particles[i].reset();
        if(m_particles[i].m_type == structure::ball) m_particles[i].shift_pos(m_original_ball_pos);
    }
    for(int i = 0; i < m_num_muscles; i++){
        m_muscles[i].reset();
    }

    m_num_steps_done = 0;
    m_reward = 0;
    m_episode_end = false;
    m_has_touched_ball = false;
}

void Environment::render(sf::RenderWindow& window){
        int j = 0;
		sf::CircleShape sh;
		for(int i = 0; i < m_num_particles; i++){
			if(j < (int)m_creature.m_sensing_points.size() && i == m_creature.m_sensing_points[j]){
				j++;
				sf::Vector2f particle_pos = m_particles[i].get_curr_pos();
				float r = m_particles[i].get_radius();
				sh.setPosition({particle_pos.x-r, particle_pos.y-r});
				sh.setRadius(r);
				sh.setFillColor(sf::Color::Green);
				window.draw(sh);
			}
			else{
				sf::Vector2f particle_pos = m_particles[i].get_curr_pos();
				float r = m_particles[i].get_radius();
				sh.setPosition({particle_pos.x-r, particle_pos.y-r});
				sh.setRadius(r);
				sh.setFillColor(sf::Color::White);
				window.draw(sh);
			}
		}
		for(int i = 0; i < m_num_springs; i++){
			std::array line = m_springs[i].get_line();
			window.draw(line.data(), line.size(), sf::PrimitiveType::Lines);
		}
		for(int i = 0; i < m_num_muscles; i++){
			std::array line = m_muscles[i].get_line();
			window.draw(line.data(), line.size(), sf::PrimitiveType::Lines);
		}

        // draw goal
        sf::Vector2f particle_pos = m_goal_pos;
        float r = m_goal_radius;
        sh.setPosition({particle_pos.x-r, particle_pos.y-r});
        sh.setRadius(r);
        sh.setFillColor(sf::Color::Red);
        window.draw(sh);
}

void Environment::step(){
    
	std::vector<float> observation(m_observation_size);
    m_creature.get_observation(observation, m_ball_pos, m_particles);
    m_creature.act(m_muscles, observation);
    
    for(int iter = 0; iter < m_num_iterations_per_frame*m_num_frames_per_creature_action; iter++){
        handle_all_muscles(m_muscles, m_dt);
        handle_all_springs(m_springs, m_dt);
        for(int i = 0; i < m_num_particles; i++){
            m_particles[i].step(m_dt);
        }			
        handle_all_collisions(m_particles);
    }

    m_ball_pos = m_particles[m_goal_center_index].get_curr_pos();

    float reward = calculate_reward();
    m_reward += reward;

    m_num_steps_done++;
}

float Environment::calculate_reward(){
    
    float reward = 0;

    float goal_ball_dist = (m_ball_pos-m_goal_pos).length();
    float ball_displacement = (m_original_ball_pos-m_ball_pos).length();
    if(ball_displacement > 5) m_has_touched_ball = true;

    if(not m_has_touched_ball) reward -= 1;
    m_reward -= goal_ball_dist/300.f;
    m_reward -= 1; // normal time thing

    if(goal_ball_dist < m_goal_radius){
        m_reward += 1000;
        m_episode_end = 1;

    }

    return reward;
}

float Environment::get_curr_reward(){
    return m_reward;
}
