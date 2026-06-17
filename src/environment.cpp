#include "headers/environment.hpp"
#include <iostream>

const int env_fps = 60;
const int env_num_frames_per_creature_action = (float)env_fps/10; // every 100ms 
const int env_num_iterations_per_frame = 16;
const float env_dt = 1.f/(env_fps*env_num_iterations_per_frame);
const int env_max_steps_per_episode = 200;
const int env_observation_size = 12;

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
    m_particles.reserve(300);
    m_springs.reserve(300);
    m_muscles.reserve(10);
    
    m_goal_center_index = create_football(m_num_particles, m_particles, m_num_springs, m_springs, env_dt);
    for(int i = 0; i < m_num_particles; i++){
        m_particles[i].shift_pos(m_original_ball_pos);
    }

    Creature_data data = create_creature_muscle_sperm(m_num_particles, m_particles, m_num_springs, m_springs, m_num_muscles, m_muscles, env_dt);
    // 20 -- 12 -- 8
    std::vector<int> layer_sizes{m_num_muscles+env_observation_size, 12, m_num_muscles};
    // 2 wasteful initliaizations 
    // once before constructor body starts
    // once when we create this temp object and copy
    m_creature = Creature{data.s_muscle_indices, data.s_sensing_points, layer_sizes};

    m_goal_radius = 10;
}


Environment::Environment(int id, sf::Vector2f ball_pos, sf::Vector2f goal_pos)
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
    m_particles.reserve(300);
    m_springs.reserve(300);
    m_muscles.reserve(10);
    
    m_goal_center_index = create_football(m_num_particles, m_particles, m_num_springs, m_springs, env_dt);
    for(int i = 0; i < m_num_particles; i++){
        m_particles[i].shift_pos(m_original_ball_pos);
    }
    Creature_data data = create_creature_muscle_sperm(m_num_particles, m_particles, m_num_springs, m_springs, m_num_muscles, m_muscles, env_dt);
    // 20 -- 12 -- 8
    std::vector<int> layer_sizes{m_num_muscles+env_observation_size, 12, m_num_muscles};

    // loading json
    std::string file_path = std::format("./saved/{}.json", id);
    std::ifstream i(file_path);
    if(!i){
        std::cout << "could not load from file: " << file_path << '\n';
        exit(-1);
    }

    json j = json::parse(i);
    std::vector<MatrixXdf> weights = j["weights"].get<std::vector<MatrixXdf>>();
    std::vector<VectorXdf> biases = j["biases"].get<std::vector<VectorXdf>>();
    std::vector<int> lsizes = j["layer_sizes"].get<std::vector<int>>();
    m_reward = j["reward"].get<float>();
    assert(lsizes == layer_sizes);

    // 2 wasteful initliaizations 
    // once before constructor body starts
    // once when we create this temp object and copy
    m_creature = Creature{data.s_muscle_indices, data.s_sensing_points
                            , weights, biases};

    m_goal_radius = 10;
}

Creature const& Environment::get_creature() const{
    return m_creature;
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

void Environment::run_episode(){
    while(not m_episode_end){
        step();
    }
}

void Environment::mutate(float mutation_rate){
    m_creature.mutate(mutation_rate);
}

void Environment::crossover(Environment const& par_1, Environment const& par_2){
    // since par_1 is const we can only const member functions of get_creature
    m_creature.crossover(par_1.get_creature(), par_2.get_creature());
}

void Environment::save(int id, float total_reard){
    m_creature.save(id, total_reard);
}

void Environment::render(sf::RenderWindow& window){
        // int j = 0;
		sf::CircleShape sh;
		for(int i = 0; i < m_num_particles; i++){
			// if(j < (int)m_creature.m_sensing_points.size() && i == m_creature.m_sensing_points[j]){
			// 	j++;
            //     // if(j != 1 and j != 4) continue;

			// 	sf::Vector2f particle_pos = m_particles[i].get_curr_pos();
			// 	float r = m_particles[i].get_radius();
			// 	sh.setPosition({particle_pos.x-r, particle_pos.y-r});
			// 	sh.setRadius(r);
			// 	sh.setFillColor(sf::Color::Green);
			// 	window.draw(sh);
			// }
			// else{
			// 	sf::Vector2f particle_pos = m_particles[i].get_curr_pos();
			// 	float r = m_particles[i].get_radius();
			// 	sh.setPosition({particle_pos.x-r, particle_pos.y-r});
			// 	sh.setRadius(r);
			// 	sh.setFillColor(sf::Color::White);
			// 	window.draw(sh);
			// }
            sf::Vector2f particle_pos = m_particles[i].get_curr_pos();
            float r = m_particles[i].get_radius();
            sh.setPosition({particle_pos.x-r, particle_pos.y-r});
            sh.setRadius(r);
            sh.setFillColor(sf::Color::White);
            window.draw(sh);
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


// DUPLICATE FUNCTIONS bc you cant give references default params -- try pointer 
void Environment::step(){
    
	std::vector<float> observation(env_observation_size);
    m_creature.get_observation(observation, m_ball_pos, m_goal_pos, m_particles);
    m_creature.act(m_muscles, observation);
    
    for(int cycle = 0; cycle < env_num_frames_per_creature_action; cycle++){

        for(int iter = 0; iter < env_num_iterations_per_frame; iter++){
            handle_all_muscles(m_muscles, env_dt);
            handle_all_springs(m_springs, env_dt);
            for(int i = 0; i < m_num_particles; i++){
                m_particles[i].step(env_dt);
            }			
            handle_all_collisions(m_particles);    
        }
    }

    m_ball_pos = m_particles[m_goal_center_index].get_curr_pos();

    float reward = calculate_reward();
    m_reward += reward;

    m_num_steps_done++;

    if(m_num_steps_done > env_max_steps_per_episode)
        m_episode_end = true;
    
}

void Environment::step(sf::RenderWindow& window, sf::Text& text, bool render_between_cycle){
    
	std::vector<float> observation(env_observation_size);
    m_creature.get_observation(observation, m_ball_pos, m_goal_pos, m_particles);
    m_creature.act(m_muscles, observation);
    
    for(int cycle = 0; cycle < env_num_frames_per_creature_action; cycle++){

        for(int iter = 0; iter < env_num_iterations_per_frame; iter++){
            handle_all_muscles(m_muscles, env_dt);
            handle_all_springs(m_springs, env_dt);
            for(int i = 0; i < m_num_particles; i++){
                m_particles[i].step(env_dt);
            }			
            handle_all_collisions(m_particles);    
        }

        if(render_between_cycle){
            window.clear();
            render(window);
            window.draw(text);
            window.display();
        }
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
    float creature_ball_dist = (m_particles[m_creature.get_apex_tip_index()].get_curr_pos() - m_ball_pos).length();

    if(ball_displacement > 5) m_has_touched_ball = true;

    if(not m_has_touched_ball){
        reward -= 1;
        reward -= creature_ball_dist/200.f;
    }
    reward -= goal_ball_dist/300.f;
    reward -= 1; // normal time thing

    if(goal_ball_dist < m_goal_radius){
        reward += 1000;
        m_episode_end = 1;
    }

    return reward;
}

float Environment::get_curr_reward(){
    return m_reward;
}
