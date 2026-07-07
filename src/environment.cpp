#include "headers/environment.hpp"
#include <iostream>

Environment::Environment(sf::Vector2f ball_pos, sf::Vector2f goal_pos)
    // m_creature has no default constructor hence need to do this
    // ensure all function parameter are defined earlier in the CLASS DECLARATION
{
    // CREATE THE BALL BEFROE THE CREATURE
    // if you do afterwords and the vector moves to a bigger size
    // all the springs which hold references will be invalidated
    
    m_id = -1;
    m_original_ball_pos = ball_pos;
    m_goal_pos = goal_pos;
    
    // after creating football in sperm creation springs vector is resized, the springs break
    // TODO: temp fix for now -- change refs to index in springs
    m_particles.reserve(env_num_particles);
    m_springs.reserve(env_num_springs);
    m_muscles.reserve(env_num_muscles);
    
    m_goal_center_index = create_football(m_num_particles, m_particles, m_num_springs, m_springs, m_id);
    for(int i = 0; i < m_num_particles; i++){
        m_particles[i].shift_pos(m_original_ball_pos);
    }
    
    // create_high_drag_block(m_num_particles, m_particles, m_num_springs, m_springs, env_dt);
    // create_self_aligning_arrow(m_num_particles, m_particles, m_num_springs, m_springs, env_dt);
    // create_collision_test_rig(m_num_particles, m_particles, m_num_springs, m_springs, env_dt);

    Creature_data data = create_creature_muscle_sperm(m_num_particles, m_particles, m_num_springs, m_springs, m_num_muscles, m_muscles, m_id);
    
    assert(m_num_particles == env_num_particles);
    assert(m_num_springs == env_num_springs);
    assert(m_num_muscles == env_num_muscles);
 
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
        
    m_id = id;
    m_original_ball_pos = ball_pos;
    m_goal_pos = goal_pos;
    
    // after creating football in sperm creation springs vector is resized, the springs break
    // temp fix for now
    m_particles.reserve(env_num_particles);
    m_springs.reserve(env_num_springs);
    m_muscles.reserve(env_num_muscles);
    
    m_goal_center_index = create_football(m_num_particles, m_particles, m_num_springs, m_springs, m_id);
    for(int i = 0; i < m_num_particles; i++){
        m_particles[i].shift_pos(m_original_ball_pos);
    }
    Creature_data data = create_creature_muscle_sperm(m_num_particles, m_particles, m_num_springs, m_springs, m_num_muscles, m_muscles, m_id);
    
    assert(m_num_particles == env_num_particles);
    assert(m_num_springs == env_num_springs);
    assert(m_num_muscles == env_num_muscles);
    
    // 20 -- 12 -- 8
    std::vector<int> layer_sizes{m_num_muscles+env_observation_size, 12, m_num_muscles};

    // loading json
    std::string file_path = std::format("./saved/{}.json", id);
    std::ifstream i(file_path);
    if(!i){
        std::cout << "could not load from file: " << file_path << " --> doing random initialization" << '\n';
        m_creature = Creature{data.s_muscle_indices, data.s_sensing_points, layer_sizes};
    }
    else{
        std::cout << "loaded from file: " << file_path << '\n';
        
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
    }

    m_goal_radius = 10;
}

Creature const& Environment::get_creature() const{
    return m_creature;
}

void Environment::copy_brain(Environment const& env){
    m_creature.copy_brain(env.get_creature());
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
    // m_reward = 0;
    m_episode_end = false;
    m_has_touched_ball = false;

    check();
}

// void Environment::run_episode(){
//     while(not m_episode_end){
//         step();
//     }
// }

void Environment::check(){
    for(int i = 0; i < m_num_particles; i++){
        m_particles[i].check();
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
void Environment::step_stage_0(){	
    
    // stage 0
	std::vector<float> observation(env_observation_size);
    m_creature.get_observation(observation, m_ball_pos, m_goal_pos, m_particles);
    m_creature.act(m_muscles, observation);
    
    // update muscle lengths
    handle_all_muscle_contraction(m_muscles);
    // stage 0

    check();
}

// for(int cycle = 0; cycle < env_num_frames_per_creature_action; cycle++){

//     for(int iter = 0; iter < env_num_iterations_per_frame; iter++){
        // launch_first_half_step();
        
void Environment::step_stage_1(){	
            // stage 1
            for(int i = 0; i < m_num_particles; i++){
                m_particles[i].first_half_step();
            }			
            check();
            
            // update acc
            //todo: move this to gpu
            handle_all_muscle_forces(m_muscles);
            // check();
            //todo: move this to gpu
            handle_all_springs(m_springs);
            check();
            
            handle_all_collisions(m_particles, m_springs, m_muscles);    
            
            for(int i = 0; i < m_num_particles; i++){
                m_particles[i].second_half_step();
            }		
            // stage 1
            	
}
            // launch_second_half_step();

void Environment::step_stage_2(){	
    // stage 2
    m_ball_pos = m_particles[m_goal_center_index].get_curr_pos();

    float reward = calculate_reward();
    m_reward += reward;

    m_num_steps_done++;

    if(m_num_steps_done > env_max_steps_per_episode)
        m_episode_end = true;
    // stage 2

    check();
}

// keeping this cpu only
void Environment::step(sf::RenderWindow& window, sf::Text& text, bool render_between_cycle){
    
	std::vector<float> observation(env_observation_size);
    m_creature.get_observation(observation, m_ball_pos, m_goal_pos, m_particles);
    m_creature.act(m_muscles, observation);

    // update muscle lengths
    handle_all_muscle_contraction(m_muscles);
    
    for(int cycle = 0; cycle < env_num_frames_per_creature_action; cycle++){

        for(int iter = 0; iter < env_num_iterations_per_frame; iter++){

            for(int i = 0; i < m_num_particles; i++){
                m_particles[i].first_half_step();
            }			
            
            // update acc
            handle_all_muscle_forces(m_muscles);
            handle_all_springs(m_springs);
            handle_all_collisions(m_particles, m_springs, m_muscles);    
            
            for(int i = 0; i < m_num_particles; i++){
                m_particles[i].second_half_step();
            }	
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

    if(m_num_steps_done > env_max_steps_per_episode)
        m_episode_end = true;
}

float Environment::calculate_reward(){
    
    float reward = 0;

    float goal_ball_dist = (m_ball_pos-m_goal_pos).length();
    // float ball_displacement = (m_original_ball_pos-m_ball_pos).length();
    float creature_ball_dist = (m_particles[m_creature.get_apex_tip_index()].get_curr_pos() - m_ball_pos).length();
    // float apex_tip_speed = m_particles[m_creature.get_apex_tip_index()].get_vel().length();

    // rn balls sinks :|
    // if(ball_displacement > 5) m_has_touched_ball = true;

    // if(not m_has_touched_ball){
        // reward -= 1;
        // reward -= creature_ball_dist/100.f;
    // }
    
    reward -= creature_ball_dist/100.f;    
    reward -= goal_ball_dist/300.f;
    reward -= 1; // normal time thing

    // std::cout << "tip speed" << apex_tip_speed << '\n';
    // std::cout.flush();

    // if present from the beginning this leads to suboptimal policy of just vibrating the tip in place
    // reward += apex_tip_speed/20;

    if(goal_ball_dist < m_goal_radius){
        reward += 1000;
        m_episode_end = 1;
    }

    return reward;
}

void Environment::reset_reward(){
    m_reward = 0;
}

float Environment::get_curr_reward(){
    return m_reward;
}
