#include "headers/environment.hpp"
#include <SFML/Graphics/Font.hpp>
#include <SFML/System/Clock.hpp>
#include <SFML/Window/Keyboard.hpp>
#include <iostream>
#include <pthread.h>
#include <cassert>

// physics params
const float max_y = 1080;
const float max_x = 1920;

// since i keep both balanced out anyways just make both 0
const sf::Vector2f gravity = {0, 0};
const float buoyancy_const = 4.f/3 * 3.141 * 0; // DO NOT MAKE DENSITY 1000

const float restitution = 0.8;
const float coefficient_friction = 1;

// if two particles in a line move along that line then viscous force on back particle must be smaller but here its same as the front one
// this completely destroys the concept of streamlined bodies 
// TODO: make viscous force more accurate ie springs(lines) face it too -- no need to do this for collisions though wont be too hard
const float viscosity = 0.02 * 20; // *10 bc viscosity is only applied at point masses which have small radius so we scale it

// env params
const int env_fps = 60;
const int env_num_frames_per_creature_action = (float)env_fps/10; // every 100ms 
const int env_num_iterations_per_frame = 16;
const float env_dt = 1.f/(env_fps*env_num_iterations_per_frame);
const int env_max_steps_per_episode = 500;
const int env_observation_size = 12;

// to ensure no other file can access these -- ie global vars restricted to this file
namespace{
	
	// 18 threads for 20 core cpu too much
	const int num_workers = 16;

	// simulation params
	const int population_size = 112;
	static_assert((population_size%num_workers == 0), "population_size is not a multiple of num_workers");

	const int num_episode_per_generation = 1;
	const int num_generations = 300;

	const float top_unchanged_percentage = 0.3;
	const float elimination_percentage = 0.4;
	
	const float mutation_rate = 0.6;
	
	const bool load_old_gen = true;
	
	// other vars
	int gen;
	std::vector<Environment> env;
	std::vector<int> rankings;
	std::vector<sf::Vector2f> goal_pos(num_episode_per_generation);
    std::vector<sf::Vector2f> ball_pos(num_episode_per_generation);

	// for thread pool
	int num_envs_started = 0;
	pthread_mutex_t lk_num_ens_started;
	
	int num_envs_done = 0;
	pthread_mutex_t lk_num_ens_done;

	// to wake up every worker to start working
	pthread_cond_t worker_cond_var;
	// to wake up main thread when everything's done
	pthread_cond_t main_thread_cond_var;
};

void * worker(void *){
	
	std::cout << "New worker started working \n";
	// work your whole life
	while(1){

		// the index of the env the worker is supoosed to run right now
		int curr_job = -1;
		
		// look for new job
		pthread_mutex_lock(&lk_num_ens_started);
		
		// if everything is done release lock and sleep -- woken up by main_thread with brodacast during next generation
		while(num_envs_started >= population_size){
			pthread_cond_wait(&worker_cond_var, &lk_num_ens_started);
		}
		
		// commit to the job
		curr_job = num_envs_started;
		num_envs_started++;
		
		// relese lock
		pthread_mutex_unlock(&lk_num_ens_started);
		
		// useless but doesnt hurt will remove later
		assert(curr_job != -1);

		// now do the job
		for(int ep_no = 0; ep_no < num_episode_per_generation; ep_no++){
			env[curr_job].run_episode();
			// TODO: do not reset muscles/praticles just move the target -- ball and creature stays as is 
			// TODO: but in the early stage creature never touches ball so fine rn
			env[curr_job].reset(ball_pos[ep_no], goal_pos[ep_no]);
		}

		// increment the done counter
		pthread_mutex_lock(&lk_num_ens_done);
		num_envs_done++;
		if(num_envs_done == population_size){
			// everything done wake up main thread to continue
			pthread_cond_broadcast(&main_thread_cond_var);
		}
		pthread_mutex_unlock(&lk_num_ens_done);
	}
	
	return NULL;
}

void* render_current_gen(void *){

	bool paused = true;
	bool step = false;
	bool render_btw_cycle = true;
	int speed_up = 2;

	int env_rank = 0;

	// COPY
	int curr_gen = gen;
	sf::Vector2f render_ball_pos = {(float)(rand()%1920), (float)(rand()%1080)};
	sf::Vector2f render_goal_pos = {(float)(rand()%1920), (float)(rand()%1080)};
	Environment env_used_for_render(render_ball_pos, render_goal_pos);
	env_used_for_render.copy_brain(env[rankings[env_rank]]);
	
	// YOU cANT copy env -- bc springs have refs not indexes so after copy new env still points to old env's particles
	// env_used_for_render.emplace(env[rankings[env_rank]]);

	sf::RenderWindow window( sf::VideoMode( { 1920, 1080 } ), "SFML works!", sf::State::Fullscreen);
    int fps = (env_fps/env_num_frames_per_creature_action);
	if (render_btw_cycle) window.setFramerateLimit(env_fps * speed_up);
	else window.setFramerateLimit(fps * speed_up);

	sf::Font font("/usr/share/fonts/adwaita-sans-fonts/AdwaitaSans-Regular.ttf");
	int num_steps_done = 0;
    float reward = 0;
	
	while ( window.isOpen() )
	{
		while ( const std::optional event = window.pollEvent() )
		{
			if ( event->is<sf::Event::Closed>() )		
				window.close();

			else if (const auto* resized = event->getIf<sf::Event::Resized>()) {
				// Update the view to match the new window size dimensions
				sf::FloatRect visibleArea(sf::Vector2f{0.f, 0.f}, sf::Vector2f{(float)resized->size.x, (float)resized->size.y});
				window.setView(sf::View(visibleArea));
    		}

			else if (const auto* keyPressed = event->getIf<sf::Event::KeyPressed>()){
				if (keyPressed->scancode == sf::Keyboard::Scan::Escape)
					paused = not paused;
				
				else if (keyPressed->scancode == sf::Keyboard::Scan::Enter)
					step = true;

				else if (keyPressed->scancode == sf::Keyboard::Scan::R){

					// update
					curr_gen = gen;
					env_used_for_render.copy_brain(env[rankings[env_rank]]);
					// env_used_for_render.emplace(env[rankings[env_rank]]);
					
                    render_ball_pos = {(float)(rand()%1920), (float)(rand()%1080)};
                    render_goal_pos = {(float)(rand()%1920), (float)(rand()%1080)};
					
					env_used_for_render.reset(render_ball_pos, render_goal_pos);
					env_used_for_render.reset_reward();

                    num_steps_done = 0;
				}

				else if (keyPressed->scancode == sf::Keyboard::Scan::Q){

					window.close();
					return NULL;
				}

				else if (keyPressed->scancode == sf::Keyboard::Scan::Up){
                    speed_up++;
					if (render_btw_cycle) window.setFramerateLimit(env_fps * speed_up);
					else window.setFramerateLimit(fps * speed_up);
                }
				else if (keyPressed->scancode == sf::Keyboard::Scan::Down){
					if(speed_up > 1) speed_up--;
					if (render_btw_cycle) window.setFramerateLimit(env_fps * speed_up);
					else window.setFramerateLimit(fps * speed_up);
                }

				else if (keyPressed->scancode == sf::Keyboard::Scan::Right){
					if(env_rank < population_size-1){
						env_rank++;	
						// update
						curr_gen = gen;
						env_used_for_render.copy_brain(env[rankings[env_rank]]);
						env_used_for_render.reset_reward();
					}
                }
				else if (keyPressed->scancode == sf::Keyboard::Scan::Left){
					if(env_rank > 0){
						env_rank--;	
						// update
						curr_gen = gen;
						env_used_for_render.copy_brain(env[rankings[env_rank]]);
						env_used_for_render.reset_reward();
					}
                }
				else if (keyPressed->scancode == sf::Keyboard::Scan::End){
					if(env_rank < population_size-10){
						env_rank += 10;	
						// update
						curr_gen = gen;
						env_used_for_render.copy_brain(env[rankings[env_rank]]);
						env_used_for_render.reset_reward();
					}
                }
				else if (keyPressed->scancode == sf::Keyboard::Scan::Home){
					if(env_rank > 9){
						env_rank -= 10;	
						// update
						curr_gen = gen;
						env_used_for_render.copy_brain(env[rankings[env_rank]]);
						env_used_for_render.reset_reward();
					}
                }
            }
		}

		if(paused && (!step)) 
			continue;
		step = false;
		
		window.clear();
        
        env_used_for_render.render(window);
		
        sf::Text text(font);
		text.setString(std::format("reward:    {:.1f}\nstps_done: {}\nspd_up:    {}\ncurr_gen:  {}\ncurr_rank: {}", 
                                            reward, num_steps_done, speed_up, curr_gen, env_rank+1)); 
		text.setCharacterSize(24);            
		text.setFillColor(sf::Color::White); 
		text.setPosition({0.f, 0.f}); 
		window.draw(text);

		window.display();
		
		env_used_for_render.step(window, text, render_btw_cycle);
        reward = env_used_for_render.get_curr_reward();
        num_steps_done++;
	}

	// wont reach here 
	// only to remove warning
	return NULL;
}

int main()
{	

    sf::Vector2f first_goal_pos = {(float)(rand()%1920), (float)(rand()%1080)};
    sf::Vector2f first_ball_pos = {(float)(rand()%1920), (float)(rand()%1080)};
	
	// BADD ALL ENVIRONMENTS WILL END UP WITH SAME INTIAL VALUES of neural netwrok DUE TO THIS
    // !std::vector<Environment> env(population_size, Environment(first_ball_pos, first_goal_pos));    
	// here the env object is created ONLY ONCE and copied to every entry in the vector

	// now ok. this also prevents copy of the big envi objects when resized
	env.reserve(population_size);

	for (int i = 0; i < (int)population_size; ++i) {
		// if a saved with id does not exist the constructor
		if(load_old_gen) env.emplace_back(i, first_ball_pos, first_goal_pos);
		else env.emplace_back(first_ball_pos, first_goal_pos);
	}

	// save-load testing
	// for(int i = 0; i < population_size; i++){
	// 	env[i].save(i, 0);
	// }
	
	// for(int i = 0; i < population_size; i++){
	// 	Environment temp(i, first_ball_pos, first_goal_pos);
	// 	assert(temp.get_creature().get_brain().get_weights() == env[i].get_creature().get_brain().get_weights());
	// 	assert(temp.get_creature().get_brain().get_biases() == env[i].get_creature().get_brain().get_biases());
	// 	assert(temp.get_curr_reward() == env[i].get_curr_reward());
	// }
	// exit(0);

    std::vector<float> rewards(population_size, 0);    

	for(int i = 0; i < population_size; i++) rankings.push_back(i);

	// only works for lambdas without capture 
	// bool (*comp)(int a, int b) = [&env](int a, int b)->bool{
	auto comp = [&rewards](int a, int b)->bool{
			return rewards[a] > rewards[b];
	};

	// start the rendering thread
	pthread_t render_thread;
	pthread_create(&render_thread, NULL, render_current_gen, NULL);
	
	// initialize mutex
	pthread_mutex_init(&lk_num_ens_started, NULL);
	pthread_mutex_init(&lk_num_ens_done, NULL);

	// create workers
	
	// to ensure no worker starts working right away -- all go to sleep
	pthread_mutex_lock(&lk_num_ens_started);
	num_envs_started = population_size;
	pthread_mutex_unlock(&lk_num_ens_started);

	pthread_mutex_lock(&lk_num_ens_done);
	num_envs_done = population_size;
	pthread_mutex_unlock(&lk_num_ens_done);
	for(int i = 0; i < num_workers; i++){
		// bad i should pobably keep track of the pthread_t objects
		pthread_t worker_thread;
		pthread_create(&worker_thread, NULL, worker, NULL);		
	}

	sf::Clock clock;
	for(gen = 0; gen < num_generations; gen++){

		clock.restart();
		// generate sequence of ball and goal positions
		for(int i = 0; i < num_episode_per_generation; i++){
			goal_pos[i] = {(float)(rand()%1920), (float)(rand()%1080)};
			ball_pos[i] = {(float)(rand()%1920), (float)(rand()%1080)};
		}

		// reset rewards
		for(int i = 0; i < population_size; i++){
			env[i].reset_reward();
			rewards[i] = 0;
		}
		
		// evaluate all creatures
		// reset job count
		pthread_mutex_lock(&lk_num_ens_started);
		num_envs_started = 0;
		pthread_mutex_unlock(&lk_num_ens_started);

		pthread_mutex_lock(&lk_num_ens_done);
		num_envs_done = 0;
		pthread_mutex_unlock(&lk_num_ens_done);

		// wake up all workers
		pthread_cond_broadcast(&worker_cond_var);
		
		// barrier -- wait for all threads to finish working
		pthread_mutex_lock(&lk_num_ens_done);
		// if everything is not done yet sleep -- will be woken up the last working thread
		while(num_envs_done != population_size){
			pthread_cond_wait(&main_thread_cond_var, &lk_num_ens_done);
		}
		pthread_mutex_unlock(&lk_num_ens_done);
		
		// copy rewards before sorting -- probably faster but so negligible compared to the sims 
		for(int i = 0; i < population_size; i++){
			rewards[i] = env[i].get_curr_reward();
		}

		// sort them
		std::sort(rankings.begin(), rankings.end(), comp);
		
		// remove bottom ones, repalce by cross_overs
		for(int i = population_size-1; i >= (1-elimination_percentage)*population_size; i--){
			int par_1 = rankings[rand()%(int)((1-elimination_percentage)*population_size)];
			int par_2 = rankings[rand()%(int)((1-elimination_percentage)*population_size)];

			env[rankings[i]].crossover(env[par_1], env[par_2]);
		}

		// mutate the middle ones
		for(int i = (1-elimination_percentage)*population_size-1; i >= top_unchanged_percentage*population_size; i--){
			env[rankings[i]].mutate(mutation_rate);
		}

		clock.stop();
		std::cout << "current generation: " << gen << '\n';
		std::cout << "current best_reward: " << rewards[rankings[0]] << '\n';
		std::cout << "time taken: " << clock.getElapsedTime().asSeconds() << "s" << '\n';
		std::cout << '\n';
		std::cout.flush();
	}

	for(int i = 0; i < population_size; i++){
		env[rankings[i]].save(i, rewards[rankings[i]]);
	}

	pthread_join(render_thread, NULL);
}
