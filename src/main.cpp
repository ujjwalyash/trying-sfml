#include "headers/environment.hpp"
#include <SFML/Graphics/Font.hpp>
#include <SFML/System/Clock.hpp>
#include <SFML/Window/Keyboard.hpp>
#include <iostream>
#include <pthread.h>
#include <cassert>
#include <unistd.h>

// to ensure no other file can access these -- ie global vars restricted to this file
namespace{
	
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

	// cpu - gpu coordination
	// which stage of step are ALL env in right now
	// tells the workers what to call
	int stage = 0;
};

void * worker(void *){
	
	// std::cout << "New worker started working \n";
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

		//! this is to minimize false sharing by giving as large as possible chunks to each thread
		num_envs_started += env_per_worker;
		// num_envs_started++;
		
		// relese lock
		pthread_mutex_unlock(&lk_num_ens_started);
		
		//// useless but doesnt hurt will remove later
		// assert(curr_job != -1);

		// now do the job
		for(int job_num = 0; job_num < env_per_worker; job_num++){
			
			for(int ep_no = 0; ep_no < num_episode_per_generation; ep_no++){
				switch (stage){
					case -1:
						env[curr_job + job_num].reset(ball_pos[ep_no], goal_pos[ep_no]);
						break;
					case 0:
						env[curr_job + job_num].step_stage_0();
						break;
					case 1:
						env[curr_job + job_num].step_stage_1();
						break;
					case 2:
						env[curr_job + job_num].step_stage_2();
						break;
					default:
						std::cerr << "unknown stage: " << stage << '\n';
						exit(-1);
				}
				// TODO: do not reset muscles/praticles just move the target -- ball and creature stays as is 
				// TODO: but in the early stage creature never touches ball so fine rn
			}
		}
		// increment the done counter
		pthread_mutex_lock(&lk_num_ens_done);
		num_envs_done += env_per_worker;
		// num_envs_done++;
		if(num_envs_done == population_size){
			// everything done wake up main thread to continue
			pthread_cond_broadcast(&main_thread_cond_var);
		}
		pthread_mutex_unlock(&lk_num_ens_done);
	}
	
	return NULL;
}

void create_workers(){
	// initialize mutex
	pthread_mutex_init(&lk_num_ens_started, NULL);
	pthread_mutex_init(&lk_num_ens_done, NULL);

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
}

// starts workers -- returns only after ALL work is finished
void launch_threads(){
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
}

inline sf::Vector2f get_random_ball_pos(){
	// sf::Vector2f disp_from_center = {(float)(rand()%(int)(max_x - 120) - 100), (float)(rand()%(int)(max_y/2.f - 60) - 100)};
	float dist = (float)(rand()%(int)(50)) + 100;
	// float dist = (float)(rand()%(int)(max_y-60)) + 200;
	float ang = (float)(rand()%360);
	sf::Vector2f disp_from_center = {dist, 0};
	disp_from_center = disp_from_center.rotatedBy(sf::degrees(ang));

	return {disp_from_center.x + max_x/2.f, disp_from_center.y + max_y/2.f};
}
inline sf::Vector2f get_random_goal_pos(){
	return {(float)(rand()%(int)max_x), (float)(rand()%(int)max_y)};
}

void* render_current_gen(void *){

	bool paused = true;
	bool step = false;
	bool render_btw_cycle = true;
	int speed_up = 2;

	int env_rank = 0;

	// COPY
	int curr_gen = gen;
	Environment env_used_for_render(-1, get_random_ball_pos(), get_random_goal_pos());
	env_used_for_render.copy_brain(env[rankings[env_rank]]);
	// Environment& env_used_for_render = env[67];
	
	// YOU cANT copy env -- bc springs have refs not indexes so after copy new env still points to old env's particles
	// env_used_for_render.emplace(env[rankings[env_rank]]);

	sf::RenderWindow window( sf::VideoMode( { 1920, 1080 } ), "SFML works!", sf::State::Fullscreen);
    int fps = (env_fps/env_num_frames_per_creature_action);
	if (render_btw_cycle) window.setFramerateLimit(env_fps * speed_up);
	else window.setFramerateLimit(fps * speed_up);

	sf::Font font("/usr/share/fonts/adwaita-sans-fonts/AdwaitaSans-Regular.ttf");
	int num_steps_done = 0;
    float reward = 0;

	// for(int i = 0; i < num_episode_per_generation; i++){
	// 	ball_pos[i] = get_random_ball_pos();
	// 	goal_pos[i] = get_random_goal_pos();
	// }

	// stage = -1;
	// launch_threads();
	
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
					
					env_used_for_render.reset(get_random_ball_pos(), get_random_goal_pos());
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

		if(paused && (!step)){
			sleep(1);
			continue;
		}
		step = false;
		
		window.clear();
        
        env_used_for_render.render(window);
        // env_used_for_render.render_gpu(window);
		
        sf::Text text(font);
		text.setString(std::format("reward:    {:.1f}\nstps_done: {}\nspd_up:    {}\ncurr_gen:  {}\ncurr_rank: {}", 
                                            reward, num_steps_done, speed_up, curr_gen, env_rank+1)); 
		text.setCharacterSize(24);            
		text.setFillColor(sf::Color::White); 
		text.setPosition({0.f, 0.f}); 
		window.draw(text);

		window.display();
		
		
		env_used_for_render.step(window, text, render_btw_cycle);
		

		// launch_big_kernel();
		
		// stage = 0;
		// launch_threads();

		// stage = 1;
		// launch_threads();
		
		// stage = 2;
		// launch_threads();

		// env_used_for_render.mutate(mutation_rate);
		
		// cudaSynchronize();

        reward = env_used_for_render.get_curr_reward();
        // reward = env_used_for_render.get_curr_reward() - gpu_mem.env_reward()[67];
        num_steps_done++;
	}

	// wont reach here 
	// only to remove warning
	return NULL;
}

int main()
{	
	// BADD ALL ENVIRONMENTS WILL END UP WITH SAME INTIAL VALUES of neural netwrok DUE TO THIS
    // !std::vector<Environment> env(population_size, Environment(first_ball_pos, first_goal_pos));    
	// here the env object is created ONLY ONCE and copied to every entry in the vector

	// now ok. this also prevents copy of the big envi objects when resized
	env.reserve(population_size);
	allocate_cuda_memory();

	for (int i = 0; i < (int)population_size; ++i) {
		env.emplace_back(i, get_random_ball_pos(), get_random_goal_pos());
		// else env.emplace_back(get_random_ball_pos(), get_random_goal_pos());
	}

    std::vector<float> rewards(population_size, 0);    

	for(int i = 0; i < population_size; i++) rankings.push_back(i);

	// only works for lambdas without capture 
	// bool (*comp)(int a, int b) = [&env](int a, int b)->bool{
	auto comp = [&rewards](int a, int b)->bool{
			return rewards[a] > rewards[b];
	};

	// only creates then, they don't start working until launch_thread called
	create_workers();
	
	// start the rendering thread
	pthread_t render_thread;
	pthread_create(&render_thread, NULL, render_current_gen, NULL);

	sf::Clock clock;
	for(gen = 0; gen < num_generations; gen++){

		clock.restart();
		// generate sequence of ball and goal positions
		for(int i = 0; i < num_episode_per_generation; i++){
			ball_pos[i] = get_random_ball_pos();
			goal_pos[i] = get_random_goal_pos();
		}

		// reset rewards
		// for(int i = 0; i < population_size; i++){
		// 	env[i].reset_reward();
		// 	// std::cout << rewards[rankings[i]] << " \n"[i==population_size-1];
		// 	rewards[i] = 0;
		// }

		// resets env
		stage = -1;
		launch_threads();

		// TODO: think about false sharing among threads working on close by data
		
		launch_big_kernel();
		cudaSynchronize();
		
		// copy rewards before sorting -- probably faster but so negligible compared to the sims 
		for(int i = 0; i < population_size; i++){
			rewards[i] = env[i].get_curr_reward_gpu();
			// rewards[i] = env[i].get_curr_reward();
			if(std::isnan(rewards[i])){
				rewards[i] = std::numeric_limits<float>::lowest();
			}
		}

		// sort them
		std::sort(rankings.begin(), rankings.end(), comp);
		
		// remove bottom ones, repalce by cross_overs
		for(int i = population_size-1; i >= (1-elimination_percentage)*population_size; i--){
			int par_1 = rankings[rand()%(int)(top_unchanged_percentage*population_size)];
			int par_2 = rankings[rand()%(int)(top_unchanged_percentage*population_size)];
			// int par_1 = rankings[rand()%(int)((1-elimination_percentage)*population_size)];
			// int par_2 = rankings[rand()%(int)((1-elimination_percentage)*population_size)];

			env[rankings[i]].crossover(env[par_1], env[par_2]);
		}

		// mutate the middle ones
		for(int i = (1-elimination_percentage)*population_size-1; i >= top_unchanged_percentage*population_size; i--){
			env[rankings[i]].mutate(mutation_rate);
		}

		clock.stop();
		if(gen % 64 == 0){
			std::cout << "current generation: " << gen << '\n';
			std::cout << "current best_reward: " << rewards[rankings[0]] << '\n';
			std::cout << "time taken: " << clock.getElapsedTime().asSeconds() << "s" << '\n';
			std::cout << '\n';
			std::cout.flush();
		}
	}


	if(num_generations != 0){
		for(int i = 0; i < population_size; i++){
			env[rankings[i]].save(i, rewards[rankings[i]]);
		}
		std::cout << "saved all envs \n";
		std::cout.flush();
	}

	free_cuda_memory();
	pthread_join(render_thread, NULL);
}
