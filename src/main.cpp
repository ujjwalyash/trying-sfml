#include "headers/environment.hpp"
#include <SFML/Graphics/Font.hpp>
#include <SFML/System/Clock.hpp>
#include <SFML/Window/Keyboard.hpp>
#include <iostream>
#include <pthread.h>

// to ensure no other file can access these -- ie global vars restricted to this file
namespace{
	std::vector<Environment> env;
	std::vector<int> rankings;
	int gen;
	int population_size = 20;
};

void* render_current_gen(void *){

	bool paused = true;
	bool step = false;
	bool render = true;
	int speed_up = 1;

	int env_rank = 0;

	// COPY
	int curr_gen = gen;
	std::optional<Environment> env_used_for_render;
	env_used_for_render.emplace(env[rankings[env_rank]]);

	sf::RenderWindow window( sf::VideoMode( { 1920, 1080 } ), "SFML works!", sf::State::Fullscreen);
    int fps = (env_fps/env_num_frames_per_creature_action);
	if (render) window.setFramerateLimit(env_fps * speed_up);
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
					env_used_for_render.emplace(env[rankings[env_rank]]);

                    sf::Vector2f ball_pos = {(float)(rand()%1920), (float)(rand()%1080)};
                    sf::Vector2f goal_pos = {(float)(rand()%1920), (float)(rand()%1080)};
					
					env_used_for_render->reset(ball_pos, goal_pos);

                    num_steps_done = 0;
				}

				else if (keyPressed->scancode == sf::Keyboard::Scan::Q){

					window.close();
					return NULL;
				}

				else if (keyPressed->scancode == sf::Keyboard::Scan::Up){
                    speed_up++;
					if (render) window.setFramerateLimit(env_fps * speed_up);
					else window.setFramerateLimit(fps * speed_up);
                }
				else if (keyPressed->scancode == sf::Keyboard::Scan::Down){
					if(speed_up > 1) speed_up--;
					if (render) window.setFramerateLimit(env_fps * speed_up);
					else window.setFramerateLimit(fps * speed_up);
                }

				else if (keyPressed->scancode == sf::Keyboard::Scan::Right){
					if(env_rank < population_size-1){
						env_rank++;	
						// update
						curr_gen = gen;
						env_used_for_render.emplace(env[rankings[env_rank]]);
					}
                }
				else if (keyPressed->scancode == sf::Keyboard::Scan::Left){
					if(env_rank > 0){
						env_rank--;	
						// update
						curr_gen = gen;
						env_used_for_render.emplace(env[rankings[env_rank]]);
					}
                }
				else if (keyPressed->scancode == sf::Keyboard::Scan::End){
					if(env_rank < population_size-10){
						env_rank += 10;	
						// update
						curr_gen = gen;
						env_used_for_render.emplace(env[rankings[env_rank]]);
					}
                }
				else if (keyPressed->scancode == sf::Keyboard::Scan::Home){
					if(env_rank > 9){
						env_rank -= 10;	
						// update
						curr_gen = gen;
						env_used_for_render.emplace(env[rankings[env_rank]]);
					}
                }
            }
		}

		if(paused && (!step)) 
			continue;
		step = false;
		
		window.clear();
        
        env_used_for_render->render(window);
		
        sf::Text text(font);
		text.setString(std::format("{:.1f}\n{}\n{}\n{}", 
                                            reward, num_steps_done, speed_up, curr_gen)); 
		text.setCharacterSize(24);            
		text.setFillColor(sf::Color::White); 
		text.setPosition({0.f, 0.f}); 
		window.draw(text);

		window.display();
		
		env_used_for_render->step(window, text, render);
        reward = env_used_for_render->get_curr_reward();
        num_steps_done++;
	}

	// wont reach here 
	// only to remove warning
	return NULL;
}

int main()
{	
	int num_generations = 10;
	int num_episode_per_generation = 2;

	float top_unchanged_percentage = 0.3;
	float elimination_percentage = 0.4;

	float mutation_rate = 0.5;
	bool load_old_gen = true;

    sf::Vector2f first_goal_pos = {(float)(rand()%1920), (float)(rand()%1080)};
    sf::Vector2f first_ball_pos = {(float)(rand()%1920), (float)(rand()%1080)};
	
	// BADD ALL ENVIRONMENTS WILL END UP WITH SAME INTIAL VALUES of neural netwrok DUE TO THIS
    // !std::vector<Environment> env(population_size, Environment(first_ball_pos, first_goal_pos));    
	// here the env object is created ONLY ONCE and copied to every entry in the vector

	// now ok
	env.reserve(population_size); // Prevents performance-heavy reallocations

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
	
    std::vector<sf::Vector2f> goal_pos(num_episode_per_generation);
    std::vector<sf::Vector2f> ball_pos(num_episode_per_generation);

	for(int i = 0; i < population_size; i++) rankings.push_back(i);

	// only works for lambdas without capture 
	// bool (*comp)(int a, int b) = [&env](int a, int b)->bool{
	auto comp = [&rewards](int a, int b)->bool{
			return rewards[a] > rewards[b];
	};

	// start the rendering thread
	pthread_t render_thread;
	pthread_create(&render_thread, NULL, render_current_gen, NULL);
	
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
			rewards[i] = 0;
		}
		
		// evaluate all creatures
		for(int i = 0; i < population_size; i++){
			for(int ep_no = 0; ep_no < num_episode_per_generation; ep_no++){
				env[i].run_episode();
				rewards[i] += env[i].get_curr_reward();
				env[i].reset(ball_pos[ep_no], goal_pos[ep_no]);
			}
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
	}

	for(int i = 0; i < population_size; i++){
		env[rankings[i]].save(i, rewards[rankings[i]]);
	}
}
