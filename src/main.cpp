#include "headers/environment.hpp"
#include <SFML/System/Clock.hpp>
#include <SFML/Window/Keyboard.hpp>
#include <pthread.h>

/*
void render_current_best(){
	bool paused = true;
	bool step = false;
	bool render = true;

	int speed_up = 1;
	sf::RenderWindow window( sf::VideoMode( { 1920, 1080 } ), "SFML works!", sf::State::Fullscreen);
    int fps = (env[0].m_fps/env[0].m_num_frames_per_creature_action);
	if (render) window.setFramerateLimit(env[0].m_fps * speed_up);
	else window.setFramerateLimit(fps * speed_up);

	sf::Font font("/usr/share/fonts/adwaita-sans-fonts/AdwaitaSans-Regular.ttf");
	sf::Clock clock;
	uint32_t time_taken = 0;
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
                    ball_pos = {(float)(rand()%1920), (float)(rand()%1080)};
                    goal_pos = {(float)(rand()%1920), (float)(rand()%1080)};
					env.reset(ball_pos, goal_pos);

                    num_steps_done = 0;
				}

				else if (keyPressed->scancode == sf::Keyboard::Scan::Up){
                    speed_up++;
                	if (render) window.setFramerateLimit(env.m_fps * speed_up);
					else window.setFramerateLimit(fps * speed_up);
                }
				else if (keyPressed->scancode == sf::Keyboard::Scan::Down){
                    if(speed_up > 1) speed_up--;
                	if (render) window.setFramerateLimit(env.m_fps * speed_up);
					else window.setFramerateLimit(fps * speed_up);
                }
            }
		}

		if(paused && (!step)) 
			continue;
		step = false;
		
		window.clear();
        
        env.render(window);
		
        sf::Text text(font);
		text.setString(std::format("{:.1f}\n{}\n{}\n{}", 
                                            reward, time_taken, num_steps_done, speed_up)); 
		text.setCharacterSize(24);            
		text.setFillColor(sf::Color::White); 
		text.setPosition({0.f, 0.f}); 
		window.draw(text);

		window.display();
		
		// physics calc start /////////////
		clock.restart();
		
		env.step(window, text, render);
        reward = env.get_curr_reward();
        num_steps_done++;
		
		// physics calc end ///////////////
		clock.stop();
		time_taken = time_taken*0.9 + clock.getElapsedTime().asMilliseconds()/9.f;
	}
}
*/

int main()
{	
	int population_size = 100;
	int num_generations = 100;
	int num_episode_per_generation = 2;

	float top_unchanged_percentage = 0.3;
	float elimination_percentage = 0.4;

	float mutation_rate = 0.5;

    sf::Vector2f first_goal_pos = {(float)(rand()%1920), (float)(rand()%1080)};
    sf::Vector2f first_ball_pos = {(float)(rand()%1920), (float)(rand()%1080)};
    std::vector<Environment> env(population_size, Environment(first_ball_pos, first_goal_pos));    
    std::vector<float> rewards(population_size, 0);    
	
    std::vector<sf::Vector2f> goal_pos(num_episode_per_generation);
    std::vector<sf::Vector2f> ball_pos(num_episode_per_generation);

	std::vector<int> rankings;
	for(int i = 0; i < population_size; i++) rankings.push_back(i);

	// only works for lambdas without capture 
	// bool (*comp)(int a, int b) = [&env](int a, int b)->bool{
	auto comp = [&rewards](int a, int b)->bool{
			return rewards[a] > rewards[b];
	};
	
	for(int gen = 0; gen < num_generations; gen++){

		// generate sequence of ball and goal positions
		for(int i = 0; i < num_episode_per_generation; i++){
			goal_pos[i] = {(float)(rand()%1920), (float)(rand()%1080)};
			ball_pos[i] = {(float)(rand()%1920), (float)(rand()%1080)};
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
			int par_1 = rankings[rand()%((int)(1-elimination_percentage)*population_size)];
			int par_2 = rankings[rand()%((int)(1-elimination_percentage)*population_size)];

			// TODO: check correctness 
			env[i].crossover(env[par_1], env[par_2]);
		}

		for(int i = (1-elimination_percentage)*population_size-1; i >= top_unchanged_percentage*population_size; i--){
			env[i].mutate(mutation_rate);
		}
		
		// reset rewards
		for(int i = 0; i < population_size; i++){
			rewards[i] = 0;
		}
	}

	for(int i = 0; i < population_size; i++){
		env[i].save(i);
	}
}
