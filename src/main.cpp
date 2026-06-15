#include "headers/environment.hpp"
#include <SFML/Graphics/Font.hpp>
#include <SFML/Graphics/Text.hpp>
#include <SFML/System/Clock.hpp>
#include <SFML/System/Vector2.hpp>
#include <SFML/Window/Keyboard.hpp>

int main()
{	
    sf::Vector2f goal_pos = {(float)(rand()%1920), (float)(rand()%1080)};
    sf::Vector2f ball_pos = {(float)(rand()%1920), (float)(rand()%1080)};
    Environment env(ball_pos, goal_pos);    
    bool paused = true;
	bool step = false;

    int speed_up = 1;
	sf::RenderWindow window( sf::VideoMode( { 1920, 1080 } ), "SFML works!", sf::State::Fullscreen);
    int fps = (env.m_fps/env.m_num_frames_per_creature_action);
	window.setFramerateLimit(fps * speed_up);

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
                	window.setFramerateLimit(fps * speed_up);
                }
				else if (keyPressed->scancode == sf::Keyboard::Scan::Down){
                    if(speed_up > 1) speed_up--;
                	window.setFramerateLimit(fps * speed_up);
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
		
		env.step();
        reward = env.get_curr_reward();
        num_steps_done++;
		
		// physics calc end ///////////////
		clock.stop();
		time_taken = time_taken*0.9 + clock.getElapsedTime().asMicroseconds()/9.f;
	}
}
