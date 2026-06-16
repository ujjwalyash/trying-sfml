// #include <SFML/Graphics/CircleShape.hpp>
// #include <SFML/Graphics/Color.hpp>
// #include <SFML/Graphics/RenderWindow.hpp>
// #include <SFML/Graphics/Font.hpp>
// #include <SFML/Graphics/Text.hpp>
// #include <SFML/System/Clock.hpp>
// #include <SFML/System/Vector2.hpp>
// #include <SFML/Window/Keyboard.hpp>
// #include "headers/particle.hpp"
// #include "headers/spring.hpp"
// #include "headers/creature.hpp"
// #include "headers/muscle.hpp"
// // #include <iostream>

// int main()
// {	
// 	// simulation params
// 	int fps = 60;
// 	int num_iterations = 16;
// 	float dt = 1.f/(fps*num_iterations);
// 	bool paused = true;
// 	bool step = false;

// 	int num_particles = 0; 
// 	std::vector<Particle> particles;
// 	int num_springs = 0;
// 	std::vector<Spring> springs;
// 	int num_muscles = 0;
// 	// you cant keep copy of same spring in two arrays if you modify
// 	// one you would need to modify the other too
// 	// so muscles will not be present in springs hence 
// 	// we need to WRITE THE SAME THING TWICE for Muscles too
// 	std::vector<Muscle> muscles;

// 	// after creating football in sperm creation springs vector is resized, the springs break
// 	// temp fix for now
// 	particles.reserve(200);
// 	springs.reserve(300);
// 	muscles.reserve(10);

// 	create_football(num_particles, particles, num_springs, springs, dt);
// 	Creature creature = create_creature_muscle_sperm(num_particles, particles, num_springs, springs, num_muscles, muscles, dt);
// 	// Creature creature({}, {}, {});

// 	std::vector<float> observation(4);
// 	sf::CircleShape target_shape(20);
// 	target_shape.setFillColor(sf::Color::Red);
// 	target_shape.setOrigin({10, 10});
// 	sf::Vector2f target_pos;

// 	sf::RenderWindow window( sf::VideoMode( { 1920, 1080 } ), "SFML works!", sf::State::Fullscreen);
// 	window.setFramerateLimit(fps);

// 	sf::Font font("/usr/share/fonts/adwaita-sans-fonts/AdwaitaSans-Regular.ttf");
// 	sf::Clock clock;
// 	uint32_t time_taken = 0;
// 	int num_frames_done = 0;

// 	while ( window.isOpen() )
// 	{
// 		while ( const std::optional event = window.pollEvent() )
// 		{
// 			if ( event->is<sf::Event::Closed>() )
// 				window.close();

// 			else if (const auto* resized = event->getIf<sf::Event::Resized>()) {
// 				// Update the view to match the new window size dimensions
// 				sf::FloatRect visibleArea(sf::Vector2f{0.f, 0.f}, sf::Vector2f{(float)resized->size.x, (float)resized->size.y});
// 				window.setView(sf::View(visibleArea));
//     		}

// 			else if (const auto* keyPressed = event->getIf<sf::Event::KeyPressed>()){
// 				if (keyPressed->scancode == sf::Keyboard::Scan::Escape)
// 					paused = not paused;
				
// 				else if (keyPressed->scancode == sf::Keyboard::Scan::Enter)
// 					step = true;

// 				else if (keyPressed->scancode == sf::Keyboard::Scan::R){
// 					// sf::Vector2f shift = {(float)(rand()%1920), (float)(rand()%1080)}; 
// 					// float rot = (float)(rand()%360)/180 * 3.141f;
// 					for(int i = 0; i < num_particles; i++){
// 						particles[i].reset();
// 					}
// 					for(int i = 0; i < num_muscles; i++){
// 						muscles[i].reset();
// 					}
// 				}
// 			}

// 			else if (sf::Mouse::isButtonPressed(sf::Mouse::Button::Left)){
// 				sf::Vector2i mouse_pos = sf::Mouse::getPosition(window);
// 				target_pos.x = (float)mouse_pos.x;
// 				target_pos.y = (float)mouse_pos.y;

// 				target_shape.setPosition(target_pos);
// 			}
// 		}

// 		if(paused && (!step)) 
// 			continue;
// 		step = false;

// 		float total_energy = 0;
		
// 		window.clear();

// 		window.draw(target_shape);

// 		int j = 0;
// 		sf::CircleShape sh;
// 		for(int i = 0; i < num_particles; i++){
// 			total_energy += particles[i].calculate_total_energy(dt);
// 			if(j < (int)creature.m_sensing_points.size() && i == creature.m_sensing_points[j]){
// 				j++;
// 				sf::Vector2f particle_pos = particles[i].get_curr_pos();
// 				float r = particles[i].get_radius();
// 				sh.setPosition({particle_pos.x-r, particle_pos.y-r});
// 				sh.setRadius(r);
// 				sh.setFillColor(sf::Color::Green);
// 				window.draw(sh);
// 			}
// 			else{
// 				sf::Vector2f particle_pos = particles[i].get_curr_pos();
// 				float r = particles[i].get_radius();
// 				sh.setPosition({particle_pos.x-r, particle_pos.y-r});
// 				sh.setRadius(r);
// 				sh.setFillColor(sf::Color::White);
// 				window.draw(sh);
// 			}
// 		}
// 		for(int i = 0; i < num_springs; i++){
// 			std::array line = springs[i].get_line();
// 			total_energy += springs[i].calculate_total_energy();
// 			window.draw(line.data(), line.size(), sf::PrimitiveType::Lines);
// 		}
// 		for(int i = 0; i < num_muscles; i++){
// 			std::array line = muscles[i].get_line();
// 			total_energy += muscles[i].calculate_total_energy();
// 			window.draw(line.data(), line.size(), sf::PrimitiveType::Lines);
// 		}
					
// 		sf::Text text(font);
// 		text.setString(std::format("{:.2e}\n{}", total_energy, time_taken)); 
// 		text.setCharacterSize(24);            
// 		text.setFillColor(sf::Color::White); 
// 		text.setPosition({0.f, 0.f}); 
// 		window.draw(text);

// 		window.display();
		
// 		// physics calc start /////////////
// 		clock.restart();
		
// 		if(num_frames_done == (int)fps/10){
// 			// creature.get_observation(observation, target_pos, particles);
// 			creature.act(muscles, observation);

// 			num_frames_done = 0;
// 		}
// 		num_frames_done++;

// 		for(int iter = 0; iter < num_iterations; iter++){
// 			handle_all_muscles(muscles, dt);
// 			handle_all_springs(springs, dt);
// 			for(int i = 0; i < num_particles; i++){
// 				particles[i].step(dt);
// 			}			
// 			handle_all_collisions(particles);
// 		}
		
// 		// physics calc end ///////////////
// 		clock.stop();
// 		time_taken = time_taken*0.9 + clock.getElapsedTime().asMicroseconds()/9.f;
// 	}
// }
