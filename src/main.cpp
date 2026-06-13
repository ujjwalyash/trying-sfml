#include <SFML/Graphics/RenderWindow.hpp>
#include <SFML/Graphics/Font.hpp>
#include <SFML/Graphics/Text.hpp>
#include <SFML/System/Clock.hpp>
#include <vector>
#include "headers/particle.hpp"
#include "headers/spring.hpp"
#include "headers/creature.hpp"
#include "headers/muscle.hpp"
// #include <iostream>

int main()
{	
	// simulation params
	int fps = 60;
	int num_iterations = 16;
	float dt = 1.f/(fps*num_iterations);
	bool paused = true;
	bool step = false;

	int num_particles = 0; 
	std::vector<Particle> particles;
	int num_springs = 0;
	std::vector<Spring> springs;
	int num_muscles = 0;
	// you cant keep copy of same spring in two arrays if you modify
	// one you would need to modify the other too
	// so muscles will not be present in springs hence 
	// we need to WRITE THE SAME THING TWICE for Muscles too
	std::vector<Muscle> muscles;
	Creature creature = create_creature_muscle_sperm(num_particles, particles, num_springs, springs, num_muscles, muscles, dt);

	std::vector<float> observation;

	sf::RenderWindow window( sf::VideoMode( { 1920, 1080 } ), "SFML works!", sf::State::Fullscreen);
	window.setFramerateLimit(fps);

	sf::Font font("/usr/share/fonts/adwaita-sans-fonts/AdwaitaSans-Regular.ttf");
	sf::Clock clock;
	uint32_t time_taken = 0;

	while ( window.isOpen() )
	{
		while ( const std::optional event = window.pollEvent() )
		{
			if ( event->is<sf::Event::Closed>() )
				window.close();

			if (const auto* resized = event->getIf<sf::Event::Resized>()) {
				// Update the view to match the new window size dimensions
				sf::FloatRect visibleArea(sf::Vector2f{0.f, 0.f}, sf::Vector2f{(float)resized->size.x, (float)resized->size.y});
				window.setView(sf::View(visibleArea));
    		}

			if (const auto* keyPressed = event->getIf<sf::Event::KeyPressed>()){
				if (keyPressed->scancode == sf::Keyboard::Scan::Escape)
					paused = not paused;
			}

			if (const auto* keyPressed = event->getIf<sf::Event::KeyPressed>()){
				if (keyPressed->scancode == sf::Keyboard::Scan::Enter)
					step = true;
			}
		}

		if(paused && (!step)) 
			continue;
		step = false;

		float total_energy = 0;
		
		window.clear();
		for(int i = 0; i < num_particles; i++){
			// particles[i].handle_boundary_spring();
			total_energy += particles[i].calculate_total_energy(dt);
			window.draw(particles[i].get_shape());
		}
		for(int i = 0; i < num_springs; i++){
			std::array line = springs[i].get_line();
			total_energy += springs[i].calculate_total_energy();
			window.draw(line.data(), line.size(), sf::PrimitiveType::Lines);
		}
		for(int i = 0; i < num_muscles; i++){
			std::array line = muscles[i].get_line();
			total_energy += muscles[i].calculate_total_energy();
			window.draw(line.data(), line.size(), sf::PrimitiveType::Lines);
		}
					
		sf::Text text(font);
		text.setString(std::format("{:.2e}\n{}", total_energy, time_taken)); 
		text.setCharacterSize(24);            
		text.setFillColor(sf::Color::White); 
		text.setPosition({0.f, 0.f}); 
		// std::cout << "total_energy: " << total_energy << '\n';
		window.draw(text);

		window.display();
		
		// physics calc start /////////////
		clock.restart();
		
		for(int iter = 0; iter < num_iterations; iter++){
			creature.act(muscles, observation);
			handle_all_muscles(muscles, dt);
			handle_all_springs(springs, dt);
			for(int i = 0; i < num_particles; i++){
				particles[i].step(dt);
			}			
			handle_all_collisions(particles);
		}
		
		// physics calc end ///////////////
		clock.stop();
		time_taken = time_taken*0.9 + clock.getElapsedTime().asMicroseconds()/9.f;
	}
}
