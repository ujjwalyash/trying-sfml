#include <SFML/Graphics/RenderWindow.hpp>
#include <SFML/Graphics/Font.hpp>
#include <SFML/Graphics/Text.hpp>
#include <SFML/System/Clock.hpp>
#include <cmath>
#include <cstdint>
#include <vector>
#include "particle.hpp"
#include "spring.hpp"
// #include <iostream>

int main()
{	
	int num_particles = 8;
	int fps = 60;
	int num_iterations = 8;
	float dt = 1.f/(fps*num_iterations);
	bool paused = true;
	bool step = false;

	std::vector<Particle> particles;
	sf::Vector2<float> old_pos, vel, curr_pos, acc;
	std::vector<float> x_coords{200, 200, 230, 230, 230, 230, 230, 230};
	std::vector<float> y_coords{200, 230, 230, 200, 160, 120, 80, 40};
	std::vector<float> mass{1,1,1,1,0.8,0.6,0.4,0.2};
	for(int i = 0; i < num_particles; i++){
		old_pos.x = x_coords[i]; old_pos.y = y_coords[i];
		// old_pos.x = 200; old_pos.y = 30*(num_particles-i);
		vel.x = 0.3*i; vel.y = 0.4*i;
		curr_pos.x = vel.x*dt + old_pos.x; curr_pos.y = vel.y*dt + old_pos.y;
		acc.x = +40, acc.y = +60;

		particles.push_back(Particle(i+1, 1, mass[i]));
		particles[i].set_pos(old_pos, curr_pos);
		particles[i].set_acc(acc);
	}

	int num_springs = 10;
	std::vector<Constraint> springs;
	springs.push_back(Constraint(particles[0], particles[1], 30.f, 1e2));
	springs.push_back(Constraint(particles[1], particles[2], 30.f, 1e2));
	springs.push_back(Constraint(particles[2], particles[3], 30.f, 1e2));
	springs.push_back(Constraint(particles[3], particles[0], 30.f, 1e2));
	springs.push_back(Constraint(particles[2], particles[0], 30.f * sqrt(2), 1e5));
	springs.push_back(Constraint(particles[1], particles[3], 30.f * sqrt(2), 1e5));
	springs.push_back(Constraint(particles[3], particles[4], 40.f, 1e2));
	springs.push_back(Constraint(particles[4], particles[5], 40.f, 1e2));
	springs.push_back(Constraint(particles[5], particles[6], 40.f, 1e2));
	springs.push_back(Constraint(particles[6], particles[7], 40.f, 1e2));

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
		clock.restart();

		for(int iter = 0; iter < num_iterations; iter++){
			handle_all_springs(springs, dt);
			for(int i = 0; i < num_particles; i++){
				particles[i].step(dt/2);
			}
			handle_all_collisions(particles);
		}

		clock.stop();
		time_taken = time_taken*0.9 + clock.getElapsedTime().asMicroseconds()/9.f;
		
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
					
		sf::Text text(font);
		text.setString(std::format("{:.2e}\n{}", total_energy, time_taken)); 
		text.setCharacterSize(24);            
		text.setFillColor(sf::Color::White); 
		text.setPosition({0.f, 0.f}); 
		// std::cout << "total_energy: " << total_energy << '\n';
		window.draw(text);

		window.display();
	}
}
