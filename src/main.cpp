#include <SFML/Graphics/RenderWindow.hpp>
#include <SFML/Graphics/Font.hpp>
#include <SFML/Graphics/Text.hpp>
#include "particle.hpp"
#include "constraint.hpp"
#include <iostream>

int main()
{	
	int num_particles = 2;
	int fps = 60;
	float dt = 1.f/fps;
	bool paused = true;
	bool step = false;

	std::vector<Particle> particles;
	sf::Vector2<float> old_pos, vel, curr_pos, acc;
	for(int i = 0; i < num_particles; i++){
		old_pos.x = 200; old_pos.y = 100*(num_particles-i);
		vel.x = 200*(1-2*i); vel.y = 0;
		curr_pos.x = vel.x*dt + old_pos.x; curr_pos.y = vel.y*dt + old_pos.y;
		acc.x = 0, acc.y = +0;

		particles.push_back(Particle(i+1));
		particles[i].set_pos(old_pos, curr_pos);
		particles[i].set_acc(acc);
	}

	int num_constraints = 1;
	std::vector<Constraint> contraints;
	Constraint c1(particles[0], particles[1], 100.f);
	contraints.push_back(c1);

	sf::RenderWindow window( sf::VideoMode( { 1920, 1080 } ), "SFML works!", sf::State::Fullscreen);
	window.setFramerateLimit(fps);

	sf::Font font("/usr/share/fonts/adwaita-sans-fonts/AdwaitaSans-Regular.ttf");

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
		// window.draw( shape );
		for(int i = 0; i < num_particles; i++){
			particles[i].step(dt);
		}
		handle_all_collisions(particles);
		handle_all_constraints(contraints);
		
		for(int i = 0; i < num_particles; i++){
			// particles[i].handle_boundary_constraint();
			total_energy += particles[i].calculate_total_energy(dt);
			window.draw(particles[i].get_shape());
		}
		for(int i = 0; i < num_constraints; i++){
			std::array line = contraints[i].get_line();
		    window.draw(line.data(), line.size(), sf::PrimitiveType::Lines);
		}
		

			
		sf::Text text(font);
		text.setString(std::format("{:.2e}", total_energy)); 
		text.setCharacterSize(24);            
		text.setFillColor(sf::Color::White); 
		text.setPosition({0.f, 0.f}); 
		// std::cout << "total_energy: " << total_energy << '\n';
		window.draw(text);

		window.display();
	}
}
