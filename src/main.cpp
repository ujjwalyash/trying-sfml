#include <SFML/Graphics/RenderWindow.hpp>
#include <vector>
#include "particle.hpp"
#include "constraint.hpp"
// #include <iostream>

int main()
{	
	int num_particles = 2;
	int fps = 240;
	float dt = 1.f/fps;
	bool paused = true;
	bool step = false;

	std::vector<Particle> particles;
	sf::Vector2<float> old_pos, vel, curr_pos, acc;
	for(int i = 0; i < num_particles; i++){
		old_pos.x = 200; old_pos.y = 110*(num_particles-i);
		vel.x = 100*i; vel.y = 0;
		curr_pos.x = vel.x*dt + old_pos.x; curr_pos.y = vel.y*dt + old_pos.y;
		acc.x = 60, acc.y = +100;

		particles.push_back(Particle(i+1));
		particles[i].set_pos(old_pos, curr_pos);
		particles[i].set_acc(acc);
	}

	std::vector<Constraint> contraints;
	Constraint c1(particles[0], particles[1], 100.f);
	contraints.push_back(c1);

	sf::RenderWindow window( sf::VideoMode( { 1920, 1080 } ), "SFML works!", sf::State::Fullscreen);
	window.setFramerateLimit(fps);

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

		// float total_energy = 0;

		window.clear();
		// window.draw( shape );
		for(int i = 0; i < num_particles; i++){
			particles[i].step(dt);
		}
		handle_all_collisions(particles);
		handle_all_constraints(contraints);
		for(int i = 0; i < num_particles; i++){
			// particles[i].handle_boundary_constraint();
			// total_energy += particles[i].calculate_total_energy(dt);
			window.draw(particles[i].get_shape());
		}
		
		// std::cout << "total_energy: " << total_energy << '\n';

		window.display();
	}
}
