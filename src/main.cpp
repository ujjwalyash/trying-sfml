#include <SFML/Graphics/RenderWindow.hpp>
#include <vector>
#include "particle.hpp"
#include <iostream>

int main()
{	
	int num_particles = 50;
	int fps = 240;
	float dt = 1.f/fps ;

	std::vector<Particle> particles(num_particles);
	sf::Vector2<float> old_pos, vel, curr_pos, acc;
	for(int i = 0; i < num_particles; i++){
		old_pos.x = 20; old_pos.y = 20*(num_particles-i);
		vel.x = 30+(i==15)*20; vel.y = 0;
		curr_pos.x = vel.x*dt + old_pos.x; curr_pos.y = vel.y*dt + old_pos.y;
		acc.x = 40, acc.y = +60;
		particles[i].set_pos(old_pos, curr_pos);
		particles[i].set_acc(acc);
	}

	sf::RenderWindow window( sf::VideoMode( { 1920, 1080 } ), "SFML works!", sf::State::Fullscreen);
	window.setFramerateLimit(fps);
	// std::cout << window.getSize().x << " " << window.getSize().y;;
	

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
		}

		window.clear();
		// window.draw( shape );
		for(int i = 0; i < num_particles; i++){
			particles[i].step(dt);
		}
		handle_all_collisions(particles);
		for(int i = 0; i < num_particles; i++){
			// particles[i].handle_boundary_constraint();
			window.draw(particles[i].get_shape());
		}
		
		window.display();
	}
}
