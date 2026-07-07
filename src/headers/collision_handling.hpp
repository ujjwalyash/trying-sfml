#include "muscle.hpp"

void handle_particle_particle_collision(Particle& p1, Particle& p2);
void handle_particle_spring_collision(Particle& p1, Particle& p2);
void handle_all_collisions(std::vector<Particle>& particles, std::vector<Spring>& springs, std::vector<Muscle>& muscles);