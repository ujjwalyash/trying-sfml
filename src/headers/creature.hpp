#pragma once
#include "particle.hpp"
#include "spring.hpp"
#include "muscle.hpp"
#include <vector>

class Neural_Net
{
    public:
        
        // takes current activations and changes them
        void forward(std::vector<float>& current_activations, std::vector<float>& observation);
};

class Creature{

    private:
        
        std::vector<int> m_muscle_index; // indicies in the springs array 
        std::vector<float> m_current_activations; // ordered according to muslces_index
        
        Neural_Net m_brain; // ordered according to muslces_index

    public:

        Creature(std::vector<int> muscle_index);
        void act(std::vector<Muscle>& muscles, std::vector<float>& observation);    

};

void create_creature_bacteriophage(int& num_particles, std::vector<Particle>& particles, int& num_springs, std::vector<Spring>& springs, int& num_muscles, std::vector<Muscle>& muscles, float dt);
Creature create_creature_muscle_sperm (int& num_particles, std::vector<Particle>& particles, int& num_springs, std::vector<Spring>& springs, int& num_muscles, std::vector<Muscle>& muscles, float dt);
void create_creature_motor_sperm  (Creature& creature, int& num_particles, std::vector<Particle>& particles, int& num_springs, std::vector<Spring>& springs, int& num_muscles, std::vector<Muscle>& muscles, float dt);
void create_creature_rope         (int& num_particles, std::vector<Particle>& particles, int& num_springs, std::vector<Spring>& springs, int& num_muscles, std::vector<Muscle>& muscles, float dt);