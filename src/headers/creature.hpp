#pragma once
#include "particle.hpp"
#include "spring.hpp"
#include "muscle.hpp"
#include "Eigen/Core"
#include <SFML/System/Vector2.hpp>
#include <vector>

typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> MatrixXd;
// row vec
typedef Eigen::Matrix<float, 1, Eigen::Dynamic> VectorXd;

class Neural_Net
{       
    private:
        
        std::vector<int> m_layer_sizes;
        std::vector<MatrixXd> m_weights; // supposed to be keep on multiplied to the right 
        std::vector<VectorXd> m_biases; // supposed to be keep on multiplied to the right 

    public:

        Neural_Net(std::vector<int> layer_sizes);    
        Neural_Net(std::vector<MatrixXd>& layers, std::vector<VectorXd>& biases);    
        // takes current activations and changes them
        void forward(std::vector<float>& current_activations, std::vector<float>& observation);
};

class Creature{

    // make dervied classes taking this as base for each diff creature
    // creature_create_sperm func will be within
    // will have own custom get_obs
    private:
        
        std::vector<int> m_muscle_index; // indicies in the springs array 
        std::vector<float> m_current_activations; // ordered according to muslces_index
        
        Neural_Net m_brain; // ordered according to muslces_index

        // ONYL FOR TESTING MAKE IT PRIVATE AGAIN LATER
    public:
        std::vector<int> m_sensing_points;

    public:

        Creature(std::vector<int> muscle_index, std::vector<int> sensing_points, std::vector<int> layer_sizes);
        Creature(std::vector<int> muscle_index, std::vector<int> sensing_points, std::vector<MatrixXd>& layers, std::vector<VectorXd>& biases);
        void act(std::vector<Muscle>& muscles, std::vector<float>& observation);    

        // will be moved to the derived class
        void get_observation(std::vector<float>& obs, sf::Vector2f target_pos, const std::vector<Particle>& particles);

};

// class Sperm: public Creature
// {
//     public:

// }

Creature create_creature_muscle_sperm (int& num_particles, std::vector<Particle>& particles, int& num_springs, std::vector<Spring>& springs, int& num_muscles, std::vector<Muscle>& muscles, float dt);

void create_creature_bacteriophage(int& num_particles, std::vector<Particle>& particles, int& num_springs, std::vector<Spring>& springs, int& num_muscles, std::vector<Muscle>& muscles, float dt);
void create_creature_motor_sperm  (Creature& creature, int& num_particles, std::vector<Particle>& particles, int& num_springs, std::vector<Spring>& springs, int& num_muscles, std::vector<Muscle>& muscles, float dt);
void create_football              (int& num_particles, std::vector<Particle>& particles, int& num_springs, std::vector<Spring>& springs, float dt);