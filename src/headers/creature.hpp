#pragma once
#include "particle.hpp"
#include "spring.hpp"
#include "muscle.hpp"
#include <Eigen/Core>
#include <EigenRand/EigenRand>
#include <json.hpp>

// row vec
typedef Eigen::Matrix<float, 1, Eigen::Dynamic> VectorXdf;
typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> MatrixXdf;

using json = nlohmann::json;
namespace Eigen{
    void to_json(json& j, MatrixXdf const& matrix);
    void from_json(const nlohmann::json& j, MatrixXdf& matrix);
    
    // remove this duplication
    void to_json(json& j, VectorXdf const& vector);
    void from_json(const nlohmann::json& j, VectorXdf& vector);
}

class Neural_Net
{       
    private:
        
        float m_mutation_std = 0.3;

        std::vector<int> m_layer_sizes;
        std::vector<MatrixXdf> m_weights; // supposed to be keep on multiplied to the right 
        std::vector<VectorXdf> m_biases; // supposed to be keep on multiplied to the right 

    public:

        Neural_Net(std::vector<int> layer_sizes);    
        Neural_Net(std::vector<MatrixXdf>& weights, std::vector<VectorXdf>& biases);    
        // takes current activations and changes them
        void forward(std::vector<float>& current_activations, std::vector<float>& observation);

        std::vector<MatrixXdf> const& get_weights() const;
        std::vector<VectorXdf> const& get_biases() const;
        void crossover(Neural_Net const& par_1, Neural_Net const& par_2);
        void mutate(float mutation_rate);

        void save(int id, float reward);
};

class Creature{

    // make dervied classes taking this as base for each diff creature
    // creature_create_sperm func will be within
    // will have own custom get_obs
    private:
        
        std::vector<int> m_muscle_index; // indicies in the springs array 
        std::vector<float> m_current_activations; // ordered according to muslces_index
        std::vector<int> m_sensing_points;
        
        Neural_Net m_brain; // ordered according to muslces_index

    public:

        // idk if i should allow this but it here to make environment constructor better
        Creature();

        Creature(std::vector<int> muscle_index, std::vector<int> sensing_points, std::vector<int> layer_sizes);
        Creature(std::vector<int> muscle_index, std::vector<int> sensing_points, std::vector<MatrixXdf>& weights, std::vector<VectorXdf>& biases);
        void act(std::vector<Muscle>& muscles, std::vector<float>& observation);    

        // will be moved to the derived class
        // passing the entire particles array not a good practice(maybe)
        void get_observation(std::vector<float>& obs, sf::Vector2f ball_pos, sf::Vector2f goal_pos, const std::vector<Particle>& particles);
        int get_apex_tip_index();

        Neural_Net const& get_brain() const;
        void crossover(Creature const& par_1, Creature const& par_2);
        void mutate(float mutation_rate);

        void save(int id, float reward);
        void copy_brain(Creature const& creature);

};

int create_football              (int& num_particles, std::vector<Particle>& particles, int& num_springs, std::vector<Spring>& springs, float dt);
// return type for creation functions below
struct Creature_data{
    std::vector<int> s_muscle_indices;
    std::vector<int> s_sensing_points;
};

Creature_data create_creature_muscle_sperm (int& num_particles, std::vector<Particle>& particles, int& num_springs, std::vector<Spring>& springs, int& num_muscles, std::vector<Muscle>& muscles, float dt);

void create_creature_bacteriophage(int& num_particles, std::vector<Particle>& particles, int& num_springs, std::vector<Spring>& springs, int& num_muscles, std::vector<Muscle>& muscles, float dt);
void create_creature_motor_sperm  (Creature& creature, int& num_particles, std::vector<Particle>& particles, int& num_springs, std::vector<Spring>& springs, int& num_muscles, std::vector<Muscle>& muscles, float dt);