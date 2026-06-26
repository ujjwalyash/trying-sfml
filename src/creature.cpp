#include "headers/creature.hpp"
#include <iostream>

Eigen::Rand::P8_mt19937_64 urng(42);
// Eigen::Rand::P8_mt19937_64 urng(rand()%UINT64_MAX);

Neural_Net::Neural_Net(std::vector<int> layer_sizes)
    :m_layer_sizes(layer_sizes.begin(), layer_sizes.end())
{

    int num_matrics = layer_sizes.size()-1;
    for(int i = 0; i < num_matrics; i++){
        int in_sz = layer_sizes[i];
        int out_sz = layer_sizes[i+1];

        m_weights.push_back(Eigen::Rand::balanced<MatrixXdf>(in_sz, out_sz, urng));
        m_biases.push_back(Eigen::Rand::balanced<MatrixXdf>(1, out_sz, urng));

    }
}

Neural_Net::Neural_Net(std::vector<MatrixXdf>& weights, std::vector<VectorXdf>& biases)
    :m_weights(weights.begin(), weights.end()),
     m_biases(biases.begin(), biases.end())
{
    for(MatrixXdf m: m_weights){
        m_layer_sizes.push_back(m.rows());
    }
    m_layer_sizes.push_back(m_weights.back().cols());
}

std::vector<MatrixXdf> const& Neural_Net::get_weights() const{
    return m_weights;
}
std::vector<VectorXdf> const& Neural_Net::get_biases() const{
    return m_biases;
}

namespace Eigen{

    void to_json(json& j, MatrixXdf const& matrix){
        j = json{
            {"rows", matrix.rows()},
            {"cols", matrix.cols()},
            {"data", std::vector<float>(matrix.data(), matrix.data() + matrix.size())}
        };
    }

    void from_json(const nlohmann::json& j, MatrixXdf& matrix) {
        int rows = j.at("rows");
        int cols = j.at("cols");
        std::vector<double> data = j.at("data");
        
        matrix.resize(rows, cols);
        std::copy(data.begin(), data.end(), matrix.data());
    }
    
    void to_json(json& j, VectorXdf const& vector){
        j = json{
            {"rows", vector.rows()},
            {"cols", vector.cols()},
            {"data", std::vector<float>(vector.data(), vector.data() + vector.size())}
        };
    }

    void from_json(const nlohmann::json& j, VectorXdf& vector) {
        int rows = j.at("rows");
        int cols = j.at("cols");
        std::vector<double> data = j.at("data");
        
        vector.resize(rows, cols);
        std::copy(data.begin(), data.end(), vector.data());
    }
}

void Neural_Net::save(int id, float reward){
    json data;
    data["id"] = id;
    data["reward"] = reward;
    data["layer_sizes"] = m_layer_sizes;
    data["weights"] = m_weights;
    data["biases"] = m_biases;

    std::string file_path = std::format("./saved/{}.json", id);
    std::ofstream o(file_path);
    if (!o) {
        std::cout << "Error opening file: " << file_path << std::endl;
        return;
    }

    o << std::setw(4) << data << std::endl;
}

void Neural_Net::crossover(Neural_Net const& par_1, Neural_Net const& par_2){
    std::vector<MatrixXdf> const& par_1_w = par_1.get_weights();
    std::vector<VectorXdf> const& par_1_b = par_1.get_biases();
    
    std::vector<MatrixXdf> const& par_2_w = par_2.get_weights();
    std::vector<VectorXdf> const& par_2_b = par_2.get_biases();

    int final_out = m_layer_sizes.back();
    float break_off = float(rand()%(final_out+1))/final_out;

    // std::cout << break_off << "\n\n";
    // std::cout << par_1_w[0] << "\n\n";
    // std::cout << par_2_w[0] << "\n\n";
    // std::cout << m_weights[0] << "\n\n";

    for(int i = 0; i < (int)m_layer_sizes.size()-1; i++){
        // currently we would need to move columns since we mutiply weights to the right
        // change this to respect row major order of storage(but check how eigen stores before)
        int num_rows = m_layer_sizes[i];
        int num_cols = m_layer_sizes[i+1];

        int first_half = (int)(break_off * num_cols);
        if(first_half == 0) first_half++;
        if(first_half == num_cols) first_half--;

        m_weights[i].block(0, 0, num_rows, first_half)
                                        = par_1_w[i].block(0, 0, num_rows, first_half);

        m_biases[i].block(0, 0, 1, first_half) = par_1_b[i].block(0, 0, 1, first_half); 
        

        m_weights[i].block(0, first_half, num_rows, num_cols-first_half)
                                        = par_2_w[i].block(0, first_half, num_rows, num_cols-first_half);

        m_biases[i].block(0, first_half, 1, num_cols-first_half) = par_2_b[i].block(0, first_half, 1, num_cols-first_half); 
        // for(int j = 0; j < num_cols; j++){
        //     if(j < (int)(break_off * num_cols)){
        //         // take col from p1
        //         m_weights[i].block(0, j, num_rows, 1)
        //             = par_1_w[i].block(0, j, num_rows, 1);

        //         m_biases[i](j) = par_1_b[i](j); 
        //     }
        //     else{
        //         // take col from p2
        //         m_weights[i].block(0, j, num_rows, 1)
        //             = par_2_w[i].block(0, j, num_rows, 1);

        //         m_biases[i](j) = par_2_b[i](j); 
        //     }
        // }
    }

    // std::cout << m_weights[0] << "\n\n";
    // std::cout.flush();

    // exit(0);
}

void Neural_Net::mutate(float mutation_rate){

    for(int i = 0; i < (int)m_layer_sizes.size()-1; i++){
        int in_sz = m_layer_sizes[i];
        int out_sz = m_layer_sizes[i+1];
        
        MatrixXdf weight_noise = m_mutation_std * Eigen::Rand::normal<MatrixXdf>(in_sz, out_sz, urng);
        MatrixXdf rand_w_mat    = Eigen::Rand::balanced<MatrixXdf>(in_sz, out_sz, urng);
        // rand w mat has [-1, 1] so we use 2*mutation_rate-1 instead of mutations rate
        MatrixXdf weight_mask = (rand_w_mat.array() < 2*mutation_rate-1).cast<float>();
        m_weights[i] += m_mutation_std * weight_noise.cwiseProduct(weight_mask);
        
        MatrixXdf bias_noise = m_mutation_std * Eigen::Rand::normal<MatrixXdf>(1, out_sz, urng);
        MatrixXdf rand_b_mat  = Eigen::Rand::balanced<MatrixXdf>(1, out_sz, urng);
        MatrixXdf bias_mask = (rand_b_mat.array() < 2*mutation_rate-1).cast<float>();

        m_biases[i] += m_mutation_std * bias_noise.cwiseProduct(bias_mask);
    }
}

void Neural_Net::forward(std::vector<float>& current_activations, std::vector<float>& observation){

    VectorXdf activation(m_layer_sizes[0]);
    assert(m_layer_sizes[0] == (int)(current_activations.size() + observation.size()));
    int ind = 0;
    for(float a: current_activations){
        activation(0, ind) = 2*a-1;
        ind++;
    }
    for(float o: observation){
        activation(0, ind) = o;
        ind++;
    }

    // std::cout << activation << '\n';

    for(int layer_no = 0; layer_no < (int)m_weights.size()-1; layer_no++){
        activation = activation * m_weights[layer_no] + m_biases[layer_no];
        activation = activation.cwiseMax(0);
    }
    activation = activation * m_weights[m_weights.size()-1];
    // approximate sigmoid as x/(1+|x|)
    activation = 0.5f*(1.f + (activation.array()/(1.f+activation.array().abs())));
    
    for(int i = 0; i < (int)current_activations.size(); i++){
        current_activations[i] = activation(0, i);
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

Creature::Creature()
    :m_brain({})
{}

Creature::Creature(std::vector<int> muscle_index, std::vector<int> sensing_points, std::vector<int> layer_sizes)
    :m_muscle_index(muscle_index.begin(), muscle_index.end()),
    m_current_activations(m_muscle_index.size(), 0),
    m_sensing_points(sensing_points.begin(), sensing_points.end()),
    m_brain(layer_sizes)
{}

Creature::Creature(std::vector<int> muscle_index, std::vector<int> sensing_points, std::vector<MatrixXdf>& weights, std::vector<VectorXdf>& biases)
    :m_muscle_index(muscle_index.begin(), muscle_index.end()),
    m_current_activations(m_muscle_index.size(), 0),
    m_sensing_points(sensing_points.begin(), sensing_points.end()),
    m_brain(weights, biases)
{}

Neural_Net const& Creature::get_brain() const{
    return m_brain;
}

void Creature::crossover(Creature const& par_1, Creature const& par_2){
    m_brain.crossover(par_1.get_brain(), par_2.get_brain());
}

void Creature::mutate(float mutation_rate){
    m_brain.mutate(mutation_rate);
}

void Creature::save(int id, float reward){
    m_brain.save(id, reward);
}

void Creature::copy_brain(Creature const& creature){
    // neural net has no const members and refs so safe to copy
    m_brain = creature.get_brain();
}

int Creature::get_apex_tip_index(){
    return m_sensing_points[0];
}

void Creature::act(std::vector<Muscle>& muscles, std::vector<float>& observation){
    m_brain.forward(m_current_activations, observation);
    for(int i=0; i < (int)m_muscle_index.size(); i++){
        muscles[i].set_activation(m_current_activations[i]);
    }
}

void Creature::get_observation(std::vector<float>& obs, sf::Vector2f ball_pos, sf::Vector2f goal_pos, const std::vector<Particle>& particles){
    
    // use a parameter somewhere instead of hardcoding 12
    assert(obs.size() == 12);
    std::vector<int> vals(12);
    // for ball
    float mn = 10000;
    float mx = 0;
    for(int i = 0; i < 4; i++){
        vals[i] = (ball_pos-particles[m_sensing_points[i]].get_curr_pos()).length();
        mn = fmin(mn, vals[i]);
        mx = fmax(mx, vals[i]);
    }

    float dem = mx-mn;
    for(int i = 0; i < 4; i++){
        obs[i] = 2.f*(vals[i]-mn)/dem - 1;
    }

    // for goal
    mn = 10000;
    mx = 0;
    for(int i = 4; i < 8; i++){
        vals[i] = (goal_pos-particles[m_sensing_points[i-4]].get_curr_pos()).length();
        mn = fmin(mn, vals[i]);
        mx = fmax(mx, vals[i]);
    }

    dem = mx-mn;
    for(int i = 4; i < 8; i++){
        obs[i] = 2.f*(vals[i]-mn)/dem - 1;
    }

    // for walls
    // points 0, 3
    obs[8] = 2.f*(particles[m_sensing_points[0]].get_curr_pos().x/max_x) - 1.f;
    obs[9] = 2.f*(particles[m_sensing_points[0]].get_curr_pos().y/max_y) - 1.f;
    
    obs[10] = 2.f*(particles[m_sensing_points[3]].get_curr_pos().x/max_x) - 1.f;
    obs[11] = 2.f*(particles[m_sensing_points[3]].get_curr_pos().y/max_y) - 1.f;
}

Creature_data create_creature_muscle_sperm(int& num_particles, std::vector<Particle>& particles, int& num_springs, std::vector<Spring>& springs
                                            , int& num_muscles, std::vector<Muscle>& muscles, float dt){

    sf::Vector2<float> old_pos, vel, curr_pos, acc;

    // --- CRITICAL PARAMETERS (PRESERVED UNCHANGED) ---
    int id_offset = num_particles;
    sf::Vector2f shift = {660, 390};

    // we cant really make things lighter unless we lower the spring constants
    const float b_const = 4.f / 3.f * 3.141f * 10.f; 
    const float head_heaviness = 1.f;

    // --- 1. DEFINE ORIGINAL CONTROL VERTICES ---
    sf::Vector2f corners[9];
    corners[0] = {300.0f, 150.0f}; // Head Center Core
    corners[1] = {300.0f, 124.0f}; // 1. Top Apex
    corners[2] = {307.5f, 137.0f}; // 2. Top-Right Mid-Wall
    corners[3] = {315.0f, 150.0f}; // 3. Right Lateral Apex
    corners[4] = {307.5f, 163.0f}; // 4. Bottom-Right Mid-Wall
    corners[5] = {300.0f, 176.0f}; // 5. Bottom Apex (Tail Base)
    corners[6] = {292.5f, 163.0f}; // 6. Bottom-Left Mid-Wall
    corners[7] = {285.0f, 150.0f}; // 7. Left Lateral Apex
    corners[8] = {292.5f, 137.0f}; // 8. Top-Left Mid-Wall

    // Original guide landmarks for interpolation path
    sf::Vector2f tail_landmarks[] = {
        corners[5],       
        {300.0f, 205.0f - 10.f}, 
        {300.0f, 235.0f - 10.f}, 
        {300.0f, 265.0f - 10.f}, 
        {300.0f, 295.0f - 10.f}, 
        {300.0f, 325.0f - 10.f}  
    };
    
    float tail_radii[] = {1.2f, 1.6f, 1.4f, 1.2f, 1.0f, 0.8f};

    std::vector<sf::Vector2f> positions;
    std::vector<float> mass;
    std::vector<float> radius;

    // --- 2. GENERATE DENSE HEAD SHELL ---
    positions.push_back(corners[0]);
    radius.push_back(1.5f);
    mass.push_back(head_heaviness * b_const * (1.5f * 1.5f * 1.5f)); 

    int apex_indices[9];
    const int HEAD_SUBDIVISIONS = 4; 

    for(int i = 1; i <= 8; i++) {
        apex_indices[i] = positions.size(); 
        int next_corner_idx = (i == 8) ? 1 : i + 1;
        
        sf::Vector2f pA = corners[i];
        sf::Vector2f pB = corners[next_corner_idx];

        for(int s = 0; s < HEAD_SUBDIVISIONS; s++) {
            float t = (float)s / (float)HEAD_SUBDIVISIONS;
            positions.push_back(pA + t * (pB - pA));
            radius.push_back(1.4f);
            mass.push_back(head_heaviness * b_const * (1.4f * 1.4f * 1.4f)); // Preserved 1.2f factor
        }
    }

    // --- 3. GENERATE SLIM ULTRA-THIN DOUBLE-STRAND TAIL ---
    std::vector<int> left_tail_indices;
    std::vector<int> right_tail_indices;
    // const int TAIL_SUBDIVISIONS = 4; 
    std::vector<int> tail_subdivs{6, 6, 6, 6, 8};

    for(int i = 0; i < 5; i++) {
        sf::Vector2f tA = tail_landmarks[i];
        sf::Vector2f tB = tail_landmarks[i + 1];
        float rA = tail_radii[i];  float rB = tail_radii[i + 1];

        const int TAIL_SUBDIVISIONS = tail_subdivs[i];
        for(int s = 1; s <= TAIL_SUBDIVISIONS; s++) {
            float t = (float)s / (float)TAIL_SUBDIVISIONS;
            sf::Vector2f center_pos = tA + t * (tB - tA);
            float current_radius = rA + t * (rB - rA);
            
            float half_width = current_radius * 1.5f; 

            float mass_scaling = 1.f;

            // Left Strand Node
            left_tail_indices.push_back(positions.size());
            positions.push_back(center_pos + sf::Vector2f(-half_width, 0.0f));
            radius.push_back(current_radius * 0.7); 
            mass.push_back(mass_scaling * b_const * (radius.back() * radius.back() * radius.back()));

            // Right Strand Node
            right_tail_indices.push_back(positions.size());
            positions.push_back(center_pos + sf::Vector2f(half_width, 0.0f));
            radius.push_back(current_radius * 0.7);
            mass.push_back(mass_scaling * b_const * (radius.back() * radius.back() * radius.back()));
        }
    }

    // --- 4. INSTANTIATE PARTICLES IN ENGINE ---
    std::vector<int> sensing_points;
    // THIS DOES NOT CHANGE RADIUS OF PARTICLES JUST THE DISTANCES
    for(int i = 0; i < (int)positions.size(); i++){
        old_pos = positions[i] + shift;
        vel.x = 0.01f * i; vel.y = 0.01f * i; 
        // curr_pos.x = vel.x * dt + old_pos.x; curr_pos.y = vel.y * dt + old_pos.y;
        acc = {0.0f, 0.0f};

        particles.push_back(Particle(i + 1 + id_offset, radius[i], mass[i], old_pos, vel, structure::creature));
        particles[particles.size()-1].set_acc(acc);

        if(positions[i] == corners[1] or positions[i] == corners[3]
                or positions[i] == corners[7] or i == (int)positions.size()-1){
            
            sensing_points.push_back(particles.size()-1);
        }
    }

    // --- 5. CONSTRUCT CONNECTION NETWORK (SPRINGS) ---
    // Preserved stiffness properties completely unchanged
    const float RIGID_TENDON   = 1e6f;  
    const float FLEXIBLE_SPINE = 1e4f;  
    const float ACTUATOR       = 1e3f;  

    auto add_spring = [&](int idxA, int idxB, float stiffness) {
        float dx = positions[idxA].x - positions[idxB].x;
        float dy = positions[idxA].y - positions[idxB].y;
        float exact_length = std::sqrt(dx * dx + dy * dy);
        assert(exact_length != 0);
        springs.push_back(Spring(particles[idxA+id_offset], particles[idxB+id_offset], exact_length, stiffness));
    };

    auto add_muscle = [&](int idxA, int idxB, float stiffness) {
        float dx = positions[idxA].x - positions[idxB].x;
        float dy = positions[idxA].y - positions[idxB].y;
        float exact_length = std::sqrt(dx * dx + dy * dy);
        assert(exact_length != 0);
        muscles.push_back(Muscle(particles[idxA+id_offset], particles[idxB+id_offset], exact_length, stiffness));
    };

    // A. SOLID HEAD SPOKES & OUTER RING
    for(int i = 1; i < left_tail_indices[0]; i++) {
        add_spring(0, i, RIGID_TENDON);
        int next_idx = (i == left_tail_indices[0] - 1) ? 1 : i + 1;
        add_spring(i, next_idx, RIGID_TENDON);
    }

    // B. ANCHOR TAIL ROOTS TO THE BASE OF THE HEAD
    int head_left_base = apex_indices[6];  
    int head_right_base = apex_indices[4]; 
    int head_left_bottom = apex_indices[6] - 2;  
    int head_right_bottom = apex_indices[4] + 2; 
    int bottom_tip = apex_indices[5]; 
    
    add_spring(head_left_base, left_tail_indices[0], RIGID_TENDON);
    add_spring(head_right_base, right_tail_indices[0], RIGID_TENDON);
    add_spring(head_left_bottom, left_tail_indices[0], RIGID_TENDON);
    add_spring(head_right_bottom, right_tail_indices[0], RIGID_TENDON);
    add_spring(bottom_tip, left_tail_indices[0], RIGID_TENDON);
    add_spring(bottom_tip, right_tail_indices[0], RIGID_TENDON);

    // add_spring(head_left_base,   left_tail_indices[1], RIGID_TENDON);
    // add_spring(head_right_base, right_tail_indices[1], RIGID_TENDON);
    // add_spring(head_bottom_tip,  left_tail_indices[1], RIGID_TENDON);
    // add_spring(head_bottom_tip, right_tail_indices[1], RIGID_TENDON);

    // C. INTERNAL AXONEME TRUSS & DISTRIBUTED COMPLIANT MUSCLE/TENDON SYSTEM
    int num_tail_segments = left_tail_indices.size();
    for(int i = 0; i < num_tail_segments; i++) {
        
        // 1. Horizontal cross-rungs (Rigidly holds the thin micro-ladder shape profile)
        add_spring(left_tail_indices[i], right_tail_indices[i], RIGID_TENDON);

        if(i < num_tail_segments - 1) {
            // 2. Shear diagonals (Prevents the thin strands from buckling or overlapping)
            add_spring(left_tail_indices[i], right_tail_indices[i + 1], RIGID_TENDON);
            add_spring(right_tail_indices[i], left_tail_indices[i + 1], RIGID_TENDON);

            // 3. Selective Actuation Strategy:
            // Instead of activating all 20 segments, we turn only 4 strategic segments into 
            // active muscle pairs (ACTUATOR). The remaining 16 segments are turned into 
            // springy, flexible tendons (FLEXIBLE_SPINE).
            // This leaves the GA with exactly 4 pairs of muscles to coordinate.
            // bool is_active_muscle = (i == 0 || i == 6 || i == 12 || i == 18);

            // std::vector<int> tail_subdivs{6, 6, 6, 6, 8};
            // *----**----**----**----**------*
            // 0123456789
            bool is_active_muscle = (i == 0 || i == 7 || i == 14 || i == 21);

            float scaling = (1-pow(float(i)/(num_tail_segments-1), 2))*100 + 1;

            if (is_active_muscle) {
                add_muscle(right_tail_indices[i], right_tail_indices[i + 1], ACTUATOR * scaling);
                add_muscle(left_tail_indices[i], left_tail_indices[i + 1], ACTUATOR * scaling);
            } else {
                add_spring(right_tail_indices[i], right_tail_indices[i + 1], FLEXIBLE_SPINE * scaling);
                add_spring(left_tail_indices[i], left_tail_indices[i + 1], FLEXIBLE_SPINE * scaling);
            }
        }
    }

    
    num_particles = particles.size();
    num_springs = springs.size();
    num_muscles = muscles.size();

    std::vector<int> muscle_indices;
    for(int i = 0; i < num_muscles; i++){
        muscle_indices.push_back(i);
    }

    Creature_data data{ muscle_indices, sensing_points };

    return data;
}

int create_football(int& num_particles, std::vector<Particle>& particles, int& num_springs, std::vector<Spring>& springs, float dt) {

    const float ball_float_sink_factor = 1.f; 

    // --- 2. DIMENSIONS & STRUCTURAL PARAMETERS ---
    const sf::Vector2f ball_center = {0.0f, 0.0f}; // Placed perfectly within reach of the sperm
    const float BALL_RADIUS = 20.0f;                  // Scaled size relative to the sperm head
    const int EDGE_POINTS = 50;                       // Evenly distributed points around the rim
    const float GAP_DISTANCE = 1.f;                  // Explicit tiny gap (in pixels) between edge circles
    const float CENTER_PARTICLE_RADIUS = 1.0f;

    // Calculate the precise chord distance between adjacent points along the circle
    float chord_length = 2.0f * BALL_RADIUS * std::sin(3.14159265f / (float)EDGE_POINTS);
    
    // Automatically derive the radius to guarantee they don't overlap and preserve the gap
    const float EDGE_PARTICLE_RADIUS = (chord_length - GAP_DISTANCE) / 2.0f;

    // Grab the global buoyancy constant defined in your engine
    const float b_const = 4.f / 3.f * 3.141f * 15.f;

    // Track starting index offset if particles already exist in the vectors
    int start_particle_idx = particles.size();

    // Temp vectors to calculate setup arrays before instantiation
    std::vector<sf::Vector2f> positions;
    std::vector<float> mass;
    std::vector<float> radius;

    // --- 3. GENERATE CENTER CORE NODE ---
    int center_index = 0;
    positions.push_back(ball_center);
    radius.push_back(CENTER_PARTICLE_RADIUS);
    float neutral_center_mass = b_const * (CENTER_PARTICLE_RADIUS * CENTER_PARTICLE_RADIUS * CENTER_PARTICLE_RADIUS);
    mass.push_back(neutral_center_mass * ball_float_sink_factor);

    // --- 4. GENERATE NON-OVERLAPPING RIM WITH EXPLICIT GAPS ---
    int center_node_index = start_particle_idx;
    int first_edge_index = start_particle_idx + 1;

    for (int i = 0; i < EDGE_POINTS; i++) {
        float angle = (i * 2.0f * 3.14159265f) / (float)EDGE_POINTS;
        sf::Vector2f offset = { std::cos(angle) * BALL_RADIUS, std::sin(angle) * BALL_RADIUS };
        
        positions.push_back(ball_center + offset);
        radius.push_back(EDGE_PARTICLE_RADIUS);

        float neutral_edge_mass = b_const * (EDGE_PARTICLE_RADIUS * EDGE_PARTICLE_RADIUS * EDGE_PARTICLE_RADIUS);
        mass.push_back(neutral_edge_mass * ball_float_sink_factor);
    }

    // --- 5. PUSH NEW PARTICLES INTO ENGINE ARRAY ---
    for (size_t i = 0; i < positions.size(); i++) {
        sf::Vector2f old_pos = positions[i];
        sf::Vector2f vel = { 0.001f * i, 0.001f * i }; 
        // sf::Vector2f curr_pos = vel * dt + old_pos;
        sf::Vector2f acc = { 0.0f, 0.0f };

        int global_id = start_particle_idx + i + 1; 
        particles.push_back(Particle(global_id, radius[i], mass[i], old_pos, vel, structure::ball));
        particles.back().set_acc(acc);
    }

    // --- 6. WEAVE THE SPRING CONSTANT NETWORK ---
    const float BALL_SKIN_STIFFNESS  = 5e5f; // Perimeter structural strength
    const float BALL_SPOKE_STIFFNESS = 3e4f; // Internal pressure preservation

    auto add_ball_spring = [&](int idxA, int idxB, float stiffness) {
        float dx = positions[idxA - start_particle_idx].x - positions[idxB - start_particle_idx].x;
        float dy = positions[idxA - start_particle_idx].y - positions[idxB - start_particle_idx].y;
        float exact_length = std::sqrt(dx * dx + dy * dy);
        springs.push_back(Spring(particles[idxA], particles[idxB], exact_length, stiffness));
    };

    for (int i = 0; i < EDGE_POINTS; i++) {
        int current_edge_idx = first_edge_index + i;
        int next_edge_idx = first_edge_index + ((i + 1) % EDGE_POINTS);

        // A. Structural Rim Wheel (Forms the circular skin of the football)
        add_ball_spring(current_edge_idx, next_edge_idx, BALL_SKIN_STIFFNESS);

        // B. Central Pressure Spokes (Prevents the football from collapsing inward)
        add_ball_spring(center_node_index, current_edge_idx, BALL_SPOKE_STIFFNESS);

        // C. Cross-Diametric Bracing (Maintains perfect roundness during high impact)
        int opposite_edge_idx = first_edge_index + ((i + (EDGE_POINTS / 2)) % EDGE_POINTS);
        if (i < EDGE_POINTS / 2) { 
            add_ball_spring(current_edge_idx, opposite_edge_idx, BALL_SPOKE_STIFFNESS * 0.5f);
        }
    }

    // Update global reference numbers for your game loop renderer
    num_particles = particles.size();
    num_springs = springs.size();

    return center_index;
}

void create_collision_test_rig(int& num_particles, std::vector<Particle>& particles, int& num_springs, std::vector<Spring>& springs, float dt) {
    // --- 1. PREVENT POINTER/REFERENCE DRAG (CRITICAL FOR YOUR ENGINE) ---
    // Total particles: 7 (Left Cluster) + 7 (Right Cluster) = 14 particles total.
    // Pre-reserving memory ensures that std::vector never reallocates and invalidates Spring references.
    particles.reserve(30);
    springs.reserve(100);

    // --- 2. CONFIGURATION PARAMETERS ---
    const float TEST_RADIUS = 15.0f;       // Large radius as requested
    const float TEST_MASS = 500.0f;        // Heavy mass to minimize viscosity damping impact
    const float CLUSTER_RADIUS = 50.0f;    // Spatial spacing radius from the local center
    const int RIM_POINTS = 6;              // Hexagonal outer rim layout (6 outer + 1 center = 7 particles per cluster)
    const float HIGH_SPEED = 200.0f;       // Aggressive initial velocity for collision testing

    // Cluster Centers (positioned horizontally, moving toward each other)
    sf::Vector2f left_center = { 400.0f, 540.0f };
    sf::Vector2f right_center = { 1500.0f, 545.0f }; // Slight Y offset to test angular deflection/rotation

    // --- 3. HELPER LAMBDA TO GENERATE A STRUCTURAL CLUSTER ---
    auto generate_cluster = [&](sf::Vector2f center, sf::Vector2f velocity) {
        int cluster_start_idx = particles.size();

        // A. Add Center Node
        particles.push_back(Particle(particles.size() + 1, TEST_RADIUS, TEST_MASS, center, velocity, structure::cluster));
        particles.back().set_acc({0.0f, 0.0f});

        // B. Add Perimeter Nodes (Hexagonal layout, mathematically guaranteeing no overlaps)
        for (int i = 0; i < RIM_POINTS; i++) {
            float angle = (i * 2.0f * 3.14159265f) / (float)RIM_POINTS;
            sf::Vector2f offset = { std::cos(angle) * CLUSTER_RADIUS, std::sin(angle) * CLUSTER_RADIUS };
            sf::Vector2f pos = center + offset;

            particles.push_back(Particle(particles.size() + 1, TEST_RADIUS, TEST_MASS, pos, velocity, structure::cluster));
            particles.back().set_acc({0.0f, 0.0f});
        }

        // C. Weave Structural Springs
        const float TEST_STIFFNESS = 1e4f; // Rigid springs to maintain shape during high-speed impact
        
        int center_idx = cluster_start_idx;
        int first_rim_idx = cluster_start_idx + 1;

        auto add_test_spring = [&](int idxA, int idxB) {
            sf::Vector2f pA = particles[idxA].get_curr_pos();
            sf::Vector2f pB = particles[idxB].get_curr_pos();
            float exact_length = std::sqrt((pA.x - pB.x)*(pA.x - pB.x) + (pA.y - pB.y)*(pA.y - pB.y));
            springs.push_back(Spring(particles[idxA], particles[idxB], exact_length, TEST_STIFFNESS));
        };

        for (int i = 0; i < RIM_POINTS; i+=2) {
            int current_rim = first_rim_idx + i;
            int next_rim = first_rim_idx + ((i + 1) % RIM_POINTS);

            // Rim Structural Edge
            add_test_spring(current_rim, next_rim);

            // Center Spoke
            // add_test_spring(center_idx, current_rim);

            // Cross-diameter support spring (avoids structural flattening)
            // int opposite_rim = first_rim_idx + ((i + (RIM_POINTS / 2)) % RIM_POINTS);
            // if (i < RIM_POINTS / 2) {
            //     add_test_spring(current_rim, opposite_rim);
            // }
        }
    };

    // --- 4. EXECUTE BUILD ---
    // Left cluster moving right fast
    generate_cluster(left_center, { HIGH_SPEED, 0.0f });

    // Right cluster moving left fast
    generate_cluster(right_center, { -HIGH_SPEED, 0.0f });

    // Sync reference metrics back to the engine runner loop
    num_particles = particles.size();
    num_springs = springs.size();
}

/*
// void create_creature_motor_sperm(Creature& creature, int& num_particles, std::vector<Particle>& particles, int& num_springs, std::vector<Spring>& springs, int& num_muscles, std::vector<Muscle>& muscles, float dt){
    
//     particles.clear();
//     springs.clear();
    
//     sf::Vector2<float> old_pos, vel, curr_pos, acc;

//     // --- CRITICAL PARAMETERS (PRESERVED UNCHANGED) ---
//     const float b_const = 4.f / 3.f * 3.141f * 20.f; 

//     // --- 1. DEFINE ORIGINAL CONTROL VERTICES ---
//     sf::Vector2f corners[9];
//     corners[0] = {300.0f, 150.0f}; // Head Center Core
//     corners[1] = {300.0f, 124.0f}; // 1. Top Apex
//     corners[2] = {307.5f, 137.0f}; // 2. Top-Right Mid-Wall
//     corners[3] = {315.0f, 150.0f}; // 3. Right Lateral Apex
//     corners[4] = {307.5f, 163.0f}; // 4. Bottom-Right Mid-Wall
//     corners[5] = {300.0f, 176.0f}; // 5. Bottom Apex (Tail Base)
//     corners[6] = {292.5f, 163.0f}; // 6. Bottom-Left Mid-Wall
//     corners[7] = {285.0f, 150.0f}; // 7. Left Lateral Apex
//     corners[8] = {292.5f, 137.0f}; // 8. Top-Left Mid-Wall

//     // Original guide landmarks for interpolation path
//     sf::Vector2f tail_landmarks[] = {
//         corners[5],       
//         {300.0f, 205.0f}, 
//         {300.0f, 235.0f}, 
//         {300.0f, 265.0f}, 
//         {300.0f, 295.0f}, 
//         {300.0f, 325.0f}  
//     };
    
//     float tail_radii[] = {1.2f, 1.6f, 1.4f, 1.2f, 1.0f, 0.8f};

//     std::vector<sf::Vector2f> positions;
//     std::vector<float> mass;
//     std::vector<float> radius;

//     // --- 2. GENERATE DENSE HEAD SHELL ---
//     positions.push_back(corners[0]);
//     radius.push_back(1.5f);
//     mass.push_back(1.04f * b_const * (1.5f * 1.5f * 1.5f)); 

//     int apex_indices[9];
//     const int HEAD_SUBDIVISIONS = 4; 

//     for(int i = 1; i <= 8; i++) {
//         apex_indices[i] = positions.size(); 
//         int next_corner_idx = (i == 8) ? 1 : i + 1;
        
//         sf::Vector2f pA = corners[i];
//         sf::Vector2f pB = corners[next_corner_idx];

//         for(int s = 0; s < HEAD_SUBDIVISIONS; s++) {
//             float t = (float)s / (float)HEAD_SUBDIVISIONS;
//             positions.push_back(pA + t * (pB - pA));
//             radius.push_back(1.2f);
//             mass.push_back(1.2f * b_const * (1.2f * 1.2f * 1.2f)); // Preserved 1.2f factor
//         }
//     }

//     // --- 3. GENERATE SINGLE-LINE SPINE CORE (Ultra-Slim Filament) ---
//     std::vector<int> spine_indices;
//     spine_indices.push_back(apex_indices[5]); // Tail connects to head base

//     const int TAIL_SUBDIVISIONS = 4; 
//     for(int i = 0; i < 5; i++) {
//         sf::Vector2f tA = tail_landmarks[i];
//         sf::Vector2f tB = tail_landmarks[i + 1];
//         float rA = tail_radii[i];  float rB = tail_radii[i + 1];

//         for(int s = 1; s <= TAIL_SUBDIVISIONS; s++) {
//             float t = (float)s / (float)TAIL_SUBDIVISIONS;
//             sf::Vector2f center_pos = tA + t * (tB - tA);
//             float current_radius = rA + t * (rB - rA);
            
//             spine_indices.push_back(positions.size());
//             positions.push_back(center_pos);
//             radius.push_back(current_radius);
//             mass.push_back(1.01f * b_const * (current_radius * current_radius * current_radius));
//         }
//     }

//     // --- 4. GENERATE INTERNAL MICRO-MOTOR TERMINALS ---
//     // These reside entirely within the spine line profile to keep visuals sleek
//     std::vector<int> motor_L(spine_indices.size(), -1);
//     std::vector<int> motor_R(spine_indices.size(), -1);
//     const float MOTOR_LEVER_ARM = 3.5f; // Extremely tight internal radius

//     for(size_t i = 1; i < spine_indices.size() - 1; i++) {
//         sf::Vector2f core_pos = positions[spine_indices[i]];
//         float r_motor = 0.3f; // Ultra-lightweight internal coordinate point
//         float m_motor = 1.01f * b_const * (r_motor * r_motor * r_motor);

//         // Left Motor Chassis Pin
//         motor_L[i] = positions.size();
//         positions.push_back(core_pos + sf::Vector2f(-MOTOR_LEVER_ARM, 0.0f));
//         radius.push_back(r_motor);
//         mass.push_back(m_motor);

//         // Right Motor Chassis Pin
//         motor_R[i] = positions.size();
//         positions.push_back(core_pos + sf::Vector2f(MOTOR_LEVER_ARM, 0.0f));
//         radius.push_back(r_motor);
//         mass.push_back(m_motor);
//     }

//     num_particles = positions.size(); 

//     // --- 5. INSTANTIATE PARTICLES IN ENGINE ---
//     for(int i = 0; i < num_particles; i++){
//         old_pos = positions[i] * 3.f;
//         vel.x = 0.01f * i; vel.y = 0.01f * i; 
//         curr_pos.x = vel.x * dt + old_pos.x; curr_pos.y = vel.y * dt + old_pos.y;
//         acc = {0.0f, 0.0f};

//         particles.push_back(Particle(i + 1, radius[i], mass[i], old_pos, curr_pos, structure::creature));
//         particles[i].set_acc(acc);
//     }

//     // --- 6. CONSTRUCT INDEPENDENT MOTOR NETWORK (SPRINGS) ---
//     const float RIGID_TENDON   = 1e5f;  
//     const float FLEXIBLE_SPINE = 5e3f;  
//     const float ACTUATOR       = 1e2f;  

//     auto add_spring = [&](int idxA, int idxB, float stiffness) {
//         float dx = positions[idxA].x - positions[idxB].x;
//         float dy = positions[idxA].y - positions[idxB].y;
//         float exact_length = std::sqrt(dx * dx + dy * dy) * 3.f;
//         springs.push_back(Spring(particles[idxA], particles[idxB], exact_length, stiffness));
//     };

//     // A. SOLID HEAD SPOKES & OUTER RING
//     for(int i = 1; i < spine_indices[1]; i++) {
//         add_spring(0, i, RIGID_TENDON);
//         int next_idx = (i == spine_indices[1] - 1) ? 1 : i + 1;
//         add_spring(i, next_idx, RIGID_TENDON);
//     }

//     // B. INEXTENSIBLE SPINE CONNECTIONS
//     // Sequential links are locked completely solid so length never shifts
//     for(size_t i = 0; i < spine_indices.size() - 1; i++) {
//         add_spring(spine_indices[i], spine_indices[i+1], RIGID_TENDON);
//     }

//     // C. RIGID MOTOR CHASSIS MOUNTING (Welding T-Bars to Upstream Segment)
//     for(size_t i = 1; i < spine_indices.size() - 1; i++) {
//         // Welded directly to the current joint vertex
//         add_spring(spine_indices[i], motor_L[i], RIGID_TENDON);
//         add_spring(spine_indices[i], motor_R[i], RIGID_TENDON);
        
//         // Cross-braced to the previous segment vertex to lock rotational alignment
//         add_spring(spine_indices[i-1], motor_L[i], RIGID_TENDON);
//         add_spring(spine_indices[i-1], motor_R[i], RIGID_TENDON);
//     }

//     // D. LOCALIZED MOTOR DRIVES & PASSIVE FLEXIBLE JOINTS
//     // Controls localized joint rotation. To keep your GA search space perfectly compact,
//     // we activate only 4 high-torque joint motor servos (ACTUATOR). The remaining 
//     // joints act as passive flexible couplers (FLEXIBLE_SPINE) to propagate the wave cleanly.
//     for(size_t i = 1; i < spine_indices.size() - 1; i++) {
        
//         // 4 distinct localized motor units down the flagellum backbone
//         bool is_active_servo = (i == 1 || i == 5 || i == 10 || i == 15);
//         float motor_stiffness = is_active_servo ? ACTUATOR : FLEXIBLE_SPINE;

//         // Drive links directly control the angle of the downstream segment
//         add_spring(motor_L[i], spine_indices[i+1], motor_stiffness);
//         add_spring(motor_R[i], spine_indices[i+1], motor_stiffness);
//     }

//     num_springs = springs.size();
// }

// void create_creature_bacteriophage(int& num_particles, std::vector<Particle>& particles, int& num_springs, std::vector<Spring>& springs, int& num_muscles, std::vector<Muscle>& muscles, float dt){
    
//     // Back to a clean 23 Particles: No weird outriggers needed anymore
//     num_particles = 23;
//     particles.clear();
//     springs.clear();
    
//     sf::Vector2<float> old_pos, vel, curr_pos, acc;

//     // --- 1. DEFINE BIOLOGICAL PROPERTIES (Mass, Radius, Initial Positions) ---
//     std::vector<float> x_coords(num_particles);
//     std::vector<float> y_coords(num_particles);
//     std::vector<float> mass(num_particles);
//     std::vector<float> radius(num_particles);

//     // [Capsid Head Core] - Heavy, dense center of mass (Keeps it oriented)
//     x_coords[0] = 300.0f; y_coords[0] = 140.0f; mass[0] = 6.0f; radius[0] = 1.5f;

//     // [Slim Capsid Outer Loop] - Clean, streamlined bacteriophage head profile
//     x_coords[1] = 300.0f; y_coords[1] = 110.0f; mass[1] = 0.3f; radius[1] = 1.8f; // Top Apex
//     x_coords[2] = 312.0f; y_coords[2] = 125.0f; mass[2] = 0.3f; radius[2] = 1.8f; // Top-Right
//     x_coords[3] = 312.0f; y_coords[3] = 155.0f; mass[3] = 0.8f; radius[3] = 1.5f; // Bottom-Right Base
//     x_coords[4] = 300.0f; y_coords[4] = 165.0f; mass[4] = 1.0f; radius[4] = 1.5f; // Neck Center
//     x_coords[5] = 288.0f; y_coords[5] = 155.0f; mass[5] = 0.8f; radius[5] = 1.5f; // Bottom-Left Base
//     x_coords[6] = 288.0f; y_coords[6] = 125.0f; mass[6] = 0.3f; radius[6] = 1.8f; // Top-Left

//     // [Central Sheath & Baseplate Trunk] - High-density vertical backbone (Sinking ballast)
//     x_coords[7] = 300.0f; y_coords[7] = 190.0f; mass[7] = 5.0f; radius[7] = 1.2f; // Mid Sheath
//     x_coords[8] = 300.0f; y_coords[8] = 220.0f; mass[8] = 6.0f; radius[8] = 1.2f; // Baseplate Center
//     x_coords[9] = 292.0f; y_coords[9] = 220.0f; mass[9] = 2.0f; radius[9] = 1.2f; // Left Hip Joint
//     x_coords[10]= 308.0f; y_coords[10]= 220.0f; mass[10]= 2.0f; radius[10]= 1.2f; // Right Hip Joint

//     // [Special Muscle Insertion Points - Legs] - Tucked tight to hips for clean look
//     x_coords[11] = 295.0f; y_coords[11] = 210.0f; mass[11] = 0.3f; radius[11] = 1.0f; 
//     x_coords[12] = 295.0f; y_coords[12] = 230.0f; mass[12] = 0.3f; radius[12] = 1.0f; 
//     x_coords[13] = 305.0f; y_coords[13] = 210.0f; mass[13] = 0.3f; radius[13] = 1.0f; 
//     x_coords[14] = 305.0f; y_coords[14] = 230.0f; mass[14] = 0.3f; radius[14] = 1.0f; 

//     // [Tail Fibers / Moving Legs] - Low mass, high radius (Highly buoyant swimming paddles)
//     // Outer Leg Pair
//     x_coords[15] = 272.0f; y_coords[15] = 240.0f; mass[15] = 0.4f; radius[15] = 1.4f; // Left Knee 1
//     x_coords[16] = 255.0f; y_coords[16] = 270.0f; mass[16] = 0.1f; radius[16] = 2.2f; // Left Tip 1
//     x_coords[17] = 328.0f; y_coords[17] = 240.0f; mass[17] = 0.4f; radius[17] = 1.4f; // Right Knee 1
//     x_coords[18] = 345.0f; y_coords[18] = 270.0f; mass[18] = 0.1f; radius[18] = 2.2f; // Right Tip 1
//     // Inner Leg Pair
//     x_coords[19] = 282.0f; y_coords[19] = 245.0f; mass[19] = 0.4f; radius[19] = 1.4f; // Left Knee 2
//     x_coords[20] = 268.0f; y_coords[20] = 285.0f; mass[20] = 0.1f; radius[20] = 2.2f; // Left Tip 2
//     x_coords[21] = 318.0f; y_coords[21] = 245.0f; mass[21] = 0.4f; radius[21] = 1.4f; // Right Knee 2
//     x_coords[22] = 332.0f; y_coords[22] = 285.0f; mass[22] = 0.1f; radius[22] = 2.2f; // Right Tip 2

//     // --- 2. INITIALIZE PARTICLES ---
//     for(int i = 0; i < num_particles; i++){
//         old_pos.x = x_coords[i];
//         old_pos.y = y_coords[i];
        
//         vel.x = 0.1f * i; 
//         vel.y = 0.1f * i;
        
//         curr_pos.x = vel.x * dt + old_pos.x; 
//         curr_pos.y = vel.y * dt + old_pos.y;
        
//         acc.x = 0.0f; 
//         acc.y = 0.0f;

//         particles.push_back(Particle(i + 1, radius[i], mass[i], old_pos, curr_pos, structure::creature));
//         particles[i].set_acc(acc);
//     }

//     // --- 3. CREATE CONNECTIVITY MATRIX (Springs) ---
//     const float TENDON = 1e5f;
//     const float MUSCLE = 1e2f;

//     auto add_spring = [&](int idxA, int idxB, float stiffness) {
//         float dx = x_coords[idxA] - x_coords[idxB];
//         float dy = y_coords[idxA] - y_coords[idxB];
//         float exact_length = std::sqrt(dx * dx + dy * dy);
//         springs.push_back(Spring(particles[idxA], particles[idxB], exact_length, stiffness));
//     };

//     // A. SLIM HEAD STRUCTURAL EXOSKELETON (Tendons)
//     for(int i = 1; i <= 6; i++) add_spring(0, i, TENDON); 
//     for(int i = 1; i <= 5; i++) add_spring(i, i + 1, TENDON); 
//     add_spring(6, 1, TENDON);

//     // B. ANTI-ROTATION SOLID TRUSS NECK (Tendons)
//     // Connecting the head's base base corners (3 and 5) directly to the spine (7) with high stiffness
//     add_spring(4, 7, TENDON); // Center neck bone
//     add_spring(3, 7, TENDON); // Right neck structural brace (Stops left/right tilting)
//     add_spring(5, 7, TENDON); // Left neck structural brace (Stops left/right tilting)

//     // C. RIGID SPINE & BASEPLATE (Tendons)
//     add_spring(7, 8,  TENDON); // Mid Spine to Baseplate Center
//     add_spring(8, 9,  TENDON); // Center to Left Hip
//     add_spring(8, 10, TENDON); // Center to Right Hip
//     add_spring(7, 9,  TENDON); 
//     add_spring(7, 10, TENDON); 

//     // D. SECURE LEG MUSCLE ANCHORS TO TRUNK (Tendons)
//     add_spring(11, 7, TENDON); add_spring(11, 9, TENDON);
//     add_spring(12, 8, TENDON); add_spring(12, 9, TENDON);
//     add_spring(13, 7, TENDON); add_spring(13, 10, TENDON);
//     add_spring(14, 8, TENDON); add_spring(14, 10, TENDON);

//     // E. LEG STRUCTURAL BONES (Tendons)
//     add_spring(9,  15, TENDON); add_spring(15, 16, TENDON); 
//     add_spring(10, 17, TENDON); add_spring(17, 18, TENDON); 
//     add_spring(9,  19, TENDON); add_spring(19, 20, TENDON); 
//     add_spring(10, 21, TENDON); add_spring(21, 22, TENDON); 

//     // F. ACTUATED LEG MUSCLES (Muscles)

//     // --- Leg 1 (Left Outer) Muscles ---
//     add_spring(11, 15, MUSCLE); add_spring(12, 15, MUSCLE);
//     add_spring(11, 16, MUSCLE); add_spring(12, 16, MUSCLE);

//     // --- Leg 2 (Right Outer) Muscles ---
//     add_spring(13, 17, MUSCLE); add_spring(14, 17, MUSCLE);
//     add_spring(13, 18, MUSCLE); add_spring(14, 18, MUSCLE);

//     // --- Leg 3 (Left Inner) Muscles ---
//     add_spring(11, 19, MUSCLE); add_spring(12, 19, MUSCLE);
//     add_spring(11, 20, MUSCLE); add_spring(12, 20, MUSCLE);

//     // --- Leg 4 (Right Inner) Muscles ---
//     add_spring(13, 21, MUSCLE); add_spring(14, 21, MUSCLE);
//     add_spring(13, 22, MUSCLE); add_spring(14, 22, MUSCLE);

//     num_springs = springs.size();
// }

*/