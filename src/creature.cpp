#include "headers/creature.hpp"
#include <iostream>

Eigen::Rand::P8_mt19937_64 urng(42);
std::random_device rd;
std::mt19937 gen(rd());
// Eigen::Rand::P8_mt19937_64 urng(rand()%UINT64_MAX);

// 1 iteration is one wave 
// total_iters * freq = 2 pi
const float base_freq = (2.f * 3.141f) / ((float)env_num_iterations_per_frame * (float)env_num_frames_per_creature_action);
std::normal_distribution<float> dist(0, base_freq / 10.f);

Neural_Net::Neural_Net(std::vector<int> layer_sizes)
    :m_layer_sizes(layer_sizes.begin(), layer_sizes.end())
{

    int num_matrics = layer_sizes.size()-1;
    for(int i = 0; i < num_matrics; i++){
        int in_sz = layer_sizes[i];
        int out_sz = layer_sizes[i+1];

        m_weights.push_back(Eigen::Rand::balanced<MatrixXdf>(in_sz, out_sz, urng));
        m_biases.push_back(Eigen::Rand::balanced<MatrixXdf>(1, out_sz, urng));

        m_freq = base_freq + dist(gen) * 5.f;
    }
}

Neural_Net::Neural_Net(std::vector<MatrixXdf>& weights, std::vector<VectorXdf>& biases, float freq)
    :m_weights(weights.begin(), weights.end()),
     m_biases(biases.begin(), biases.end()),
     m_freq(freq)
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
float Neural_Net::get_freq() const{
    return m_freq;
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
    data["freq"] = m_freq;

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
                            float par_1_f = par_1.get_freq();
    
    std::vector<MatrixXdf> const& par_2_w = par_2.get_weights();
    std::vector<VectorXdf> const& par_2_b = par_2.get_biases();
                            float par_2_f = par_2.get_freq();

    int final_out = m_layer_sizes.back();
    float break_off = float(rand()%(final_out+1))/final_out;

    // std::cout << break_off << "\n\n";
    // std::cout << par_1_w[0] << "\n\n";
    // std::cout << par_2_w[0] << "\n\n";
    // std::cout << m_weights[0] << "\n\n";

    //! ONYL DOING CROSSOVER FOR THE FIRST LAYER
    for(int i = 0; i < 1; i++){
    // for(int i = 0; i < (int)m_layer_sizes.size()-1; i++){
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

    if(rand() % 2 == 0){
        m_freq = par_1_f;
    }
    else{
        m_freq = par_2_f;
    }
    // std::cout << m_weights[0] << "\n\n";
    // std::cout.flush();

    // exit(0);
}

void Neural_Net::mutate(float mutation_rate){

    for(int i = 0; i < (int)m_layer_sizes.size()-1; i++){
        int in_sz = m_layer_sizes[i];
        int out_sz = m_layer_sizes[i+1];
        
        MatrixXdf weight_noise = Eigen::Rand::normal<MatrixXdf>(in_sz, out_sz, urng);
        MatrixXdf rand_w_mat    = Eigen::Rand::balanced<MatrixXdf>(1, out_sz, urng);
        // rand w mat has [-1, 1] so we use 2*mutation_rate-1 instead of mutations rate
        MatrixXdf weight_mask = (rand_w_mat.array() < 2*mutation_rate-1).cast<float>();
        m_weights[i] += m_mutation_std * weight_noise.cwiseProduct(weight_mask.replicate(in_sz, 1));
        
        MatrixXdf bias_noise = Eigen::Rand::normal<MatrixXdf>(1, out_sz, urng);
        MatrixXdf rand_b_mat  = Eigen::Rand::balanced<MatrixXdf>(1, out_sz, urng);
        MatrixXdf bias_mask = (rand_b_mat.array() < 2*mutation_rate-1).cast<float>();
        m_biases[i] += m_mutation_std * bias_noise.cwiseProduct(bias_mask);

        if(rand()%10 < mutation_rate * 10)
            m_freq += dist(gen);
    }
}

void Neural_Net::forward(std::vector<float>& current_activations, std::vector<float>& observation, int t){

    VectorXdf activation(m_layer_sizes[0]);
    assert(m_layer_sizes[0] == (int)(observation.size()));
    assert(env_observation_size == (int)(observation.size()));

    for(int i=0; i<env_observation_size-1; i++){
        activation(0, i) = observation[i];
    }
    activation(0, env_observation_size-1) = sin(t * m_freq);
    // std::cout << activation(0, env_observation_size-1) << '\n';
    // std::cout << activation << '\n';

    for(int layer_no = 0; layer_no < (int)m_weights.size()-1; layer_no++){
        activation = activation * m_weights[layer_no] + m_biases[layer_no];
        activation = activation.cwiseMax(0);
    }
    activation = activation * m_weights[m_weights.size()-1] + m_biases[m_biases.size()-1];
    // approximate sigmoid as x/(1+|x|)
    activation = 0.5f*(1.f + (activation.array()/(1.f+activation.array().abs())));
    
    for(int i = 0; i < (int)current_activations.size(); i++){
        current_activations[i] = activation(0, i);
    }

    // for(int i = 0; i < (int)current_activations.size(); i++){
    //     current_activations[i] = 1 - i % 2;
    //     // if i forget
    //     assert(num_generations == 0);
    // }
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

Creature::Creature(std::vector<int> muscle_index, std::vector<int> sensing_points, std::vector<MatrixXdf>& weights, std::vector<VectorXdf>& biases, float freq)
    :m_muscle_index(muscle_index.begin(), muscle_index.end()),
    m_current_activations(m_muscle_index.size(), 0),
    m_sensing_points(sensing_points.begin(), sensing_points.end()),
    m_brain(weights, biases, freq)
{}

Neural_Net const& Creature::get_brain() const{
    return m_brain;
}

void Creature::crossover(Creature const& par_1, Creature const& par_2){
    m_brain.crossover(par_1.get_brain(), par_2.get_brain());
}

void Creature::mutate(float mut_rate){
    m_brain.mutate(mut_rate);
}

void Creature::save(int id, float reward){
    m_brain.save(id, reward);
}

void Creature::copy_brain(Creature const& creature){
    // neural net has no const members and refs so safe to copy
    m_brain = creature.get_brain();
}

int Creature::get_apex_tip_index(){
    return m_sensing_points[1];
}

int Creature::get_bottom_tip_index(){
    return m_sensing_points.back();
}

void Creature::act(std::vector<Muscle>& muscles, std::vector<float>& observation, int t){
    m_brain.forward(m_current_activations, observation, t);
    for(int i=0; i < (int)m_muscle_index.size(); i++){
        muscles[i].set_activation(m_current_activations[i]);
    }
}

void Creature::get_observation(std::vector<float>& obs, sf::Vector2f ball_pos, sf::Vector2f goal_pos, const std::vector<Particle>& particles){
    
    // use a parameter somewhere instead of hardcoding 12
    // -1 for the sin thing added later on
    assert(obs.size() == env_observation_size);
    
    // for ball
    sf::Vector2f vec_head_to_ball = (ball_pos-particles[m_sensing_points[1]].get_curr_pos());
    float ball_dist = vec_head_to_ball.length();
    vec_head_to_ball /= ball_dist;
    
    sf::Vector2f vec_ctr_apx_tip = (particles[m_sensing_points[1]].get_curr_pos()-particles[m_sensing_points[0]].get_curr_pos());
    // assert(11.f == vec_ctr_apx_tip.length());
    vec_ctr_apx_tip = vec_ctr_apx_tip.normalized();
    
    obs[0] = ball_dist/700.f;
    obs[1] = vec_ctr_apx_tip.angleTo(vec_head_to_ball).asDegrees()/180.f;
    
    sf::Vector2f head_vel = (particles[m_sensing_points[0]].get_vel());
    obs[2] = head_vel.dot(vec_head_to_ball);
    obs[3] = head_vel.dot({-vec_head_to_ball.y, vec_head_to_ball.x});
    
    sf::Vector2f rel_goal_pos = (goal_pos-particles[m_sensing_points[1]].get_curr_pos());
    obs[4] = rel_goal_pos.length()/700.f;
    obs[5] = rel_goal_pos.angleTo(vec_head_to_ball).asDegrees()/180.f;
    
    // tail curvatures
    // sensing pts are 2-3-4, 5-6-7, 8-9-10, ...
    int lines = (m_sensing_points.size() - 2) - 2;
    assert(6+lines+5 == env_observation_size);
    assert(lines == 8);
    for(int i = 0; i < lines; i++){
        sf::Vector2f vec1 = particles[m_sensing_points[2 + i]].get_curr_pos();
        sf::Vector2f vec2 = particles[m_sensing_points[2 + i+1]].get_curr_pos();
        sf::Vector2f vec3 = particles[m_sensing_points[2 + i+2]].get_curr_pos();
        obs[i+6] = (vec2-vec1).angleTo(vec3-vec2).asDegrees()/90.f;
    }
    
    
    // for walls
    // points 0, 3
    obs[6+lines+0] = 2.f*(particles[m_sensing_points[0]].get_curr_pos().x/max_x) - 1.f;
    obs[6+lines+1] = 2.f*(particles[m_sensing_points[0]].get_curr_pos().y/max_y) - 1.f;
    obs[6+lines+2] = 2.f*(particles[m_sensing_points.back()].get_curr_pos().x/max_x) - 1.f;
    obs[6+lines+3] = 2.f*(particles[m_sensing_points.back()].get_curr_pos().y/max_y) - 1.f;

    // obs[10 + 4] = this is for sin wave

}
// void Creature::get_observation_gpu(std::vector<float>& obs, sf::Vector2f ball_pos, sf::Vector2f goal_pos, const std::vector<Particle>& particles){
    
//     // use a parameter somewhere instead of hardcoding 12
//     assert(obs.size() == 12);
//     std::vector<int> vals(12);
//     // for ball
//     float mn = 10000;
//     float mx = 0;
//     for(int i = 0; i < 4; i++){
//         vals[i] = (ball_pos-particles[m_sensing_points[i]].get_curr_pos_gpu()).length();
//         mn = fmin(mn, vals[i]);
//         mx = fmax(mx, vals[i]);
//     }

//     float dem = mx-mn;
//     for(int i = 0; i < 4; i++){
//         obs[i] = 2.f*(vals[i]-mn)/dem - 1;
//     }

//     // for goal
//     mn = 10000;
//     mx = 0;
//     for(int i = 4; i < 8; i++){
//         vals[i] = (goal_pos-particles[m_sensing_points[i-4]].get_curr_pos_gpu()).length();
//         mn = fmin(mn, vals[i]);
//         mx = fmax(mx, vals[i]);
//     }

//     dem = mx-mn;
//     for(int i = 4; i < 8; i++){
//         obs[i] = 2.f*(vals[i]-mn)/dem - 1;
//     }

//     // for walls
//     // points 0, 3
//     obs[8] = 2.f*(particles[m_sensing_points[0]].get_curr_pos_gpu().x/max_x) - 1.f;
//     obs[9] = 2.f*(particles[m_sensing_points[0]].get_curr_pos_gpu().y/max_y) - 1.f;
    
//     obs[10] = 2.f*(particles[m_sensing_points[3]].get_curr_pos_gpu().x/max_x) - 1.f;
//     obs[11] = 2.f*(particles[m_sensing_points[3]].get_curr_pos_gpu().y/max_y) - 1.f;
// }

// ============================================================================
// create_creature_muscle_swimmer
//
// Redesign goals vs. the original:
//  1. Remove the redundant second "head" welded to the tail tip. Real
//     fish/eel tails taper to a light, flexible tip -- that's where you
//     want max flexibility and min inertia, not more rigid structure/mass.
//  2. Fix the stiffness hierarchy so it's *consistent* along the whole tail:
//         RIGID_TENDON (head)  >>  ACTUATOR  >  RUNG  >  SPINE  >  BRACE
//     In the original, `FLEXIBLE_SPINE * scaling` could exceed 1e7 -- an
//     order of magnitude stiffer than the "rigid" head tendon (1e6), and
//     comparable to or stiffer than the muscle actuator. A muscle can't
//     bend a passive structure that's stiffer than itself.
//  3. Taper stiffness DOWN toward the tail tip (was backwards before --
//     the old `dist`-based scaling made the tail *stiffer* near its ends).
//     This lets the wave amplify toward the tip, like a real fish tail.
//  4. Actuate every segment (both rails), not every other segment. This
//     gives a controller/learner the full space of tail shapes needed to
//     synthesize a clean traveling wave, instead of half the DOF being
//     dead weight.
//  5. Add proprioceptive sensing points along the tail (not just at the
//     ends) so a learned controller can actually perceive body curvature
//     / wave phase -- necessary feedback for coordinating an undulation.
//  6. Slightly leaner head profile (less frontal drag) and a slightly
//     higher head:tail mass ratio (more stable "keel" so tail thrust
//     doesn't just spin the body in place).
// ============================================================================

Creature_data create_creature_muscle_swimmer(int& num_particles, std::vector<Particle>& particles, int& num_springs, std::vector<Spring>& springs
                                            , int& num_muscles, std::vector<Muscle>& muscles, int env_id){

    sf::Vector2<float> old_pos, vel, acc;

    // --- CRITICAL PARAMETERS ---
    int id_offset = num_particles;
    int spring_id = num_springs + num_muscles;
    sf::Vector2f shift = {660, 290};

    const float b_const = 4.f / 3.f * 3.141f * 10.f;
    const float head_heaviness = 4.f; // slightly heavier keel than before (was 3.f)

    // --- 1. HEAD: leaner profile than before (less frontal drag) ---
    sf::Vector2f corners[9];

    float head_vertical_length = 22.0f;              // was 25.0f -- leaner
    float horizontal_length = head_vertical_length * (26.0f / 52.0f); // narrower ratio than 30/52

    float half_h = head_vertical_length / 2.0f;
    float half_w = horizontal_length / 2.0f;
    float mid_h  = half_h / 2.0f;
    float mid_w  = half_w / 2.0f;

    sf::Vector2f center = {300.0f, 150.0f};

    corners[0] = center;
    corners[1] = center + sf::Vector2f{ 0.0f,   -half_h };
    corners[2] = center + sf::Vector2f{  mid_w,  -mid_h  };
    corners[3] = center + sf::Vector2f{  half_w,  0.0f   };
    corners[4] = center + sf::Vector2f{  mid_w,   mid_h  };
    corners[5] = center + sf::Vector2f{  0.0f,    half_h };
    corners[6] = center + sf::Vector2f{ -mid_w,   mid_h  };
    corners[7] = center + sf::Vector2f{ -half_w,  0.0f   };
    corners[8] = center + sf::Vector2f{ -mid_w,  -mid_h  };

    // --- TAIL GUIDE PATH: 6 landmarks, more pronounced taper than before ---
    sf::Vector2f tail_landmarks[] = {
        corners[5],
        {300.0f, 185.0f},
        {300.0f, 220.0f},
        {300.0f, 255.0f},
        {300.0f, 290.0f},
        {300.0f, 325.0f}
    };
    // Thicker near the head (more muscle leverage / structural support),
    // thinner at the tip (lower inertia, easier to whip).
    float tail_radii[] = {1.8f, 1.5f, 1.1f, 0.8f, 0.6f, 0.45f};

    std::vector<sf::Vector2f> positions;
    std::vector<float> mass;
    std::vector<float> radius;

    float head_part_radius = 0.7f;

    // --- 2. DENSE HEAD SHELL (unchanged approach, leaner geometry) ---
    positions.push_back(corners[0]);
    radius.push_back(0.75f);
    mass.push_back(head_heaviness * b_const * (pow(1.5f, 3)));

    int apex_indices[9];
    const int HEAD_SUBDIVISIONS = 2;

    for(int i = 1; i <= 8; i++) {
        apex_indices[i] = positions.size();
        int next_corner_idx = (i == 8) ? 1 : i + 1;
        sf::Vector2f pA = corners[i];
        sf::Vector2f pB = corners[next_corner_idx];
        for(int s = 0; s < HEAD_SUBDIVISIONS; s++) {
            float t = (float)s / (float)HEAD_SUBDIVISIONS;
            positions.push_back(pA + t * (pB - pA));
            radius.push_back(head_part_radius);
            mass.push_back(head_heaviness * b_const * (pow(0.7f, 3)));
        }
    }

    // --- 3. TAIL: single tapered flexible double-rail whip, no tip head ---
    std::vector<int> left_tail_indices;
    std::vector<int> right_tail_indices;
    int tail_midpoint_index = -1;

    // 5 spans, 5 subdivisions each = 25 segments. Fewer than the original's
    // 30, but still enough spatial resolution for a smooth wave, while
    // keeping the action/observation space a bit more tractable to learn.
    std::vector<int> tail_subdivs{5, 5, 5, 5, 5};
    int num_tail_segments = 0;
    for (int s : tail_subdivs) num_tail_segments += s;

    int seg_counter = 0;
    for(int i = 0; i < 5; i++) {
        sf::Vector2f tA = tail_landmarks[i];
        sf::Vector2f tB = tail_landmarks[i + 1];
        float rA = tail_radii[i];  float rB = tail_radii[i + 1];

        const int TAIL_SUBDIVISIONS = tail_subdivs[i];
        for(int s = 1; s <= TAIL_SUBDIVISIONS; s++) {
            float t = (float)s / (float)TAIL_SUBDIVISIONS;
            sf::Vector2f center_pos = tA + t * (tB - tA);
            float current_radius = rA + t * (rB - rA);
            float half_width = current_radius * 1.4f;

            if (seg_counter == num_tail_segments / 2) {
                tail_midpoint_index = positions.size() + id_offset;
            }

            left_tail_indices.push_back(positions.size());
            positions.push_back(center_pos + sf::Vector2f(-half_width, 0.0f));
            radius.push_back(current_radius * 0.7f);
            mass.push_back(b_const * (radius.back() * radius.back() * radius.back()));

            right_tail_indices.push_back(positions.size());
            positions.push_back(center_pos + sf::Vector2f(half_width, 0.0f));
            radius.push_back(current_radius * 0.7f);
            mass.push_back(b_const * (radius.back() * radius.back() * radius.back()));

            seg_counter++;
        }
    }

    // --- 4. INSTANTIATE PARTICLES ---
    std::vector<int> sensing_points;
    for(int i = 0; i < (int)positions.size(); i++){
        old_pos = positions[i] + shift;
        vel = {0.0f, 0.0f};
        acc = {0.0f, 0.0f};
        particles.push_back(Particle(env_id, i + id_offset, radius[i], mass[i], old_pos, vel, structure::creature));

        // Orientation landmarks on the head.

        // only rhe apex tip
        if(positions[i] == corners[1] || positions[i] == corners[0]) {
        // if(positions[i] == corners[1] or positions[i] == corners[3] or positions[i] == corners[7]) {
            sensing_points.push_back(particles.size()-1);
        }
    }
    // Proprioceptive sensing along the tail: roughly 1/4, 1/2, 3/4 and tip.
    // This gives a controller feedback about the body's current curvature /
    // wave phase, which it needs in order to coordinate an undulation --
    // the original design only exposed a single point at the (now removed)
    // second head.
    std::vector<int> fracs;
    for(int i = 0; i < 10; i++)
        fracs.push_back(i * 10);
    
    for (int frac : fracs) {
        int idx = std::min((int)left_tail_indices.size() - 1,
                            (int)((frac / 100.0f) * (left_tail_indices.size() - 1)));
        // sensing_points.push_back(left_tail_indices[idx]);
        sensing_points.push_back(id_offset + right_tail_indices[idx]);
    }

    // --- 5. SPRINGS ---
    // Stiffness hierarchy is kept globally consistent:
    //   RIGID_TENDON (head)  >>  ACTUATOR  >  RUNG  >  SPINE  >  BRACE
    // and this ordering holds at every point along the taper, unlike the
    // original where scaling could push SPINE above RIGID_TENDON.
    const float RIGID_TENDON = 1e6f;

    const float RUNG_BASE  = 1.5e4f, RUNG_MIN  = 4.0e3f; // holds ladder width
    const float SPINE_BASE = 4.0e3f, SPINE_MIN = 1.0e3f; // rail bending resistance
    const float BRACE_STIFF = 1.0e3f;                    // constant light anti-buckling shear
    const float ACT_BASE = 3.0e4f, ACT_MIN = 1.2e4f;     // muscle contraction stiffness

    auto add_spring = [&](int idxA, int idxB, float stiffness, bool outside_body) {
        float dx = positions[idxA].x - positions[idxB].x;
        float dy = positions[idxA].y - positions[idxB].y;
        float exact_length = std::sqrt(dx * dx + dy * dy);
        assert(exact_length != 0);
        springs.push_back(Spring(particles[idxA+id_offset], particles[idxB+id_offset], env_id, spring_id, exact_length, stiffness, outside_body));
        spring_id++;
    };

    auto add_muscle = [&](int idxA, int idxB, float stiffness, bool outside_body) {
        float dx = positions[idxA].x - positions[idxB].x;
        float dy = positions[idxA].y - positions[idxB].y;
        float exact_length = std::sqrt(dx * dx + dy * dy);
        assert(exact_length != 0);
        muscles.push_back(Muscle(particles[idxA+id_offset], particles[idxB+id_offset], env_id, spring_id, exact_length, stiffness, outside_body));
        spring_id++;
    };

    // 5a. Head spokes & outer ring (unchanged approach -- keep the head rigid).
    int head_left_base  = apex_indices[6] + 2;
    int head_right_base = apex_indices[4] - 2;
    int head_left_bottom  = apex_indices[6];
    int head_right_bottom = apex_indices[4];
    int bottom_tip = apex_indices[5];

    for(int i = 1; i < left_tail_indices[0]; i++) {
        add_spring(0, i, RIGID_TENDON, false);
        int next_idx = (i == left_tail_indices[0] - 1) ? 1 : i + 1;
        bool in_gap = (i >= head_right_base && i < head_left_base);
        add_spring(i, next_idx, RIGID_TENDON, !in_gap);
    }

    // 5b. Anchor tail root to base of head.
    add_spring(left_tail_indices[0], head_left_base, RIGID_TENDON, true);
    add_spring(head_right_base, right_tail_indices[0], RIGID_TENDON, true);
    add_spring(head_left_bottom, left_tail_indices[1 % left_tail_indices.size()], RIGID_TENDON, false);
    add_spring(head_right_bottom, right_tail_indices[1 % right_tail_indices.size()], RIGID_TENDON, false);
    add_spring(bottom_tip, left_tail_indices[0], RIGID_TENDON, false);
    add_spring(bottom_tip, right_tail_indices[0], RIGID_TENDON, false);
    add_spring(apex_indices[7], apex_indices[3], RIGID_TENDON, false);
    add_spring(apex_indices[1], apex_indices[5], RIGID_TENDON, false);

    // 5c. Tail structure: rungs, rails (spine), shear braces -- all tapered
    // so the tail gets progressively softer toward the tip. No second head:
    // the final rung spring (left.back()-right.back()) simply closes the tip.
    for(int i = 0; i < num_tail_segments; i++) {
        float t = (float)i / (float)(num_tail_segments - 1); // 0 at head, 1 at tip

        float rung_stiff  = RUNG_BASE  - (RUNG_BASE  - RUNG_MIN)  * t;
        float spine_stiff = SPINE_BASE - (SPINE_BASE - SPINE_MIN) * t;

        // Cross-rung: holds the thin ladder profile width.
        add_spring(left_tail_indices[i], right_tail_indices[i], rung_stiff, false);

        if (i < num_tail_segments - 1) {
            // Rails (longitudinal spine) -- always softer than the muscle
            // that has to bend them.
            add_spring(left_tail_indices[i], left_tail_indices[i+1], spine_stiff, true);
            add_spring(right_tail_indices[i], right_tail_indices[i+1], spine_stiff, true);
        }

        if (i + 3 < num_tail_segments) {
            // Shear diagonals prevent buckling/overlap; kept light and flat
            // so they don't dominate the passive stiffness budget.
            add_spring(left_tail_indices[i], right_tail_indices[i+3], BRACE_STIFF, false);
            add_spring(right_tail_indices[i], left_tail_indices[i+3], BRACE_STIFF, false);
        }
    }

    // 5d. Muscles: EVERY segment gets an antagonist pair (both rails),
    // tapered so contraction strength eases off toward the tip -- this is
    // what lets the tip whip with amplified amplitude rather than being
    // dragged rigidly by a strong distal actuator.
    for(int i = 0; i < num_tail_segments - 1; i++) {
        float t = (float)i / (float)(num_tail_segments - 2);
        float act_stiff = ACT_BASE - (ACT_BASE - ACT_MIN) * t;

        add_muscle(right_tail_indices[i], right_tail_indices[i+1], act_stiff, true);
        add_muscle(left_tail_indices[i], left_tail_indices[i+1], act_stiff, true);
    }

    num_particles = particles.size();
    num_springs = springs.size();
    num_muscles = muscles.size();

    std::vector<int> muscle_indices;
    for(int i = 0; i < num_muscles; i++){
        muscle_indices.push_back(i);
    }

    Creature_data data{ muscle_indices, sensing_points, tail_midpoint_index };
    return data;
}

/*
Creature_data create_creature_muscle_sperm(int& num_particles, std::vector<Particle>& particles, int& num_springs, std::vector<Spring>& springs
                                            , int& num_muscles, std::vector<Muscle>& muscles, int env_id){

    sf::Vector2<float> old_pos, vel, curr_pos, acc;

    // --- CRITICAL PARAMETERS (PRESERVED UNCHANGED) ---
    int id_offset = num_particles;
    int spring_id = num_springs + num_muscles;
    sf::Vector2f shift = {660, 290};

    // we cant really make things lighter unless we lower the spring constants
    const float b_const = 4.f / 3.f * 3.141f * 10.f; 
    const float head_heaviness = 3.f;

    // --- 1. DEFINE ORIGINAL CONTROL VERTICES ---
    sf::Vector2f corners[9];
    sf::Vector2f b_corners[9];
    // corners[0] = {300.0f, 150.0f}; // Head Center Core
    // corners[1] = {300.0f, 124.0f}; // 1. Top Apex
    // corners[2] = {307.5f, 137.0f}; // 2. Top-Right Mid-Wall
    // corners[3] = {315.0f, 150.0f}; // 3. Right Lateral Apex
    // corners[4] = {307.5f, 163.0f}; // 4. Bottom-Right Mid-Wall
    // corners[5] = {300.0f, 176.0f}; // 5. Bottom Apex (Tail Base)
    // corners[6] = {292.5f, 163.0f}; // 6. Bottom-Left Mid-Wall
    // corners[7] = {285.0f, 150.0f}; // 7. Left Lateral Apex
    // corners[8] = {292.5f, 137.0f}; // 8. Top-Left Mid-Wall
// --- TWEAKABLE DIMENSION PARAMETER ---
    float head_vertical_length = 25.0f; // Change this to dynamically adjust the height
    float vertical_length = head_vertical_length; // Change this to dynamically adjust the height

    // Proportional width scale derived from your original profile ratio (30.0w / 52.0h)
    float horizontal_length = vertical_length * (30.0f / 52.0f); 

    float half_h = vertical_length / 2.0f;
    float half_w = horizontal_length / 2.0f;
    float mid_h  = half_h / 2.0f;
    float mid_w  = half_w / 2.0f;

    sf::Vector2f center = {300.0f, 150.0f};

    // --- CALCULATED COORDINATES ---
    corners[0] = center;                                                 // Head Center Core
    corners[1] = center + sf::Vector2f{ 0.0f,   -half_h };               // 1. Top Apex
    corners[2] = center + sf::Vector2f{  mid_w,  -mid_h  };              // 2. Top-Right Mid-Wall
    corners[3] = center + sf::Vector2f{  half_w,  0.0f   };              // 3. Right Lateral Apex
    corners[4] = center + sf::Vector2f{  mid_w,   mid_h  };              // 4. Bottom-Right Mid-Wall
    corners[5] = center + sf::Vector2f{  0.0f,    half_h };               // 5. Bottom Apex (Tail Base)
    corners[6] = center + sf::Vector2f{ -mid_w,   mid_h  };              // 6. Bottom-Left Mid-Wall
    corners[7] = center + sf::Vector2f{ -half_w,  0.0f   };              // 7. Left Lateral Apex
    corners[8] = center + sf::Vector2f{ -mid_w,  -mid_h  };              // 8. Top-Left Mid-Wall

    // Original guide landmarks for interpolation path
    sf::Vector2f tail_landmarks[] = {
        corners[5],       
        {300.0f, 205.0f - 20.f}, 
        {300.0f, 235.0f - 20.f}, 
        {300.0f, 265.0f - 20.f}, 
        {300.0f, 295.0f - 20.f}, 
        {300.0f, 325.0f - 20.f}  
    };

    
    float tail_radii[] = {1.6f, 1.4f, 1.2f, 1.2f, 1.4f, 1.6f};
    // float tail_radii[] = {1.2f, 1.6f, 1.4f, 1.2f, 1.0f, 0.8f};

    std::vector<sf::Vector2f> positions;
    std::vector<float> mass;
    std::vector<float> radius;

    float head_part_radius = 0.7f;

    // --- 2. GENERATE DENSE HEAD SHELL ---
    positions.push_back(corners[0]);
    radius.push_back(0.75f);
    mass.push_back(head_heaviness * b_const * (pow(1.5f, 3))); 

    int apex_indices[9];
    const int HEAD_SUBDIVISIONS = 2; 

    for(int i = 1; i <= 8; i++) {
        apex_indices[i] = positions.size(); 
        int next_corner_idx = (i == 8) ? 1 : i + 1;
        
        sf::Vector2f pA = corners[i];
        sf::Vector2f pB = corners[next_corner_idx];

        for(int s = 0; s < HEAD_SUBDIVISIONS; s++) {
            float t = (float)s / (float)HEAD_SUBDIVISIONS;
            positions.push_back(pA + t * (pB - pA));
            radius.push_back(head_part_radius);
            mass.push_back(head_heaviness * b_const * (pow(0.7f, 3))); // Preserved 1.2f factor
            // mass.push_back(head_heaviness * b_const * (pow(head_part_radius, 3))); // Preserved 1.2f factor
        }
    }

    // --- 3. GENERATE SLIM ULTRA-THIN DOUBLE-STRAND TAIL ---
    std::vector<int> left_tail_indices;
    std::vector<int> right_tail_indices;
    int sperm_center_index;
    // const int TAIL_SUBDIVISIONS = 4; 
    std::vector<int> tail_subdivs{6, 6, 6, 6, 6};
    // std::vector<int> tail_subdivs{6, 6, 6, 6, 8};

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

            if(i == 2 && s == 4){
                sperm_center_index = positions.size() + id_offset;
            }

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

        // --- 2. GENERATE DENSE HEAD SHELL --- OTHER END

    vertical_length = 13.0f; // Change this to dynamically adjust the height

    // Proportional width scale derived from your original profile ratio (30.0w / 52.0h)
    horizontal_length = vertical_length; 

    half_h = vertical_length / 2.0f;
    half_w = horizontal_length / 2.0f;
    mid_h  = half_h / 2.0f;
    mid_w  = half_w / 2.0f;
    
    sf::Vector2f downward_shift = {0, 155};
    sf::Vector2f b_center = {300.0f, 150.0f + downward_shift.y};

    // --- CALCULATED COORDINATES ---
    b_corners[0] = b_center;                                                 // Head Center Core
    b_corners[1] = b_center + sf::Vector2f{ 0.0f,   -half_h };               // 1. Top Apex
    b_corners[2] = b_center + sf::Vector2f{  mid_w,  -mid_h  };              // 2. Top-Right Mid-Wall
    b_corners[3] = b_center + sf::Vector2f{  half_w,  0.0f   };              // 3. Right Lateral Apex
    b_corners[4] = b_center + sf::Vector2f{  mid_w,   mid_h  };              // 4. Bottom-Right Mid-Wall
    b_corners[5] = b_center + sf::Vector2f{  0.0f,    half_h };               // 5. Bottom Apex (Tail Base)
    b_corners[6] = b_center + sf::Vector2f{ -mid_w,   mid_h  };              // 6. Bottom-Left Mid-Wall
    b_corners[7] = b_center + sf::Vector2f{ -half_w,  0.0f   };              // 7. Left Lateral Apex
    b_corners[8] = b_center + sf::Vector2f{ -mid_w,  -mid_h  };              // 8. Top-Left Mid-Wall

    positions.push_back(b_corners[0]);
    radius.push_back(0.75f);
    mass.push_back(b_const * (pow(1.5f, 3))); 

    int apex_indices_bottom[9];
    const int B_HEAD_SUBDIVISIONS = 2; 

    for(int i = 1; i <= 8; i++) {
        apex_indices_bottom[i] = positions.size(); 
        int next_corner_idx = (i == 8) ? 1 : i + 1;
        
        sf::Vector2f pA = b_corners[i];
        sf::Vector2f pB = b_corners[next_corner_idx];

        for(int s = 0; s < B_HEAD_SUBDIVISIONS; s++) {
            float t = (float)s / (float)B_HEAD_SUBDIVISIONS;
            positions.push_back(pA + t * (pB - pA));
            radius.push_back(0.25f);
            mass.push_back(b_const * (pow(0.7f, 3))); // Preserved 1.2f factor
            // mass.push_back(head_heaviness * b_const * (pow(head_part_radius, 3))); // Preserved 1.2f factor
        }
    }

    assert(sperm_length >= b_center.y - center.y + (head_vertical_length + vertical_length)/2);

    // positions.push_back(tail_landmarks[5] + sf::Vector2f{0, 8});
    // radius.push_back(5.f);
    // mass.push_back(1 * b_const * (radius.back() * radius.back() * radius.back()));

    // --- 4. INSTANTIATE PARTICLES IN ENGINE ---
    std::vector<int> sensing_points;
    // THIS DOES NOT CHANGE RADIUS OF PARTICLES JUST THE DISTANCES
    for(int i = 0; i < (int)positions.size(); i++){
        old_pos = positions[i] + shift;
        vel.x = 0.0f * i; vel.y = 0.0f * i; 
        // curr_pos.x = vel.x * dt + old_pos.x; curr_pos.y = vel.y * dt + old_pos.y;
        acc = {0.0f, 0.0f};

        
        particles.push_back(Particle(env_id, i + id_offset, radius[i], mass[i], old_pos, vel, structure::creature));
        // particles[particles.size()-1].set_acc(acc);

        if(positions[i] == corners[1] or positions[i] == corners[3]
                or positions[i] == corners[7] or i == (int)positions.size()-1){
            
            // we need the global index here
            sensing_points.push_back(particles.size()-1);
        }
    }

    // --- 5. CONSTRUCT CONNECTION NETWORK (SPRINGS) ---
    // Preserved stiffness properties completely unchanged
    // const float RIGID_TENDON   = 1e6f;  
    // const float FLEXIBLE_SPINE = 1e4f;  
    // const float ACTUATOR       = 1e4f;
    const float RIGID_TENDON   = 1e6f;  // Keep for the head structure
    const float DIAGONAL_BRACE = 1e3f;  // Weak enough to bend, strong enough to hold shape

    const float FLEXIBLE_SPINE = 2e4f;  // Lower passive resistance
    const float ACTUATOR       = 1e4f;  // 10x stronger than the spine/braces!  

    auto add_spring = [&](int idxA, int idxB, float stiffness, bool outside_body) {
        float dx = positions[idxA].x - positions[idxB].x;
        float dy = positions[idxA].y - positions[idxB].y;
        float exact_length = std::sqrt(dx * dx + dy * dy);
        assert(exact_length != 0);
        springs.push_back(Spring(particles[idxA+id_offset], particles[idxB+id_offset], env_id, spring_id, exact_length, stiffness, outside_body));
        spring_id++;
    };

    // B. ANCHOR TAIL ROOTS TO THE BASE OF THE HEAD
    int head_left_base = apex_indices[6] + 2;  
    int head_right_base = apex_indices[4] - 2; 
    int head_left_bottom = apex_indices[6];  
    int head_right_bottom = apex_indices[4]; 
    int bottom_tip = apex_indices[5]; 

    // A. SOLID HEAD SPOKES & OUTER RING
    for(int i = 1; i < left_tail_indices[0]; i++) {
        add_spring(0, i, RIGID_TENDON, false);
        int next_idx = (i == left_tail_indices[0] - 1) ? 1 : i + 1;
        if(i >= head_right_base && i < head_left_base){
            add_spring(i, next_idx, RIGID_TENDON, false);
        }
        else{
            add_spring(i, next_idx, RIGID_TENDON, true);
        }
    }
    
    add_spring(left_tail_indices[0], head_left_base, RIGID_TENDON, true);
    add_spring(head_right_base, right_tail_indices[0], RIGID_TENDON, true);
    
    add_spring(head_left_bottom, left_tail_indices[1], RIGID_TENDON, false);
    add_spring(head_right_bottom, right_tail_indices[1], RIGID_TENDON, false);
    add_spring(bottom_tip, left_tail_indices[0], RIGID_TENDON, false);
    add_spring(bottom_tip, right_tail_indices[0], RIGID_TENDON, false);
    
    add_spring(apex_indices[7], apex_indices[3], RIGID_TENDON, false);
    add_spring(apex_indices[1], apex_indices[5], RIGID_TENDON, false);
    
    // B. ANCHOR TAIL ROOTS TO THE BASE OF THE TAIL HEAD
    int b_head_left_base = apex_indices_bottom[8];  
    int b_head_right_base = apex_indices_bottom[2]; 
    int b_head_left_bottom = apex_indices_bottom[6] + 2;  
    int b_head_right_bottom = apex_indices_bottom[4] - 2; 
    int b_top_tip = apex_indices_bottom[1]; 

    // A. SOLID HEAD SPOKES & OUTER RING
    for(int i = b_top_tip; i < (int)positions.size(); i++) {
        add_spring(b_top_tip-1, i, RIGID_TENDON, false);
        int next_idx = (i == (int)positions.size() - 1) ? b_top_tip : i + 1;
        if(i >= b_head_left_base && i < b_head_right_base){
            add_spring(i, next_idx, RIGID_TENDON, false);
        }
        else{
            add_spring(i, next_idx, RIGID_TENDON, true);
        }
    }
    
    add_spring(b_head_left_base, left_tail_indices[left_tail_indices.size()-2], RIGID_TENDON, true);
    add_spring(b_head_left_bottom, left_tail_indices.back(), RIGID_TENDON, false);
    add_spring(b_top_tip, left_tail_indices.back(), RIGID_TENDON, false);

    add_spring(right_tail_indices[right_tail_indices.size()-2], b_head_right_base, RIGID_TENDON, true);
    add_spring(right_tail_indices.back(), b_head_right_bottom, RIGID_TENDON, false);
    add_spring(right_tail_indices.back(), b_top_tip, RIGID_TENDON, false);


    // C. INTERNAL AXONEME TRUSS & DISTRIBUTED COMPLIANT MUSCLE/TENDON SYSTEM
    int num_tail_segments = left_tail_indices.size();
    for(int i = 0; i < num_tail_segments; i++) {
        
        float dist = abs((float)i - (float)num_tail_segments/2);
        float scaling = 500.f * fmax(dist/((float)num_tail_segments/2), 0.2f);
        // 1. Horizontal cross-rungs (Rigidly holds the thin micro-ladder shape profile)
        add_spring(left_tail_indices[i], right_tail_indices[i], DIAGONAL_BRACE * 2.0f * scaling, false);

        if(i < num_tail_segments - 1) {
            if(i + 3 < num_tail_segments) {
                // 2. Shear diagonals (Prevents the thin strands from buckling or overlapping)
                add_spring(left_tail_indices[i], right_tail_indices[i + 3], DIAGONAL_BRACE * scaling, false);
                add_spring(right_tail_indices[i], left_tail_indices[i + 3], DIAGONAL_BRACE * scaling, false);
            }

            // bool is_active_muscle = (i == 0 || i == 7 || i == 14 || i == 21);

            // float scaling = (1-pow(float(i)/(num_tail_segments-1), 2))*100 + 1;

            if (i % 2 == 0) {
                // add_muscle(right_tail_indices[i], right_tail_indices[i + 1], ACTUATOR * scaling, true);
                // add_muscle(left_tail_indices[i + 1], left_tail_indices[i], ACTUATOR * scaling, true);
            } else {
                add_spring(right_tail_indices[i], right_tail_indices[i + 1], FLEXIBLE_SPINE * scaling, true);
                add_spring(left_tail_indices[i + 1], left_tail_indices[i], FLEXIBLE_SPINE * scaling, true);
            }
        }
    }

    auto add_muscle = [&](int idxA, int idxB, float stiffness, bool outside_body) {
        float dx = positions[idxA].x - positions[idxB].x;
        float dy = positions[idxA].y - positions[idxB].y;
        float exact_length = std::sqrt(dx * dx + dy * dy);
        assert(exact_length != 0);
        muscles.push_back(Muscle(particles[idxA+id_offset], particles[idxB+id_offset], env_id, spring_id, exact_length, stiffness, outside_body));
        spring_id++;
    };
    
    for(int i = 0; i < num_tail_segments; i++) {
        
        if(i < num_tail_segments - 1) {
            // bool is_active_muscle = (i == 0 || i == 7 || i == 14 || i == 21);

            // float dist = abs((float)i - (float)num_tail_segments/2);
            float scaling = 100.f;
            // float scaling = 200.f * fmax(dist/((float)num_tail_segments/2), 0.5f);
            // float scaling = (1-pow(float(i)/(num_tail_segments-1), 2))*100 + 1;

            if (i % 2 == 0) {
                add_muscle(right_tail_indices[i], right_tail_indices[i + 1], ACTUATOR * scaling, true);
                add_muscle(left_tail_indices[i + 1], left_tail_indices[i], ACTUATOR * scaling, true);
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

    Creature_data data{ muscle_indices, sensing_points, sperm_center_index};

    return data;
}
*/

int create_football(int& num_particles, std::vector<Particle>& particles, int& num_springs, std::vector<Spring>& springs, int env_id) {

    const float ball_float_sink_factor = 1.f; 

    // --- 2. DIMENSIONS & STRUCTURAL PARAMETERS ---
    const sf::Vector2f ball_center = {0.0f, 0.0f}; // Placed perfectly within reach of the sperm
    const float BALL_RADIUS = 20.0f;                  // Scaled size relative to the sperm head
    const int EDGE_POINTS = 40;                       // Evenly distributed points around the rim
    const float GAP_DISTANCE = 0.15f;                  // Explicit tiny gap (in pixels) between edge circles
    const float CENTER_PARTICLE_RADIUS = 1.0f;

    assert(ball_radius == BALL_RADIUS);

    // Calculate the precise chord distance between adjacent points along the circle
    float chord_length = 2.0f * BALL_RADIUS * std::sin(3.14159265f / (float)EDGE_POINTS);
    
    // Automatically derive the radius to guarantee they don't overlap and preserve the gap
    const float EDGE_PARTICLE_RADIUS = (chord_length - GAP_DISTANCE) / 2.0f;

    // Grab the global buoyancy constant defined in your engine
    // const float b_const = buoyancy_const;
    const float b_const = 4.f/3.f * 3.141f * 14.8f;
    assert(buoyancy_const == 0);
    

    // Track starting index offset if particles already exist in the vectors
    int start_particle_idx = particles.size();
    int spring_id = springs.size();

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

        int global_id = start_particle_idx + i; 
        particles.push_back(Particle(env_id, global_id, radius[i], mass[i], old_pos, vel, structure::ball));
        // particles.back().set_acc(acc);
    }

    // --- 6. WEAVE THE SPRING CONSTANT NETWORK ---
    const float BALL_SKIN_STIFFNESS  = 5e5f; // Perimeter structural strength
    const float BALL_SPOKE_STIFFNESS = 3e4f; // Internal pressure preservation

    auto add_ball_spring = [&](int idxA, int idxB, float stiffness, bool outside) {
        float dx = positions[idxA - start_particle_idx].x - positions[idxB - start_particle_idx].x;
        float dy = positions[idxA - start_particle_idx].y - positions[idxB - start_particle_idx].y;
        float exact_length = std::sqrt(dx * dx + dy * dy);
        springs.push_back(Spring(particles[idxA], particles[idxB], env_id, spring_id, exact_length, stiffness, outside));
        spring_id++;
    };

    for (int i = 0; i < EDGE_POINTS; i++) {
        int current_edge_idx = first_edge_index + i;
        int next_edge_idx = first_edge_index + ((i + 1) % EDGE_POINTS);

        // A. Structural Rim Wheel (Forms the circular skin of the football)
        add_ball_spring(current_edge_idx, next_edge_idx, BALL_SKIN_STIFFNESS, true);

        // B. Central Pressure Spokes (Prevents the football from collapsing inward)
        add_ball_spring(center_node_index, current_edge_idx, BALL_SPOKE_STIFFNESS, false);

        // C. Cross-Diametric Bracing (Maintains perfect roundness during high impact)
        int opposite_edge_idx = first_edge_index + ((i + (EDGE_POINTS / 2)) % EDGE_POINTS);
        if (i < EDGE_POINTS / 2) { 
            add_ball_spring(current_edge_idx, opposite_edge_idx, BALL_SPOKE_STIFFNESS * 0.5f, false);
        }
    }

    // Update global reference numbers for your game loop renderer
    num_particles = particles.size();
    num_springs = springs.size();

    return center_index;
}

/*

void create_self_aligning_arrow(int& num_particles, std::vector<Particle>& particles, int& num_springs, std::vector<Spring>& springs, float dt) {
    particles.reserve(particles.size() + 4);
    springs.reserve(springs.size() + 10);

    const float RADIUS = 5.0f;
    const float MASS = 50.0f;
    const float STIFFNESS = 2e4f;
    
    const sf::Vector2f center = { 1600.0f, 300.0f };
    const sf::Vector2f velocity = { 0.0f, 0.0f }; 

    int start_idx = particles.size();

    // Spawn pointing 30 degrees offset so we can watch it self-correct
    float angle_offset = 45.0f * (3.14159265f / 180.0f); 

    auto rotate_vector = [&](sf::Vector2f vec, float rad) -> sf::Vector2f {
        return { vec.x * std::cos(rad) - vec.y * std::sin(rad), vec.x * std::sin(rad) + vec.y * std::cos(rad) };
    };

    // Unrotated base coordinates
    sf::Vector2f tip_local   = {  60.0f,   0.0f };  // Sharp point
    sf::Vector2f baseR_local = rotate_vector(tip_local, 2*3.141f/3.f);  // Lower tail
    // sf::Vector2f baseR_local = { -40.0f,  40.0f };  // Lower tail
    sf::Vector2f baseL_local = rotate_vector(tip_local, 4*3.141f/3.f);  // Upper tail
    // sf::Vector2f baseL_local = { -40.0f, -40.0f };  // Upper tail

    sf::Vector2f p_tip   = center + rotate_vector(tip_local, angle_offset);
    sf::Vector2f p_baseR = center + rotate_vector(baseR_local, angle_offset);
    sf::Vector2f p_baseL = center + rotate_vector(baseL_local, angle_offset);

    particles.push_back(Particle(particles.size() + 1, RADIUS, MASS, p_tip,   velocity, structure::cluster)); // start_idx + 0
    particles.push_back(Particle(particles.size() + 1, RADIUS, MASS, p_baseR, velocity, structure::cluster)); // start_idx + 1
    particles.push_back(Particle(particles.size() + 1, RADIUS, MASS, p_baseL, velocity, structure::cluster)); // start_idx + 2
    
    // Central balance node
    // particles.push_back(Particle(particles.size() + 1, RADIUS, MASS/10, center,  velocity, structure::cluster)); // start_idx + 3

    // --- 2. SPRINGS WEAVING ---
    auto add_spring = [&](int idxA, int idxB, bool is_outer) {
        sf::Vector2f posA = particles[idxA].get_curr_pos();
        sf::Vector2f posB = particles[idxB].get_curr_pos();
        float length = std::sqrt((posA.x - posB.x)*(posA.x - posB.x) + (posA.y - posB.y)*(posA.y - posB.y));
        springs.push_back(Spring(particles[idxA], particles[idxB], length, STIFFNESS, is_outer));
    };

    // Outer Perimeter (Strictly CLOCKWISE: Outside is on the Left of the vector p1 -> p2)
    add_spring(start_idx + 0, start_idx + 1, true);  // From Tip down to Lower Rear Wing
    add_spring(start_idx + 1, start_idx + 2, true);  // From Lower Rear Wing straight up to Upper Rear Wing
    add_spring(start_idx + 2, start_idx + 0, true);  // From Upper Rear Wing back down to the Tip

    // // Structural interior
    // add_spring(start_idx + 3, start_idx + 0, false);
    // add_spring(start_idx + 3, start_idx + 1, false);
    // add_spring(start_idx + 3, start_idx + 2, false);

    num_particles = particles.size();
    num_springs = springs.size();
}

void create_high_drag_block(int& num_particles, std::vector<Particle>& particles, int& num_springs, std::vector<Spring>& springs, float dt) {
    // particles.reserve(particles.size() + 5);
    // springs.reserve(springs.size() + 10);

    const float RADIUS = 15.0f;
    const float MASS = 500.0f;
    const float STIFFNESS = 2e4f;
    const sf::Vector2f center = { 300.0f, 300.0f };
    const sf::Vector2f velocity = { 2.0f, 10.0f }; 
    const float half_side = 45.0f;

    int start_idx = particles.size();

    // --- 1. PARTICLES ---
    // Corner positions mapped out
    sf::Vector2f p_topLeft     = center + sf::Vector2f{ -half_side, -half_side }; 
    sf::Vector2f p_topRight    = center + sf::Vector2f{  half_side, -half_side }; 
    sf::Vector2f p_bottomRight = center + sf::Vector2f{  half_side,  half_side }; 
    sf::Vector2f p_bottomLeft  = center + sf::Vector2f{ -half_side,  half_side }; 

    particles.push_back(Particle(particles.size() + 1, RADIUS, MASS, p_topLeft,     -velocity*20.f , structure::cluster)); // start_idx + 0
    particles.push_back(Particle(particles.size() + 1, RADIUS, MASS, p_topRight,    velocity, structure::cluster)); // start_idx + 1
    particles.push_back(Particle(particles.size() + 1, RADIUS, MASS, p_bottomRight, velocity, structure::cluster)); // start_idx + 2
    particles.push_back(Particle(particles.size() + 1, RADIUS, MASS, p_bottomLeft,  velocity, structure::cluster)); // start_idx + 3
    
    // Center node
    particles.push_back(Particle(particles.size() + 1, RADIUS, MASS, center, velocity, structure::cluster));        // start_idx + 4

    // --- 2. SPRINGS WEAVING ---
    auto add_spring = [&](int idxA, int idxB, bool is_outer) {
        sf::Vector2f posA = particles[idxA].get_curr_pos();
        sf::Vector2f posB = particles[idxB].get_curr_pos();
        float length = std::sqrt((posA.x - posB.x)*(posA.x - posB.x) + (posA.y - posB.y)*(posA.y - posB.y));
        springs.push_back(Spring(particles[idxA], particles[idxB], length, STIFFNESS, is_outer));
    };

    // Outer Perimeter (Strictly CLOCKWISE: Outside environment is now on the left)
    add_spring(start_idx + 0, start_idx + 1, true);  // Top Face (Left to Right)
    add_spring(start_idx + 1, start_idx + 2, true);  // Front/Right Face (Top to Bottom)
    add_spring(start_idx + 2, start_idx + 3, true);  // Bottom Face (Right to Left)
    add_spring(start_idx + 3, start_idx + 0, true);  // Back/Left Face (Bottom to Top)

    // Internal structural core (is_outer = false)
    add_spring(start_idx + 4, start_idx + 0, false);
    add_spring(start_idx + 4, start_idx + 1, false);
    add_spring(start_idx + 4, start_idx + 2, false);
    add_spring(start_idx + 4, start_idx + 3, false);
    add_spring(start_idx + 0, start_idx + 2, false); 
    add_spring(start_idx + 1, start_idx + 3, false); 

    num_particles = particles.size();
    num_springs = springs.size();
}

void create_collision_test_rig(int& num_particles, std::vector<Particle>& particles, int& num_springs, std::vector<Spring>& springs, float dt) {
    // --- 1. PREVENT POINTER/REFERENCE DRAG (CRITICAL FOR YOUR ENGINE) ---
    // Total particles: 7 (Left Cluster) + 7 (Right Cluster) = 14 particles total.
    // Pre-reserving memory ensures that std::vector never reallocates and invalidates Spring references.
    // particles.reserve(30);
    // springs.reserve(100);

    // --- 2. CONFIGURATION PARAMETERS ---
    const float TEST_RADIUS = 15.0f;       // Large radius as requested
    const float TEST_MASS = 500.0f;        // Heavy mass to minimize viscosity damping impact
    const float CLUSTER_RADIUS = 50.0f;    // Spatial spacing radius from the local center
    const int RIM_POINTS = 6;              // Hexagonal outer rim layout (6 outer + 1 center = 7 particles per cluster)
    const float HIGH_SPEED = 20.0f;       // Aggressive initial velocity for collision testing

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

        auto add_test_spring = [&](int idxA, int idxB, bool out) {
            sf::Vector2f pA = particles[idxA].get_curr_pos();
            sf::Vector2f pB = particles[idxB].get_curr_pos();
            float exact_length = std::sqrt((pA.x - pB.x)*(pA.x - pB.x) + (pA.y - pB.y)*(pA.y - pB.y));
            springs.push_back(Spring(particles[idxA], particles[idxB], exact_length, TEST_STIFFNESS, out));
        };

        for (int i = 0; i < RIM_POINTS; i+=2) {
            int current_rim = first_rim_idx + i;
            int next_rim = first_rim_idx + ((i + 1) % RIM_POINTS);

            // Rim Structural Edge
            add_test_spring(current_rim, next_rim, true);

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

*/