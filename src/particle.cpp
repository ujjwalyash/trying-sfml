#include "headers/particle.hpp"
#include <SFML/System/Vector2.hpp>
#include <algorithm>
#include <cmath>
#include <vector>

// params
float max_y = 1080;
float max_x = 1920;

sf::Vector2f gravity = {6, 8};
float buoyancy_const = 4.f/3 * 3.141 * 15; // DO NOT MAKE DENSITY 1000

float restitution = 0.8;
float coefficient_friction = 0;
float viscosity = 0.02 * 20; // *10 bc viscosity is only applied at point masses which have small radius so we scale it

Particle::Particle(int id, float radius, float mass)
    :m_id(id),
     m_spring_acc({0, 0}),
     m_radius(radius),
     m_cube_radius(radius*radius*radius),
     m_mass(mass),
     m_sqrt_mass(sqrt(mass))
{
    m_body_shape.setRadius(m_radius); 
    m_body_shape.setOrigin({m_radius, m_radius}); 
    m_body_shape.setFillColor(sf::Color::White);

    m_buoyancy_acc = -buoyancy_const*(m_cube_radius)*gravity/m_mass;
}

void Particle::set_pos(sf::Vector2<float> old_pos, sf::Vector2<float> curr_pos){
    m_old_pos = old_pos;
    m_curr_pos = curr_pos;
}

sf::Vector2<float> Particle::get_curr_pos() const{
    return m_curr_pos;
}

sf::Vector2<float> Particle::get_old_pos() const{
    return m_old_pos;
}

float Particle::get_radius() const{
    return m_radius;
}

float Particle::get_mass() const{ return m_mass; }
float Particle::get_sqrt_mass() const{ return m_sqrt_mass; }

int Particle::get_id() const{
    return m_id;
}

void Particle::set_acc(sf::Vector2<float> acc){
    m_acc = acc;
}

float Particle::calculate_total_energy(float dt){
    float vel = ((m_curr_pos-m_old_pos)/dt).length();
    return 0.5f*m_mass*(vel*vel) 
        + m_mass*(m_acc.y+gravity.y)*(max_y-m_curr_pos.y) + m_mass*(m_acc.x+gravity.x)*(max_y-m_curr_pos.x);
}

void Particle::add_spring_acc(sf::Vector2f spring_acc){
    m_spring_acc += spring_acc;
}

void Particle::step(float dt){

    float damping_factor = fmax(0.f, 1-(viscosity*m_radius*dt)/(m_mass));
    
    sf::Vector2f new_pos;
    sf::Vector2f net_acc = m_acc + gravity + m_buoyancy_acc + m_spring_acc;
    new_pos = m_curr_pos + damping_factor*(m_curr_pos-m_old_pos) + net_acc*dt*dt;
    m_spring_acc.x = 0; m_spring_acc.y = 0;

    m_old_pos = m_curr_pos;
    m_curr_pos = new_pos;

    handle_boundary();
}

void Particle::handle_boundary(){
    if(m_curr_pos.x + m_radius > max_x)
        reflect(max_x, 0, 1);
    else if(m_curr_pos.x - m_radius < 0)
        reflect(0, 0, -1);
    
    if(m_curr_pos.y + m_radius > max_y)
        reflect(max_y, 1, 1);
    else if(m_curr_pos.y - m_radius < 0)
        reflect(0, 1, -1);
}

// sign indicates weather we need to add/subtract radius to get offending point on the circle
void Particle::reflect(float wall, int axis, int sign){
    float signed_radius = sign*m_radius;
    if(axis == 0){
        m_curr_pos.x = 2*wall - (m_curr_pos.x + signed_radius) - signed_radius;        
        m_old_pos.x = 2*wall - (m_old_pos.x + signed_radius) - signed_radius;
    }
    else if(axis == 1){
        m_curr_pos.y = 2*wall - (m_curr_pos.y + signed_radius) - signed_radius;
        m_old_pos.y = 2*wall - (m_old_pos.y + signed_radius) - signed_radius;
    }
}

// function will sort particles inplace according to x coord -- particles list is permuted
// DO NOT SORT, since stl sort swaps elements the reference to these variables will be ruined
// eg if a spring hold refernces to 2 points, then after sorting the swaps make now change the underlying points which are refernced
void handle_all_collisions(std::vector<Particle>& particles){

    // we can have arrays of values ie a single array for x coords of all particles and then just sort the indicies and work

    std::vector<Particle*> sorted_particles;
    // if i dont use reference here i would just be storing the temp addresses of the copies made by for loop
    for(Particle &p: particles){
        sorted_particles.push_back(&p);
    }

    // do we really need to sort the whole thing ?? not much would have changed in a single time step -- very few inversions present
    std::sort(sorted_particles.begin(), sorted_particles.end(), 
    [](Particle *p1, Particle *p2)->bool{
        return ((*p1).get_curr_pos()).x < ((*p2).get_curr_pos()).x;
    });

    int num_particles = sorted_particles.size();
    for(int i=0; i < num_particles; i++){
        
        Particle& p1 = *sorted_particles[i];
        for(int j=i+1; j < num_particles; j++){
            
            Particle& p2 = *sorted_particles[j];

            if(p2.get_curr_pos().x - p2.get_radius() > p1.get_curr_pos().x + p1.get_radius())
                break;

            // PASSING REFS TO COPY OF p1 and p2 wont affect the actual objects
            handle_two_body_collision(p1, p2);
        }

    }
}

const sf::CircleShape& Particle::get_shape(){
    m_body_shape.setPosition(m_curr_pos);
    return m_body_shape;
}

void handle_two_body_collision(Particle& p1, Particle& p2){

    sf::Vector2f line_of_contact = p2.get_curr_pos() - p1.get_curr_pos();
    float r1 = p1.get_radius(); float r2 = p2.get_radius();
    float m1 = p1.get_mass();   float m2 = p2.get_mass();

    float dist = line_of_contact.length();
    float overlap = (r1+r2)-dist;
    if(overlap <= 0)
        return;

    sf::Vector2f p1_vel_before_collision = p1.get_curr_pos() - p1.get_old_pos();
    sf::Vector2f p2_vel_before_collision = p2.get_curr_pos() - p2.get_old_pos();
    
    sf::Vector2f p1_vel_along_loc = p1_vel_before_collision.projectedOnto(line_of_contact);
    sf::Vector2f p2_vel_along_loc = p2_vel_before_collision.projectedOnto(-line_of_contact);
    
    sf::Vector2f p1_vel_perp_loc = p1_vel_before_collision - p1_vel_along_loc; float p1_vel_perp_loc_length = p1_vel_perp_loc.length();
    sf::Vector2f p2_vel_perp_loc = p2_vel_before_collision - p2_vel_along_loc; float p2_vel_perp_loc_length = p2_vel_perp_loc.length();
    
    float c1 = (m1-m2*restitution)/(m1+m2);
    float c2 = (1+restitution)*(m2/(m1+m2));
    sf::Vector2f impulse_along_loc = (c1*p1_vel_along_loc + c2*p2_vel_along_loc) - p1_vel_along_loc;

    float magnitude_impulse_friction = fmin(coefficient_friction*impulse_along_loc.length(), p1_vel_perp_loc_length);
    sf::Vector2f impulse_perp_loc = -magnitude_impulse_friction*p1_vel_perp_loc/(p1_vel_perp_loc_length != 0 ? p1_vel_perp_loc_length  : 1.f);
    sf::Vector2f p1_vel_after_collision = p1_vel_before_collision + impulse_along_loc + impulse_perp_loc;
    
    c1 = (1+restitution)*(m1/(m1+m2));
    c2 = (m2-m1*restitution)/(m1+m2);
    impulse_along_loc = (c1*p1_vel_along_loc + c2*p2_vel_along_loc) - p2_vel_along_loc;

    magnitude_impulse_friction = fmin(coefficient_friction*impulse_along_loc.length(), p2_vel_perp_loc.length());
    impulse_perp_loc = -magnitude_impulse_friction*p2_vel_perp_loc/(p2_vel_perp_loc_length != 0 ? p2_vel_perp_loc_length  : 1.f);
    sf::Vector2f p2_vel_after_collision = p2_vel_before_collision + impulse_along_loc + impulse_perp_loc;

    sf::Vector2f p1_new_pos = p1.get_curr_pos() - (overlap/dist*0.5f)*line_of_contact;
    sf::Vector2f p2_new_pos = p2.get_curr_pos() + (overlap/dist*0.5f)*line_of_contact;

    p1.set_pos(p1_new_pos - p1_vel_after_collision, p1_new_pos);
    p2.set_pos(p2_new_pos - p2_vel_after_collision, p2_new_pos);
}