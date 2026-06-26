#include "headers/particle.hpp"
#include <SFML/System/Vector2.hpp>
#include <cmath>

Particle::Particle(int id, float radius, float mass, sf::Vector2<float> curr_pos, sf::Vector2<float> vel, structure type)
    :m_id(id),
     m_original_vel(vel),
     m_original_curr_pos(curr_pos),
     m_vel(vel),
     m_curr_pos(curr_pos),
     m_spring_acc({0, 0}),
     m_radius(radius),
     m_cube_radius(radius*radius*radius),
     m_mass(mass),
     m_sqrt_mass(sqrt(mass)),
     m_damping_factor(fmin(1, (viscosity*m_radius*env_dt)/(m_mass))),
     m_type(type)
{
    m_buoyancy_acc = -buoyancy_const*(m_cube_radius)*gravity/m_mass;
}

void Particle::set_pos_vel(sf::Vector2<float> curr_pos, sf::Vector2<float> vel){
    m_curr_pos = curr_pos;
    m_vel = vel;
}

void Particle::shift_pos(sf::Vector2<float> shift){
    m_curr_pos += shift;
}

sf::Vector2<float> Particle::get_curr_pos() const{
    return m_curr_pos;
}

sf::Vector2<float> Particle::get_vel() const{
    return m_vel;
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

float Particle::calculate_total_energy(){
    float speed = m_vel.length();
    return 0.5f*m_mass*(speed*speed) 
        + m_mass*(m_acc.y+gravity.y)*(max_y-m_curr_pos.y) + m_mass*(m_acc.x+gravity.x)*(max_y-m_curr_pos.x);
}

void Particle::add_spring_acc(sf::Vector2f spring_acc){
    m_spring_acc += spring_acc;
}

float Particle::get_bounding_box_wall(direction dir){
    switch (dir) {
        
        case direction::left:
            return m_curr_pos.x - m_radius;
        
        case direction::right:
            return m_curr_pos.x + m_radius;
    }
}

void Particle::first_half_step(){

    // this is wasteful remove this -- done
    // float damping_factor = fmax(0.f, 1-(viscosity*m_radius*env_dt)/(m_mass));

    m_vel += + 0.5f * m_acc * env_dt;
    m_curr_pos += m_vel*env_dt;
}

void Particle::second_half_step(){
    
    m_acc = gravity + m_buoyancy_acc + m_spring_acc - m_damping_factor*m_vel;
    m_spring_acc.x = 0; m_spring_acc.y = 0;
    
    m_vel = m_vel + 0.5f * m_acc * env_dt;

    // ? is it needed to check this so often -- it has many if conditions -- jumps -- slow
    // regular checks imp since if particle goes to0 far inside the wall then when finally refelcted it may skip a particles which should have collided
    // ? but how regular
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
        m_vel.x = -m_vel.x;
    }
    else if(axis == 1){
        m_curr_pos.y = 2*wall - (m_curr_pos.y + signed_radius) - signed_radius;
        m_vel.y = -m_vel.y;
    }
}

void Particle::reset(){
    m_vel = m_original_vel;
    m_curr_pos = m_original_curr_pos;
    m_acc = {0, 0};
}