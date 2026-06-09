#include "particle.hpp"
#include <algorithm>

// params
float max_y = 1080;
float max_x = 1920;

float restitution = 0.6;

Particle::Particle(){
    m_radius = 10;
    m_mass = 10;
    m_body_shape.setRadius(m_radius);
    m_body_shape.setFillColor(sf::Color::White);
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

float Particle::get_mass() const{
    return m_mass;
}

void Particle::set_acc(sf::Vector2<float> acc){
    m_acc = acc;
}

void Particle::step(float dt){

    sf::Vector2f new_pos;
    new_pos = 2.f*m_curr_pos - m_old_pos + m_acc*dt*dt;

    if(new_pos.x + m_radius > max_x)
        bounce(max_x, 0, new_pos);
    else if(new_pos.x + m_radius < 0)
        bounce(0, 0, new_pos);
    else{
        m_old_pos.x = m_curr_pos.x;
        m_curr_pos.x = new_pos.x;
    }

    if(new_pos.y + m_radius > max_y)
        bounce(max_y, 1, new_pos);
    else if(new_pos.y + m_radius < 0)
        bounce(0, 1, new_pos);
    else{
        m_old_pos.y = m_curr_pos.y;
        m_curr_pos.y = new_pos.y;
    }

}

void Particle::bounce(float wall, int axis, sf::Vector2f new_pos){
    if(axis == 0){
        new_pos.x = 2*wall - (new_pos.x + m_radius) - m_radius;
        
        m_old_pos.x = 2*wall - (m_curr_pos.x + m_radius) - m_radius;
        m_curr_pos.x = new_pos.x;
    }
    else if(axis == 1){
        new_pos.y = 2*wall - (new_pos.y + m_radius) - m_radius;
        
        m_old_pos.y = 2*wall - (m_curr_pos.y + m_radius) - m_radius;
        m_curr_pos.y = new_pos.y;
    }
}

// function will sort particles inplace according to x coord -- particles list is permuted
void handle_all_collisions(std::vector<Particle>& particles){

    std::sort(particles.begin(), particles.end(), 
    [](Particle const& p1, Particle const& p2)->bool{
        return (p1.get_curr_pos()).x < (p2.get_curr_pos()).x;
    });

    int num_particles = particles.size();
    for(int i=0; i < num_particles; i++){
        
        Particle& p1 = particles[i];
        for(int j=i+1; j < num_particles; j++){
            
            Particle& p2 = particles[j];

            if(p2.get_curr_pos().x - p2.get_radius() > p1.get_curr_pos().x + p1.get_radius())
                break;

            handle_two_body_collision(p1, p2);
        }

    }
}

sf::CircleShape& Particle::get_shape(){
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
    
    float c1 = (m1*restitution-m2)/(m1+m2);
    float c2 = (1+restitution)*(m1/(m1+m2));
    sf::Vector2f p1_vel_after_collision = (p1_vel_before_collision - p1_vel_along_loc) + c1*p1_vel_along_loc + c2*p2_vel_along_loc;
    
    c1 = (m2*restitution-m1)/(m1+m2);
    c2 = (1+restitution)*(m2/(m1+m2));
    sf::Vector2f p2_vel_after_collision = (p2_vel_before_collision - p2_vel_along_loc) + c1*p2_vel_along_loc + c2*p1_vel_along_loc;

    sf::Vector2f p1_new_pos = p1.get_curr_pos() - (overlap/dist*0.5f)*line_of_contact;
    sf::Vector2f p2_new_pos = p2.get_curr_pos() + (overlap/dist*0.5f)*line_of_contact;

    p1.set_pos(p1_new_pos - p1_vel_after_collision, p1_new_pos);
    p2.set_pos(p2_new_pos - p2_vel_after_collision, p2_new_pos);
}