#include "headers/collision_handling.hpp"
#include <iostream>

float get_bounding_box_wall(direction dir, std::pair<int, int> object_info, std::vector<Particle>& particles, std::vector<Spring>& springs, std::vector<Muscle>& muscles){
    switch (object_info.first) {
        case 0:
            return particles[object_info.second].get_bounding_box_wall(dir);
        case 1:
            return springs[object_info.second].get_bounding_box_wall(dir);
        case 2:
            return muscles[object_info.second].get_bounding_box_wall(dir);

        default:
            exit(-1);
    }
}

// sorts combined_array by x coordinate of leftmost boundary of the bounding box
void sort(std::vector<std::pair<int, int>>& combined_array, std::vector<Particle>& particles, std::vector<Spring>& springs, std::vector<Muscle>& muscles){

    int n = combined_array.size();
    std::vector<int> left_walls(n, -1);
    for(int i = 0; i < n; i++){
        // TODO: many if conds checekd here instead add left wall to combined_array itself during the formation of combined_array
        left_walls[i] = get_bounding_box_wall(direction::left, combined_array[i], particles, springs, muscles);
    }

    std::sort(combined_array.begin(), combined_array.end(), 
        [&left_walls](auto a, auto b)->bool{
            return left_walls[a.second] < left_walls[b.second];
        }
    );

    // performs insertion sort -- since each time step is small not many inversions are generated 
    // for(int i = 1; i < n; i++){
        
    //     int j = i-1;
    //     float x_i = left_walls[i];
    //     float x_j = left_walls[j];
        
    //     while(x_i < x_j){
    //         std::swap(combined_array[j+1], combined_array[j]);
    //         std::swap(left_walls[j+1], left_walls[j]);
            
    //         j--;
    //         if(j < 0) break;
    //         x_j = left_walls[j];
    //     }
    // }
}

// returns false if no collision was there between BOUNDING BOX obj's -- actual collision may or maynot be there
bool collision_check_and_handle(std::pair<int, int> obj_1, std::pair<int, int> obj_2, std::vector<Particle>& particles, std::vector<Spring>& springs, std::vector<Muscle>& muscles){

    // bad dont hardcode things -- bugs if more types introduced
    constexpr int num_of_object_types = 3;

    switch (obj_1.first + num_of_object_types*obj_2.first) {    
        
        // particle, particle
        case (0 + num_of_object_types*0):
            // obj_2 is further right on x axis

            // boudning interval intersection check -- 
            if(particles[obj_2.second].get_curr_pos().x - particles[obj_2.second].get_radius() < particles[obj_1.second].get_curr_pos().x + particles[obj_1.second].get_radius()){
                
                // the fine check is done inside the func
                handle_particle_particle_collision(particles[obj_1.second], particles[obj_2.second]);
                
                // even if no actuall collision return true since the bounding interval did collide
                return true;
            }

            break;
        
        // // particle, spring
        // case (0 + num_of_object_types*1):
        
        //     if(particles[obj_2.first].get_curr_pos().x - particles[obj_2.first].get_radius() > particles[obj_1.first].get_curr_pos().x + particles[obj_1.first].get_radius()){
        //         handle_particle_particle_collision(particles[obj_1.first], particles[obj_2.first]);
        //     }
        
        // // spring, particle
        // // ! DUPLICATE CODE
        // case (1 + num_of_object_types*0):

        default:
    }

    return false;
}

// function will sort particles inplace according to x coord -- particles list is permuted
// DO NOT SORT, since stl sort swaps elements the reference to these particles will be ruined
// if a spring hold refernces to 2 points, then after sorting the swaps make now change the underlying points which are refernced
void handle_all_collisions(std::vector<Particle>& particles, std::vector<Spring>& springs, std::vector<Muscle>& muscles){

    std::vector<std::pair<int, int>> combined_array;
    
    for(int i=0; i < (int)particles.size(); i++){
        combined_array.push_back({0, i});
    }
    // for(int i=0; i < (int)springs.size(); i++){
    //     combined_array.push_back({1, i});
    // }
    // for(int i=0; i < (int)muscles.size(); i++){
    //     combined_array.push_back({2, i});
    // }
    
    // do we really need to sort the whole thing ?? not much would have changed in a single time step -- very few inversions present
    // std::sort(combined_array.begin(), combined_array.end(), 
    // [](Particle *p1, Particle *p2)->bool{
    //     return ((*p1).get_curr_pos()).x < ((*p2).get_curr_pos()).x;
    // });
    
    // TODO: sort the first time with nlogn then onwards use insertion sort but do it after testing everthing else
    sort(combined_array, particles, springs, muscles);
    
    // for(int i = 0; i < num_objects; i++){
        //     // std::cout << get_bounding_box_wall(direction::left, combined_array[i], particles, springs, muscles) << " ";
        //     std::cout << particles[combined_array[i].second].get_bounding_box_wall(direction::left) << " ";
        // }
        // std::cout << "\n\n\n";
        
    int num_objects = combined_array.size();
    for(int i=0; i < num_objects; i++){
        
        std::pair<int, int> obj_1 = combined_array[i];
        std::pair<int, int> obj_2;

        for(int j=i+1; j < num_objects; j++){
            
            // try removing this assignment maybe helps speeds -- this loop must be quite hot (havent checked yet)
            obj_2 = combined_array[j];
            // obj_2 is further right on x axis
            if( ! collision_check_and_handle(obj_1, obj_2, particles, springs, muscles))
                break;
        }
        
    }
}

// PASSING REFS TO COPY OF p1 and p2 wont affect the actual objects
void handle_particle_particle_collision(Particle& p1, Particle& p2){

    sf::Vector2f line_of_contact = p2.get_curr_pos() - p1.get_curr_pos();
    float r1 = p1.get_radius(); float r2 = p2.get_radius();
    float m1 = p1.get_mass();   float m2 = p2.get_mass();

    float dist = line_of_contact.length();
    float overlap = (r1+r2)-dist;
    if(overlap <= 0)
        return;

    sf::Vector2f p1_vel_before_collision = p1.get_vel();
    sf::Vector2f p2_vel_before_collision = p2.get_vel();
    
    sf::Vector2f p1_vel_along_loc = p1_vel_before_collision.projectedOnto(line_of_contact);
    sf::Vector2f p2_vel_along_loc = p2_vel_before_collision.projectedOnto(-line_of_contact);
    
    sf::Vector2f p1_vel_perp_loc = p1_vel_before_collision - p1_vel_along_loc;
    sf::Vector2f p2_vel_perp_loc = p2_vel_before_collision - p2_vel_along_loc;
    // this is wrt p1 
    sf::Vector2f rel_vel_perp_loc = p2_vel_perp_loc - p1_vel_perp_loc; float rel_vel_perp_loc_length = rel_vel_perp_loc.length();
    
    float c1 = (m1-m2*restitution)/(m1+m2);
    float c2 = (1+restitution)*(m2/(m1+m2));
    sf::Vector2f impulse_along_loc = ((c1*p1_vel_along_loc + c2*p2_vel_along_loc) - p1_vel_along_loc) * m1;

    float magnitude_impulse_friction = fmin(coefficient_friction*impulse_along_loc.length(), fmax(rel_vel_perp_loc_length, min_rel_vel_for_friction)*fmin(m1,  m2));
    sf::Vector2f impulse_perp_loc = magnitude_impulse_friction*rel_vel_perp_loc/(rel_vel_perp_loc_length != 0 ? rel_vel_perp_loc_length  : 1.f);
        
    sf::Vector2f p1_vel_after_collision = p1_vel_before_collision + (impulse_along_loc + impulse_perp_loc)/m1;
    sf::Vector2f p2_vel_after_collision = p2_vel_before_collision - (impulse_along_loc + impulse_perp_loc)/m2;

    sf::Vector2f p1_new_pos = p1.get_curr_pos() - (overlap/dist*0.5f)*line_of_contact;
    sf::Vector2f p2_new_pos = p2.get_curr_pos() + (overlap/dist*0.5f)*line_of_contact;

    p1.set_pos_vel(p1_new_pos, p1_vel_after_collision);
    p2.set_pos_vel(p2_new_pos, p2_vel_after_collision);
}