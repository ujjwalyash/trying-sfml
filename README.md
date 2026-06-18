# Simulated Creatures 


## Quick start

1. run ./run.sh to build + start the simulation
2. parameters can be tuned in the main.cpp file
3. when runnning the evolution and simulation is handled by num_worker + 1 threads  the main thread does the crossover, mutations and the worker threads run the simulation to evalute the creatures
4. a seperate thread for all these is run to visualize the current generation 
5. Some stats are shown at the top right cornere
   - The current reward for this episode(the one you are seeing)
   - num_steps_done in the current episode
   - The generation number you are seeing right now
   - The rank of the creature on display right now(ranked according to rewards in last generation)

6. the controls are (press ESC to start from the black screen)
   - Esc -- pause
   - Q -- quit
   - R -- Reset the episode and switch to the latest generation
   - Enter -- run step by step when paused
   - Up -- speed up the simulation
   - Down -- slow down
   - Right/End -- move the curr_rank+1/curr_rank+10 th creature
   - Left/Home -- move the curr_rank-1/curr_rank-10 th creature
