# path-planning-ode

This is my revamped code for solving the autonomous path planning problem outlined at https://www.youtube.com/watch?v=fNBrIngCJp8&t=9s

## REQUIREMENTS:

-numpy
-sympy
-matplotlib

## MODULE USAGE:

__________________________________
    To initialize Path Planning Object:
    
    pp = PathPlanningODE(
                        starting_coordinate - tuple, (-2, -2) by default
                        ending_coordinate - tuple, (12, 12) by default
                        NUM_OF_STEPS - int, 20 by default
                        )

    __________________________________
    To create obstacles in the Path Planning field:
    
    pp.create_obstacles(
                        NUM_OF_OBSTACLES - int, 10 by default
                        coordinates - list of tuples, None by default
                        )

        Example usage:

            To place one obstacle at (5, 5)
            pp.create_obstacles( obstacle=[(5,5)] )

            To place two obstacles, one at (1,2) the other at (3,4)
            pp.create_obstacles( obstacle=[(1,2), (3,4)] )

            To place 5 obstacles randomly
            pp.create_obstacles(5)

    __________________________________
    To show current state of path planning problem:
    
    pp.show_solution()

    __________________________________
    To do one iteration of Newton's Method (calculate new path based on current path):
    
    pp.update_path()

    __________________________________
    To watch an animation of the ODE sovler in action (this is fun - this is where you'll see it
    trying to wiggle the path around into place):
    
    pp.animate_solver()

    __________________________________
    To watch the 'rover' follow the current path:
    
    pp.animate_rover()  

## SAMPLE USAGE

import pathplanning2019 # import module

pp = PathPlanningODE() # create instance of path planning object w/ default settings

pp.create_obstacles() # create 10 randomly placed obstacles

pp.animate_solver() # animate the solution process
