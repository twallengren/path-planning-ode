# path-planning-ode

This is my revamped code for solving the autonomous path planning problem outlined at https://www.youtube.com/watch?v=fNBrIngCJp8&t=9s

## REQUIREMENTS:

-numpy
-sympy
-matplotlib

## SCRIPT USAGE:

When run as a script, it will drop 5 obstacles in at random, attempt to solve the path planning problem, then plot the results

## MODULE USAGE:

The main API is the PathPlanningODE class. 

>>> pp_instance = PathPlanningODE(startingcoordinate (tuple, optional), endingcoordinate (tuple, optional)) to create instance of class

>>> pp_instance.create_obstacle() to create random obstacle

>>> pp_instance.create_obstacle((xcoordinate, ycoordinate)) to create obstacle at specific location

>>> pp_instance.update_path() to iterate Newton's Method

>>> pp_instance.show_solution() to display current results

>>> pp_instance.animate_solver() to show animation of ODE solver in progress




Hopefully this is simpler to use. The code to run the script behavior is literally 6 lines long which seems like good news to me.
