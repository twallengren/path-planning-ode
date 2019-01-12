"""
Autonomous Path Planning ODE

This module solves the ODE derived at
https://www.youtube.com/watch?v=fNBrIngCJp8&t=9s

"""
################################################################################
################################################################################
# Import necessary libraries
import numpy as np
import sympy as sp

################################################################################
################################################################################
# Define path planning class

class PathPlanningODE():
    """
    This class is the main API for the path planning program.
    """

    def __init__(self,
                 starting_coordinate = (0,0),
                 ending_coordinate = (1,1),
                 NUM_OF_STEPS = 10,
                 ):

        # Initialize obstacle list
        self.obstacle_list = []

        # Create instance of rover object
        self.Rover = Rover(starting_coordinate, ending_coordinate)

        # Create initial guess path based on rover start/end coordinates
        self.Path = Path(self.Rover, NUM_OF_STEPS)

    def create_obstacle(self,
                       coordinate = (np.random.rand(), np.random.rand()),
                       ):

        # Append obstace to obstacle list
        self.obstacle_list.append(Obstacle(coordinate))

        # Add term for obstacle to cost function
        cost += sp.exp(-((-xs + coordinate[0])**2 + (-ys + coordinate[1])**2))

        print(cost)

    
        
################################################################################
################################################################################
# Define rover class

class Rover():
    """
    This class represents the rover that needs to travel the path.
    """

    def __init__(self,
                 starting_coordinate = (0, 0),
                 ending_coordinate = (1, 1),
                 ):

        # Store starting and ending coordinates on self
        self.starting_coordinate = starting_coordinate
        self.ending_coordinate = ending_coordinate

        # Set current coordinate to starting coordinate at initialization
        self.current_coordinate = self.starting_coordinate

    def __repr__(self):

        return str(f"Rover's current coordinate is {self.current_coordinate}")

################################################################################
################################################################################
# Define obstacle class

class Obstacle():
    """
    This class represents an individual obstacle the path should avoid.
    """

    def __init__(self,
                 coordinate = (0.5, 0.5),
                 weight = 1,
                 ):

        # Store obstacle coordinate on self
        self.coordinate = coordinate
        self.weight = weight

    def __repr__(self):

        return str(self.coordinate)

################################################################################
################################################################################
# Define path class

class Path():
    """
    This class represents the path rover should travel.
    """

    def __init__(self,
                 Rover,
                 NUM_OF_STEPS = 10,
                 ):

        # Initialize straight line between rover start and end coordinates
        self.path_x = np.linspace(Rover.starting_coordinate[0], Rover.ending_coordinate[0], NUM_OF_STEPS)
        self.path_y = np.linspace(Rover.starting_coordinate[1], Rover.ending_coordinate[1], NUM_OF_STEPS)

        # Initialize path modification count (for use in __repr__ method)
        self.mod_count = 0

    def __repr__(self):

        return str(f"The path has been modified {self.mod_count} times.")
                 
################################################################################
################################################################################
# Define ODE class

class ODE():
    """
    This class represents the ODE that solves the path planning problem.
    """

    def __init__(self):

        # Create symbolic variables (need symbolic to take derivatives)
        self.xs = sp.Symbol('xs') # 'xs' stands for x-symbolic
        self.ys = sp.Symbol('ys') # 'ys' stands for y-symbolic
        self.xps = sp.Symbol('xps') # 'xps' stands for xprime-symbolic
        self.yps = sp.Symbol('yps') # 'yps' stands for yprime-symbolic
        self.ts = sp.Symbol('ts') # 'ts' stands for t-symbolic (time)

        # Initialize cost function
        self.cost = 1
