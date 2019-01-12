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

        # Create ODE object to solve
        self.ode = ODE()

    def create_obstacle(self,
                       coordinate = None,
                       ):

        # Create random coordinate if not specified
        if coordinate is None:

            coordinate = (np.random.rand(), np.random.rand())

        # Create new obstacle
        obstacle = Obstacle(coordinate)

        # Append obstace to obstacle list
        self.obstacle_list.append(obstacle)

        # Add obstacle to cost function
        self.ode.add_to_cost(obstacle)
        
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

        # Initialize ODEs from video
        self.update_ode()

    def add_to_cost(self,
                    obstacle,
                    ):

        # Get obstacle coordinates
        xcoord, ycoord = obstacle.coordinate

        # Append coordinate to cost function with appropriate term
        self.cost += sp.exp(-((-self.xs + xcoord)**2 + (-self.ys + ycoord)**2))

        # Update primary symbolic ODE
        self.update_ode()

    def update_ode(self):

        # Take derivatives of cost function
        costx = sp.diff(self.cost,self.xs)
        costy = sp.diff(self.cost,self.ys)

        # Create denominator
        denom = 2*self.cost

        # Update primary symbolic ODE
        self.x_double_prime = (costx*(self.yps**2 - self.xps**2) - 2*costy*self.xps*self.yps)/denom
        self.y_double_prime = (costy*(self.xps**2 - self.yps**2) - 2*costx*self.xps*self.yps)/denom

