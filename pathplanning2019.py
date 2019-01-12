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
import utils

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

        # Add NUM_OF_STEPS to class variables
        self.NUM_OF_STEPS = NUM_OF_STEPS

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

    def update_path(self):

        # Get step size (change in time for each point in parametrized path)
        self.step_size = self.Path.time_list[1]

        # Create solution vector with 2*NUM_OF_STEPS rows and 1 column
        # First half are path x-coordinates, second half are path y-coordinates
        self.next_path = np.concatenate((self.Path.path[0][1:self.NUM_OF_STEPS+1],
                                         self.Path.path[1][1:self.NUM_OF_STEPS+1]),
                                         axis=None)

        # Initialize array for calculating ode delta
        self.delta = np.zeros([2*self.NUM_OF_STEPS,1])

        # Define input to calculate delta at beginning of path
        x0, x1, x2 = self.Rover.starting_coordinate[0], self.next_path[0], self.next_path[1]
        y0, y1, y2 = self.Rover.starting_coordinate[1], self.next_path[self.NUM_OF_STEPS], self.next_path[self.NUM_OF_STEPS+1]

        # Calculate first set of ode deltas
        self.delta[0], self.delta[self.NUM_OF_STEPS] = self.ode.ode_delta(x0, x1, x2, y0, y1, y2, self.step_size)

        # Loop to calculate intermediate ode deltas
        for i in range(1,self.NUM_OF_STEPS-1):
            self.delta[i], self.delta[self.NUM_OF_STEPS+i] = self.ode.ode_delta(self.next_path[i-1],self.next_path[i],self.next_path[i+1],self.next_path[self.NUM_OF_STEPS-1+i],self.next_path[self.NUM_OF_STEPS+i],self.next_path[self.NUM_OF_STEPS+1+i],self.step_size)


        # Define input to calculate delta at beginning of path
        x0, x1, x2 = self.next_path[self.NUM_OF_STEPS-2], self.next_path[self.NUM_OF_STEPS-1], self.Rover.ending_coordinate[0]
        y0, y1, y2 = self.next_path[2*self.NUM_OF_STEPS-2], self.next_path[2*self.NUM_OF_STEPS-1], self.Rover.ending_coordinate[1]
        
        # Calculate final ODE delta
        self.delta[self.NUM_OF_STEPS-1], self.delta[2*self.NUM_OF_STEPS-1] = self.ode.ode_delta(x0, x1, x2, y0, y1, y2, self.step_size)

        print(self.delta)
        
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

        # Get rover coordinates
        startxcoord, startycoord = Rover.starting_coordinate
        endxcoord, endycoord = Rover.ending_coordinate

        # Initialize guess path function (builds a straight line parametrized by t)
        self.path_func = lambda t: (startxcoord * (1-t) + endxcoord * t, startycoord * (1-t) + endycoord * t)

        # Set time list with defined number of steps
        self.time_list = np.linspace(0, 1, NUM_OF_STEPS + 2)

        # Initialize guess path with t ranging from 0 to 1
        self.path = self.path_func(self.time_list)

        # Initialize path modification count (for use in __repr__ method)
        self.mod_count = 0

    def __repr__(self):

        return str(f"The path has been modified {self.mod_count} times.")

    def change_guess_func(self,
                          new_function,
                          ):

        # Set new path function
        self.path_func = new_function

        # Set new guess path with t ranging from 0 to 1
        self.path = self.path_func(np.linspace(0, 1, NUM_OF_STEPS))
                 
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
        self.f = (costx*(self.yps**2 - self.xps**2) - 2*costy*self.xps*self.yps)/denom
        self.g = (costy*(self.xps**2 - self.yps**2) - 2*costx*self.xps*self.yps)/denom

        # Update various derivatives used in the solver
        self.fx = sp.diff(self.f,self.xs)
        self.gx = sp.diff(self.g,self.xs)
        self.fxp = sp.diff(self.f,self.xps)
        self.gxp = sp.diff(self.g,self.xps)
        self.fy = sp.diff(self.f,self.ys)
        self.gy = sp.diff(self.g,self.ys)
        self.fyp = sp.diff(self.f,self.yps)
        self.gyp = sp.diff(self.g,self.yps)

        # "Lambdify" all symbolic functions
        # Allows calls to evaluate functions at specific points
        self.F = sp.lambdify((self.xs,self.ys,self.xps,self.yps),self.f)
        self.G = sp.lambdify((self.xs,self.ys,self.xps,self.yps),self.g)
        self.Fx = sp.lambdify((self.xs,self.ys,self.xps,self.yps),self.fx)
        self.Gx = sp.lambdify((self.xs,self.ys,self.xps,self.yps),self.gx)
        self.Fxp = sp.lambdify((self.xs,self.ys,self.xps,self.yps),self.fxp)
        self.Gxp = sp.lambdify((self.xs,self.ys,self.xps,self.yps),self.gxp)
        self.Fy = sp.lambdify((self.xs,self.ys,self.xps,self.yps),self.fy)
        self.Gy = sp.lambdify((self.xs,self.ys,self.xps,self.yps),self.gy)
        self.Fyp = sp.lambdify((self.xs,self.ys,self.xps,self.yps),self.fyp)
        self.Gyp = sp.lambdify((self.xs,self.ys,self.xps,self.yps),self.gyp)

    def ode_delta(self, x0, x1, x2, y0, y1, y2, h):

        xout = (x2 - 2*x1 + x0)/h - self.F(x1,y1,(x2-x0)/(2*h),(y2-y0)/(2*h))
        yout = (y2 - 2*y1 + y0)/h - self.G(x1,y1,(x2-x0)/(2*h),(y2-y0)/(2*h))

        return xout, yout
