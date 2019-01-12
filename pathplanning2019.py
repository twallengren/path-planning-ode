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
import matplotlib.pyplot as plt

################################################################################
################################################################################
# Define path planning class

class PathPlanningODE():
    """
    This class is the main API for the path planning program.
    """

    def __init__(self,
                 starting_coordinate = (0,0),
                 ending_coordinate = (10,10),
                 NUM_OF_STEPS = 20,
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
        self.Ode = ODE()

    def create_obstacle(self,
                       coordinate = None,
                       ):

        # Create random coordinate if not specified
        if coordinate is None:

            coordinate = (10*np.random.rand(), 10*np.random.rand())

        # Create new obstacle
        obstacle = Obstacle(coordinate)

        # Append obstace to obstacle list
        self.obstacle_list.append(obstacle)

        # Add obstacle to cost function
        self.Ode.add_to_cost(obstacle)

    def update_path(self):

        # Get step size (change in time for each point in parametrized path)
        self.STEP_SIZE = self.Path.time_list[1]

        # Create solution vector with 2*NUM_OF_STEPS rows and 1 column
        # First half are path x-coordinates, second half are path y-coordinates
        old_path = np.concatenate((self.Path.path[0][1:self.NUM_OF_STEPS+1],
                                         self.Path.path[1][1:self.NUM_OF_STEPS+1]),
                                         axis=None).reshape(-1, 1)

        # Initialize array for calculating ode delta
        delta = np.zeros([2*self.NUM_OF_STEPS,1])

        # Define input to calculate delta at beginning of path
        x0, x1, x2 = self.Rover.starting_coordinate[0], old_path[0], old_path[1]
        y0, y1, y2 = self.Rover.starting_coordinate[1], old_path[self.NUM_OF_STEPS], old_path[self.NUM_OF_STEPS+1]

        # Calculate first set of ode deltas
        delta[0], delta[self.NUM_OF_STEPS] = self.Ode.ode_delta(x0, x1, x2, y0, y1, y2, self.STEP_SIZE)

        # Loop to calculate intermediate ode deltas
        for i in range(1,self.NUM_OF_STEPS-1):
            delta[i], delta[self.NUM_OF_STEPS+i] = self.Ode.ode_delta(old_path[i-1],old_path[i],old_path[i+1],old_path[self.NUM_OF_STEPS-1+i],old_path[self.NUM_OF_STEPS+i],old_path[self.NUM_OF_STEPS+1+i],self.STEP_SIZE)


        # Define input to calculate delta at beginning of path
        x0, x1, x2 = old_path[self.NUM_OF_STEPS-2], old_path[self.NUM_OF_STEPS-1], self.Rover.ending_coordinate[0]
        y0, y1, y2 = old_path[2*self.NUM_OF_STEPS-2], old_path[2*self.NUM_OF_STEPS-1], self.Rover.ending_coordinate[1]
        
        # Calculate final ODE delta
        delta[self.NUM_OF_STEPS-1], delta[2*self.NUM_OF_STEPS-1] = self.Ode.ode_delta(x0, x1, x2, y0, y1, y2, self.STEP_SIZE)

        # Compute jacobian
        jacobian = self.Ode.jacobian(self.Rover.starting_coordinate, self.Rover.ending_coordinate, old_path, self.NUM_OF_STEPS, self.STEP_SIZE)

        # Get inverse of jacobian
        jacobian_inverse = np.linalg.inv(jacobian)

        # Create new path using Newton's Method
        new_path = old_path - np.dot(jacobian_inverse, delta)

        # Update Path object
        self.Path.update_path(new_path)

    def show_solution(self):

        # Plot rover
        plt.plot(self.Rover.current_coordinate[0], self.Rover.current_coordinate[1], 'ro')

        # Plot obstacles
        for obstacle in self.obstacle_list:

            plt.plot(obstacle.coordinate[0], obstacle.coordinate[1], 'x')

        # Plot solution
        plt.plot(self.Path.path[0], self.Path.path[1])

        # Show plot
        plt.show()
        
        
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
        self.startxcoord, self.startycoord = Rover.starting_coordinate
        self.endxcoord, self.endycoord = Rover.ending_coordinate

        # Initialize guess path function (builds a straight line parametrized by t)
        self.path_func = lambda t: (self.startxcoord * (1-t) + self.endxcoord * t, self.startycoord * (1-t) + self.endycoord * t)

        # Set time list with defined number of steps
        self.time_list = np.linspace(0, 1, NUM_OF_STEPS + 2)

        # Initialize guess path with t ranging from 0 to 1
        self.path = self.path_func(self.time_list)

        # Initialize path modification count (for use in __repr__ method)
        self.mod_count = 0

        # Set NUM_OF_STEPS on class variabls
        self.NUM_OF_STEPS = NUM_OF_STEPS

    def __repr__(self):

        return str(f"The path has been modified {self.mod_count} times.")

    def change_guess_func(self,
                          new_function,
                          ):

        # Set new path function
        self.path_func = new_function

        # Set new guess path with t ranging from 0 to 1
        self.path = self.path_func(np.linspace(0, 1, NUM_OF_STEPS))

    def update_path(self, new_path):

        # Break new_path into x & y and append start + end coordinates
        solx = np.zeros([self.NUM_OF_STEPS+2,1])
        solx[0] = self.startxcoord
        solx[1:self.NUM_OF_STEPS+1] = new_path[0:int(len(new_path)/2)]
        solx[self.NUM_OF_STEPS+1] = self.endxcoord

        soly = np.zeros([self.NUM_OF_STEPS+2,1])
        soly[0] = self.startycoord
        soly[1:self.NUM_OF_STEPS+1] = new_path[int(len(new_path)/2):int(len(new_path))]
        soly[self.NUM_OF_STEPS+1] = self.endycoord

        self.path = solx, soly
                 
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

    def jacobian(self, ALPHA, BETA, solmat, N, h):

        # Initialize matrix
        jac = np.zeros([2*N,2*N])

        # Populate the first row of M11 (upper left square of jac)
        jac[0,0] = -2/h**2 - self.Fx(solmat[0],solmat[N],
                            (solmat[1]-ALPHA[0])/(2*h),(solmat[N+1]-ALPHA[1])/(2*h))
        jac[0,1] = 1/h**2 - (1/(2*h))*self.Fxp(solmat[0],solmat[N],
                            (solmat[1]-ALPHA[0])/(2*h),(solmat[N+1]-ALPHA[1])/(2*h))

        # Populate the first row of M21 (lower left square of jac)
        jac[N,0] = -self.Gx(solmat[0],solmat[N],
                    (solmat[1]-ALPHA[0])/(2*h),(solmat[N+1]-ALPHA[1])/(2*h))
        jac[N,1] = (-1/(2*h))*self.Gxp(solmat[0],solmat[N],
                            (solmat[1]-ALPHA[0])/(2*h),(solmat[N+1]-ALPHA[1])/(2*h))

        # Populate the first row of M12 (upper right square of jac)
        jac[0,N] = -self.Fy(solmat[0],solmat[N],
                            (solmat[1]-ALPHA[0])/(2*h),(solmat[N+1]-ALPHA[1])/(2*h))
        jac[0,N+1] = (-1/(2*h))*self.Fyp(solmat[0],solmat[N],
                            (solmat[1]-ALPHA[0])/(2*h),(solmat[N+1]-ALPHA[1])/(2*h))

        # Populate the first row of M22 (lower right square of jac)
        jac[N,N] = -2/h**2 - self.Gy(solmat[0],solmat[N],
                            (solmat[1]-ALPHA[0])/(2*h),(solmat[N+1]-ALPHA[1])/(2*h))
        jac[N,N+1] = 1/h**2 - (1/(2*h))*self.Gyp(solmat[0],solmat[N],
                            (solmat[1]-ALPHA[0])/(2*h),(solmat[N+1]-ALPHA[1])/(2*h))

        # Loop to populate intermediate values of jac
        for i in range(1,N-1):

            # Populate intermediate values of M11
            jac[i,i-1] = 1/h**2 + (1/(2*h))*self.Fxp(solmat[i],solmat[N+i],
                            (solmat[i+1]-solmat[i-1])/(2*h),(solmat[N+1+i]-solmat[N-1+i])/(2*h))
            jac[i,i] = -2/h**2 - self.Fx(solmat[i],solmat[N+i],
                            (solmat[i+1]-solmat[i-1])/(2*h),(solmat[N+1+i]-solmat[N-1+i])/(2*h))
            jac[i,i+1] = 1/h**2 - (1/(2*h))*self.Fxp(solmat[i],solmat[N+i],
                            (solmat[i+1]-solmat[i-1])/(2*h),(solmat[N+1+i]-solmat[N-1+i])/(2*h))

            # Populate intermediate values of M21
            jac[N+i,i-1] = (1/(2*h))*self.Gxp(solmat[i],solmat[N+i],
                            (solmat[i+1]-solmat[i-1])/(2*h),(solmat[N+1+i]-solmat[N-1+i])/(2*h))
            jac[N+i,i] = -self.Gx(solmat[i],solmat[N+i],
                            (solmat[i+1]-solmat[i-1])/(2*h),(solmat[N+1+i]-solmat[N-1+i])/(2*h))
            jac[N+i,i+1] = (-1/(2*h))*self.Gxp(solmat[i],solmat[N+i],
                            (solmat[i+1]-solmat[i-1])/(2*h),(solmat[N+1+i]-solmat[N-1+i])/(2*h))

            # Populate intermediate values of M12
            jac[i,N+i-1] = (1/(2*h))*self.Fyp(solmat[i],solmat[N+i],
                            (solmat[i+1]-solmat[i-1])/(2*h),(solmat[N+1+i]-solmat[N-1+i])/(2*h))
            jac[i,N+i] = -self.Fy(solmat[i],solmat[N+i],
                            (solmat[i+1]-solmat[i-1])/(2*h),(solmat[N+1+i]-solmat[N-1+i])/(2*h))
            jac[i,N+i+1] = (-1/(2*h))*self.Fyp(solmat[i],solmat[N+i],
                            (solmat[i+1]-solmat[i-1])/(2*h),(solmat[N+1+i]-solmat[N-1+i])/(2*h))

            # Populate intermediate values of M22
            jac[N+i,N+i-1] = 1/h**2 + (1/(2*h))*self.Gyp(solmat[i],solmat[N+i],
                            (solmat[i+1]-solmat[i-1])/(2*h),(solmat[N+1+i]-solmat[N-1+i])/(2*h))
            jac[N+i,N+i] = -2/h**2 - self.Gy(solmat[i],solmat[N+i],
                            (solmat[i+1]-solmat[i-1])/(2*h),(solmat[N+1+i]-solmat[N-1+i])/(2*h))
            jac[N+i,N+i+1] = 1/h**2 - (1/(2*h))*self.Gyp(solmat[i],solmat[N+i],
                            (solmat[i+1]-solmat[i-1])/(2*h),(solmat[N+1+i]-solmat[N-1+i])/(2*h))

        # Populate final values of M11
        jac[N-1,N-2] = 1/h**2 + (1/(2*h))*self.Fxp(solmat[N-1],solmat[2*N-1],
                            (BETA[0] - solmat[N-2])/(2*h),(BETA[1] - solmat[2*N-2])/(2*h))
        jac[N-1,N-1] = -2/h**2 - self.Fx(solmat[N-1],solmat[2*N-1],
                            (BETA[0] - solmat[N-2])/(2*h),(BETA[1] - solmat[2*N-2])/(2*h))

        # Populate final values of M21
        jac[2*N-1,N-2] = (1/(2*h))*self.Gxp(solmat[N-1],solmat[2*N-1],
                            (BETA[0] - solmat[N-2])/(2*h),(BETA[1] - solmat[2*N-2])/(2*h))
        jac[2*N-1,N-1] = -self.Gx(solmat[N-1],solmat[2*N-1],
                            (BETA[0] - solmat[N-2])/(2*h),(BETA[1] - solmat[2*N-2])/(2*h))

        # Populate final values of M12
        jac[N-1,2*N-2] = (1/(2*h))*self.Fyp(solmat[N-1],solmat[2*N-1],
                            (BETA[0] - solmat[N-2])/(2*h),(BETA[1] - solmat[2*N-2])/(2*h))
        jac[N-1,2*N-1] = -self.Fy(solmat[N-1],solmat[2*N-1],
                            (BETA[0] - solmat[N-2])/(2*h),(BETA[1] - solmat[2*N-2])/(2*h))

        # Populate final values of M22
        jac[2*N-1,2*N-2] = 1/h**2 + (1/(2*h))*self.Gyp(solmat[N-1],solmat[2*N-1],
                            (BETA[0] - solmat[N-2])/(2*h),(BETA[1] - solmat[2*N-2])/(2*h))
        jac[2*N-1,2*N-1] = -2/h**2 - self.Gy(solmat[N-1],solmat[2*N-1],
                            (BETA[0] - solmat[N-2])/(2*h),(BETA[1] - solmat[2*N-2])/(2*h))

        return jac
