# Fill in the respective functions to implement the controller

# Import libraries
import numpy as np
from base_controller import BaseController
from scipy import signal, linalg
from util import *
import cmath
# CustomController class (inherits from BaseController)
class CustomController(BaseController):

    def __init__(self, trajectory):

        super().__init__(trajectory)

        # Define constants
        
        # These can be ignored in P1
        self.lr = 1.39
        self.lf = 1.55
        self.Ca = 20000
        self.Iz = 25854
        self.m = 1888.6
        self.g = 9.81

        # Add additional member variables according to your need here.

        self.e1_prev = 0
        self.e2_prev = 0
        
        self.err_prev_longitudinal = 0
        self.err_cumulative_longitudinal = 0
        
        self.Kp_longitudinal = 120
        self.Kd_longitudinal = 0.001
        self.Ki_longitudinal = 0.05

    def update(self, timestep):

        trajectory = self.trajectory

        lr = self.lr
        lf = self.lf
        Ca = self.Ca
        Iz = self.Iz
        m = self.m
        g = self.g

        # Fetch the states from the BaseController method
        delT, X, Y, xdot, ydot, psi, psidot = super().getStates(timestep)

        # Design your controllers in the spaces below. 
        # Remember, your controllers will need to use the states
        # to calculate control inputs (F, delta). 

        closest_dist,index = closestNode(X, Y, trajectory)
        # Calculating the desired distance slightly ahead of the closest point on the trajectory
        Xr = trajectory[index+40,0]
        Yr = trajectory[index+40,1]
        xrdot = 30

        # ---------------|Lateral Controller|-------------------------
        """
        Please design your lateral controller below.

        """
        e1 = closest_dist
        e2 = wrapToPi(np.arctan2((Y-Yr),(X-Xr))-psi)
        e1dot = (e1 - self.e1_prev)/delT
        e2dot = (e2 - self.e2_prev)/delT
        self.e1_prev = e1
        self.e2_prev = e2

        A = np.array([[0,1,0,0],
                      [0,-4*(Ca/(m*xrdot)),4*(Ca/m),-2*Ca*(lf-lr)/(m*xrdot)],
                      [0,0,0,1],
                      [0,-2*Ca*(lf-lr)/(Iz*xrdot),2*Ca*(lf-lr)/Iz,-2*Ca*(lf**2+lr**2)/(Iz*xrdot)]])
        B = np.array([[0],
                      [2*Ca/m],
                      [0],
                      [2*Ca*lf/Iz]])

        P = np.array([complex(-0.5,-0.5),complex(-0.5,0.5),-2,-5])
        fsf1 = signal.place_poles(A, B, P, rtol =0.001, maxiter=30)
        K = fsf1.gain_matrix        
        states = np.array([[e1],
                           [e1dot],
                           [e2],
                           [e2dot]])
        u = np.matmul(-1*K,states)
        delta = u[0,0]

        # ---------------|Longitudinal Controller|-------------------------
        """
        Please design your longitudinal controller below.
        """
        Kp = self.Kp_longitudinal
        Kd = self.Kd_longitudinal
        Ki = self.Ki_longitudinal

        err_velocity_curr = xrdot-xdot
        self.err_cumulative_longitudinal += err_velocity_curr
        F = Kp*err_velocity_curr + Kd*(err_velocity_curr-self.err_prev_longitudinal)/delT + Ki*self.err_cumulative_longitudinal*delT 
        self.err_prev_longitudinal = err_velocity_curr
        # Return all states and calculated control inputs (F, delta)
        return X, Y, xdot, ydot, psi, psidot, F, delta
