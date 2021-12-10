# Fill in the respective functions to implement the controller

# Import libraries
import numpy as np
from base_controller import BaseController
from scipy import signal, linalg
from util import *

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
        self.err_prev_lateral = 0
        self.err_cumulative_lateral = 0
        
        self.err_prev_longitudinal = 0
        self.err_cumulative_longitudinal = 0
        
        self.Kp_lateral = 5
        self.Kd_lateral = 0.001
        self.Ki_lateral = 0.001
        
        self.Kp_longitudinal = 30
        self.Kd_longitudinal = 0.005
        self.Ki_longitudinal = 0.05

    def update(self, timestep):

        trajectory = self.trajectory

        lr = self.lr
        lf = self.lf
        Ca = self.Ca
        Iz = self.Iz
        m  = self.m
        g  = self.g

        # Fetch the states from the BaseController method
        delT, X, Y, xdot, ydot, psi, psidot = super().getStates(timestep)
        # Design your controllers in the spaces below. 
        # Remember, your controllers will need to use the states
        # to calculate control inputs (F, delta). 
        Kp1 = self.Kp_lateral
        Kd1 = self.Kd_lateral
        Ki1 = self.Ki_lateral

        Kp2 = self.Kp_longitudinal
        Kd2 = self.Kd_longitudinal
        Ki2 = self.Ki_longitudinal

       
        closest_dist,index = closestNode(X, Y, trajectory)

        # Calculating the desired distance slightly ahead of the closest point on the trajectory
        Xr = trajectory[index+40,0]
        Yr = trajectory[index+40,1]
        xrdot=25
 
        # ---------------|Lateral Controller|-------------------------
        
        #Please design your lateral controller below.
        #For the calculation of the sterring angle: Error in angle
        err_angle_curr = wrapToPi(np.arctan2((Yr-Y),(Xr-X))-psi) 
        self.err_cumulative_lateral += err_angle_curr
        delta = Kp1*err_angle_curr + Kd1*(err_angle_curr-self.err_prev_lateral)/delT + Ki1*self.err_cumulative_lateral*delT
        self.err_prev_lateral = err_angle_curr

        # ---------------|Longitudinal Controller|-------------------------
        
        #Please design your longitudinal controller below.
        # For the calculation of the force: Error in velocity
        err_velocity_curr = xrdot-xdot
        self.err_cumulative_longitudinal += err_velocity_curr
        F = Kp2*err_velocity_curr + Kd2*(err_velocity_curr-self.err_prev_longitudinal)/delT + Ki2*self.err_cumulative_longitudinal*delT 
        self.err_prev_longitudinal = err_velocity_curr
        
        # Return all states and calculated control inputs (F, delta)
        return X, Y, xdot, ydot, psi, psidot, F, delta
