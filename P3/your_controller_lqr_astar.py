# Fill in the respective functions to implement the LQR optimal controller

# Import libraries
import numpy as np
from base_controller import BaseController
from scipy import signal, linalg
from util import *

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
        delT, X, Y, xdot, ydot, psi, psidot, obstacleX, obstacleY = super().getStates(timestep)

        closest_dist,index = closestNode(X, Y, trajectory)
        # Calculating the desired distance slightly ahead of the closest point on the trajectory
        Xr = trajectory[index+40,0]
        Yr = trajectory[index+40,1]
        xrdot = 35

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
        B = np.array([[0,0],
                      [2*Ca/m,0],
                      [0,0],
                      [2*Ca*lf/Iz,0]])
        Ad = linalg.expm(A*delT)
        
        f = lambda x: linalg.expm(A*x)
        xv = np.linspace(0,delT,200)
        result = np.apply_along_axis(f,0,xv.reshape(1,-1))
        Bd = np.trapz(result,xv)@B

        Q = np.identity(4)
        R = 25*np.identity(2)
        S = np.array(linalg.solve_discrete_are(Ad,Bd,Q,R))
        K = -np.array(linalg.inv(Bd.T@S@Bd + R)@(Bd.T@S@Ad))
        states = np.array([[e1],
                           [e1dot],
                           [e2],
                           [e2dot]])
        u = np.matmul(K,states)
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



        # Return all states and calculated control inputs (F, delta) and obstacle position
        return X, Y, xdot, ydot, psi, psidot, F, delta, obstacleX, obstacleY
