#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import matrix_rank
from scipy.linalg import svd
from scipy import signal


# In[2]:


c = 20000.
m = 1888.6
lf = 1.55
lr = 1.39
Iz = 25854.


# In[3]:


# For lateral control
def lateral_dynamics_check(xdot):
    print("For xdot = {}:\n".format(xdot))
    A = np.array([[0,1,0,0],
                  [0,-4*(c/(m*xdot)),4*(c/m),-2*c*(lf-lr)/(m*xdot)],
                  [0,0,0,1],
                  [0,-2*c*(lf-lr)/(Iz*xdot),2*c*(lf-lr)/Iz,-2*c*(lf**2+lr**2)/(Iz*xdot)]])
    B = np.array([[0,0],
                  [2*c/m,0],
                  [0,0],
                  [2*c*lf/Iz,0]])
    C = np.identity(4)

    p1 = B
    p2 = np.matmul(A,p1)
    p3 = np.matmul(A,p2)
    p4 = np.matmul(A,p3)

    P = np.concatenate((p1,p2,p3,p4),axis=1)
    print("Rank of controllabilty matrix = {} ".format(matrix_rank(P)))

    q1 = C
    q2 = np.matmul(q1,A)
    q3 = np.matmul(q2,A)
    q4 = np.matmul(q3,A)

    Q = np.concatenate((q1,q2,q3,q4),axis=0)
    print("Rank of observability matrix = {}\n".format(matrix_rank(Q)))


# In[4]:


# For longitudinal control
def longitudinal_dynamics_check(xdot):
    print("For xdot = {}:\n".format(xdot))
    A = np.array([[0,1],
                  [0,0]])
    B = np.array([[0,0],
                  [0,1/m]])
    C = np.identity(2)

    p1 = B
    p2 = np.matmul(A,B)

    P = np.concatenate((p1,p2),axis=1)
    print("Rank of controllability matrix = {} ".format(matrix_rank(P)))

    q1 = C
    q2 = np.matmul(C,A)

    Q = np.concatenate((q1,q2),axis=0)
    print("Rank of observability matrix = {}\n".format(matrix_rank(Q)))


# ## Excercise 1.1

# In[5]:


# Lateral dynamics
xdot = np.asarray([2.,5.,8.])
for i in range(len(xdot)):
    lateral_dynamics_check(xdot[i])
    


# In[6]:


# Longitudinal dynamics
for i in range(len(xdot)):
    longitudinal_dynamics_check(xdot[i])


# # The system is fully controllable and observable for all three values of $\dot x$ = 2,5,8 m/s

# ## Excercise 1.2(a)

# In[7]:


# Longitudinal dynamics

A = np.array([[0,1],
                  [0,0]])
B = np.array([[0,0],
                  [0,1/m]])
C = np.identity(2)
D = np.zeros((2,2))

p1 = B
p2 = np.matmul(A,B)

P = np.concatenate((p1,p2),axis=1)
    
U, s, VT = svd(P)
ratio = np.log10(s[0]/s[1]) 
print("The logarithmic ratio of the maximum and minimum singular values for the longitudinal model is {}".format(ratio))
    
num,den = signal.ss2tf(A, B, C, D)
poles = np.roots(den)
print("The poles for the longitudinal model are: {}".format(poles))


# 
# The controllabilty matrix is independent of velocity and the system is fully controllable(Ratio of the maximum and minimum singular values is 1. Since, both the poles are zero, the open loop system is unstable.

# In[8]:


# Lateral dynamics

v = np.linspace(1,40,40)
ratio = []

for xdot in v:
    A = np.array([[0,1,0,0],
                  [0,-4*(c/(m*xdot)),4*(c/m),-2*c*(lf-lr)/(m*xdot)],
                  [0,0,0,1],
                  [0,-2*c*(lf-lr)/(Iz*xdot),2*c*(lf-lr)/Iz,-2*c*(lf**2+lr**2)/(Iz*xdot)]])
    B = np.array([[0,0],
                  [2*c/m,0],
                  [0,0],
                  [2*c*lf/Iz,0]])
    C = np.identity(4)
    

    p1 = B
    p2 = np.matmul(A,p1)
    p3 = np.matmul(A,p2)
    p4 = np.matmul(A,p3)
    P = np.concatenate((p1,p2,p3,p4),axis=1)
    
    U, s, VT = svd(P)
    ratio.append(np.log10(s[0]/s[3]))
    


# In[9]:


plt.figure(figsize=(6,6))
plt.plot(v,ratio)
plt.xlabel("v")
plt.ylabel("Ratio")
plt.title('Ratio of logarithm of greatest and least singular values')
plt.show()


# ## Excercise 1.2(b)

# In[10]:


v = np.linspace(1,100,100)
p1 = []
p2 = []
p3 = []
p4 = []
for xdot in v:
    A = np.array([[0,1,0,0],
                  [0,-4*(c/(m*xdot)),4*(c/m),-2*c*(lf-lr)/(m*xdot)],
                  [0,0,0,1],
                  [0,-2*c*(lf-lr)/(Iz*xdot),2*c*(lf-lr)/Iz,-2*c*(lf**2+lr**2)/(Iz*xdot)]])
    B = np.array([[0,0],
                  [2*c/m,0],
                  [0,0],
                  [2*c*lf/Iz,0]])
    C = np.identity(4)
    D = np.zeros((4,2))
    
    num,den = signal.ss2tf(A, B, C, D)
    poles = np.roots(den)
    p1.append(poles[0].real)
    p2.append(poles[1].real)
    p3.append(poles[2].real)
    p4.append(poles[3].real)


# In[11]:


plt.figure(figsize=(12,12))

plt.subplot(2,2,1)
plt.plot(v,p1)
plt.xlabel("v")
plt.ylabel("Pole 1")
plt.title('Re(p1) vs v')

plt.subplot(2,2,2)
plt.plot(v,p2)
plt.xlabel("v")
plt.ylabel("Pole 2")
plt.title('Re(p2) vs v')

plt.subplot(2,2,3)
plt.plot(v,p3)
plt.xlabel("v")
plt.ylabel("Pole 3")
plt.title('Re(p3) vs v')

plt.subplot(2,2,4)
plt.plot(v,p4)
plt.xlabel("v")
plt.ylabel("Pole 4")
plt.title('Re(p4) vs v')

plt.show()


# Controllability:
# 
# For very low values of velocity the controllabilty matrix would be susceptible to lose full rank and become uncontrollable. But with an increase in velocity we may conclude that the system would be fully controllable.
# 
# Stability:
# 
# From the 1st plot, we observe that the pole has a negative real part.
# From the 2nd plot, we observe that at low velocities the pole has a negative real part and at high velocities the pole has a positive real part.
# From the 3rd plot, we can observe that at some velocities which are small one of the poles is greater than 0 (though the magnitude is very less).
# From the 4th plot, we find that one of the poles is always equal to zero.
# 
# It is clear, that the system is not stable since the real part of all the poles should be strictly negative for stability. As t->infinity the system would blow up and the possibilty of becoming unstable increases as the velocity increases.
