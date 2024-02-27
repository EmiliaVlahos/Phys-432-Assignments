#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Phys 432; Assignment 3 Code
Simulation of lava flow down an inclined plane.
@author: Emilia Vlahos

Created on Sat Feb 24 23:11:01 2024

"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg


#%% Set up grid 
H = 1 # heigt of lava in m
n = 500 # resolution in y
dy = H/n

tf = 5 # stop time in s
nsteps = 500 # number of timesteps
dt = tf/nsteps

#%% Gravitational contributions
g = 9.81 # gravitational acceleration in m/s^2
alpha = 10 # inclination of slope in degrees
K = g*np.sin(np.deg2rad(alpha)) 

#%% Viscous contributions
rho = 2700 # density of basaltic lava (from Wikipedia)
v = 1 # kinematic viscosity of lava (in m^2/s) as estimated in class

#%% Initialize Plot
y = np.linspace(0, H, n)
U = np.zeros(n) # assume lava is initially at rest
u_steady = -(g/v) * np.sin(np.deg2rad(alpha)) * (y**2/2 - H*y) # analytic steady state solution

plt.ion()
fig, ax = plt.subplots(1,1)
pl, = ax.plot(y, U)
ax.plot(y, u_steady, '--k', label='Steady State Solution')
ax.set_title("Flow of Lava")
ax.set_ylabel("Speed of lava [m/s]")
ax.set_xlabel("Height of lava [m]")
plt.legend()

fig.canvas.draw() 

#%% Define evolution matrix A
beta = v * dt/dy**2

# Calculate the matrix A in banded form
a = -beta * np.ones(n)
b = (1 + 2*beta) * np.ones(n) # No slip boudary condition is inculded without altering matrix since u0=0 always
c = -beta * np.ones(n)
a[0] = 0.0 
b[-1] = 1 + beta # No stress boudary condition
c[-1] = 0.0

A = np.row_stack((a,b,c))

#%% Evole U, updating at each timestep
for i in range(1, nsteps):
    U = scipy.linalg.solve_banded((1,1), A, U + dt*K)
    pl.set_ydata(U)
    fig.canvas.draw()
    plt.pause(0.001) 

plt.show()

