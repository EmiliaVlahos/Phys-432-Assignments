#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phys 432; Assignment 4 Code
Hydro Solver for Adiabatic Shocks
@author: emiliavlahos

Created on Fri Mar 22 11:55:14 2024

"""

import numpy as np
import matplotlib.pyplot as plt

# Set up the grid, time and grid spacing, and the sound speed squared
n = 1000
x, dx = np.linspace(0, 1, n, retstep=True)
nsteps = 3000
t, dt  = np.linspace(0, 0.15, nsteps, retstep=True)
gamma = 5/3 # Adiabatic index

f1 = np.ones(n) # rho
f2 = np.zeros(n) # rho x u
f3 = np.ones(n) # rho x e_tot

A = 75 # amplitude
std = 0.05
mean = 0.5
f3 += A * np.exp(-(1/2)*((x-mean)/std)**2)


P = (gamma - 1) * (f3/f1 - (1/2) * (f2/f1)**2) * f1
cs = np.sqrt(gamma * P / f1)
mach = f2/f1 / cs

#%%
def advect_step(q, u, dt, dx):
    
    # calculate fluxes
    j = np.zeros(n-1)
    
    for i in range(n-1):
        if u[i] > 0.0:
            j[i] = q[i] * u[i]
        else:
            j[i] =  q[i+1] * u[i]
            
    # do the update
    q[1:-1] -= (dt/dx) * (j[1:] - j[:-1])
    
    q[0] -= (dt/dx) * j[0]
    q[-1] +=  (dt/dx) * j[-1]
    
    return q


#%%
# Initialize plots

plt.ion()
fig, (ax1, ax2) = plt.subplots(2,1, sharex=True)
fig.suptitle('Adiabatic Shock Propagation')
x1, = ax1.plot(x, f1, 'ro', ms=1)
x2, = ax2.plot(x, mach, "ko", ms=1)

ax1.set_xlim([0, 1])
ax1.set_ylim([1-5, 1+5])

ax1.set_ylabel('Density')
ax2.set_ylabel("Mach Number")
ax2.set_ylim([-5, 5])

ax2.set_xlabel('x')

fig.canvas.draw()

#%%

# do the iterations
for count in range(nsteps):
    
    # compute advection velocity
    u = (1/2) * ((f2[:-1]/f1[:-1]) + (f2[1:]/f1[1:]))
    
    # advect density, then momentum
    f1 = advect_step(f1, u, dt, dx)
    f2 = advect_step(f2, u, dt, dx)
    
    # compute pressure and apply the pressure gradient force to the momentum equation.
    P = (gamma - 1) * (f3/f1 - (1/2) * (f2/f1)**2) * f1
    f2[1:-1] -= dt * (P[2:] - P[:-2]) / (2.0 * dx)
    f2[0] -= dt * (P[1] - P[0]) / (2.0 * dx)
    f2[-1] -= dt * (P[-1] - P[-2]) / (2.0 * dx)
    
    # re-calculate the advection velocities
    u = (1/2) * ((f2[:-1]/f1[:-1]) + (f2[1:]/f1[1:]))
    
    # advect energy
    f3 = advect_step(f3, u, dt, dx)
    
    # compute pressure and apply the pressure gradient force to the momentum equation.
    P = (gamma - 1) * (f3/f1 - (f2/f1)**2 / 2) * f1
    
    f3[1:-1] -= dt * (P[2:]*f2[2:]/f1[2:] - P[:-2]*f2[:-2]/f1[:-2]) / (2.0 * dx)
    f3[0] -= dt * (P[1]*f2[1]/f1[1] - P[0]*f2[0]/f1[0]) / (2.0 * dx)
    f3[-1] -= dt * (P[-1]*f2[-1]/f1[-1] - P[-2]*f2[-2]/f1[-2]) / (2.0 * dx)
    
    # re-calculate pressure and sound speed
    P = (gamma - 1) * (f3/f1 - (1/2) * (f2/f1)**2) * f1
    cs = np.sqrt(gamma * P / f1)
    mach = f2/f1 / cs
    
    # update the plot
    x1.set_ydata(f1)
    x2.set_ydata(mach)
    fig.canvas.draw()
    plt.pause(0.001)

