#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Q3: Vortex interaction: numerical exercise
Simulates the leapfrogging behaviour of two interacting vortex rings

Created on Mon Feb 12 2024

@author: Emilia Vlahos
@collab: Julien Hacot-Slonosky, Spencer Geddes, Maryn Askew, Guilherme Caumo
"""

import numpy as np
import matplotlib.pyplot as pl


dt = 2
Nsteps = 250

## Setting up initial conditions of vortex rings
# Vortex centres in x and y
y_v = np.array([20, -20, 20, -20], dtype="f")
x_v = np.array([-200, -200, -150, -150], dtype="f")
# Circulation
k_v = np.array([15, -15, 15, -15]) 

## Initialize the plot
pl.ion()
fig, ax = pl.subplots(1,1)
# mark the initial positions of vortices
p, = ax.plot(x_v, y_v, 'k+', markersize=10) 

## draw the initial velocity streamlines
# intialize cartesian grid to animate over
ngrid = 250
Y, X = np.mgrid[-ngrid:ngrid:360j, -ngrid:ngrid:360j] 

vel_x = np.zeros(np.shape(Y)) #this holds x-velocity over velocity feild
vel_y = np.zeros(np.shape(Y)) #this holds y-velocity over velocity feild

# masking radius for better visualization of the vortex centres
r_mask = 5 # within this mask, you will not plot any streamline 

# sum over the contributions from each vortex on the total velocity feild
for i in range(len(x_v)): #looping over each vortex
    r = np.sqrt((X - x_v[i])**2 + (Y - y_v[i])**2)
    
    # x and y components given by the relation between cylindrical and cartesian unit vectors
    vel_x -= k_v[i] * (Y - y_v[i]) / r**2
    vel_y += k_v[i] * (X - x_v[i]) / r**2
    
    # set the masking area to NaN
    vel_x[r < r_mask] = np.nan
    vel_y[r < r_mask] = np.nan

# set up the boundaries of the simulation box
ax.set_xlim([-ngrid, ngrid])
ax.set_ylim([-ngrid, ngrid])

# initial plot of the streamlines
ax.streamplot(X, Y, vel_x, vel_y, density=[0.5, 0.5]) 
fig.canvas.draw()


# Evolve vortices over Nsteps

count = 0 #begin step count
while count < Nsteps:
    
    ## Compute and update advection velocity
    vel_x = np.zeros(np.shape(x_v)) #this holds x-velocity at vortex positions
    vel_y = np.zeros(np.shape(x_v)) #this holds y-velocity at vortex positions
    
    # sum over the contributions from each vortex on the velocity feild at the other vortices
    for i in range(len(x_v)):

        r = np.sqrt((x_v - x_v[i])**2 + (y_v - y_v[i])**2)

        # exclude contribution from vortex[i] on itself (r=0)
        vel_x[r!=0] -= k_v[i] * (y_v[r!=0] - y_v[i]) / r[r!=0]**2
        vel_y[r!=0] += k_v[i] * (x_v[r!=0] - x_v[i]) / r[r!=0]**2
        
    
    # update the positions of vortices
    x_v += vel_x * dt
    y_v += vel_y * dt
    
    # re-initialize the total velocity field
    vel_x = np.zeros(np.shape(Y)) #this holds x-velocity over velocity feild
    vel_y = np.zeros(np.shape(Y)) #this holds y-velocity over velocity feild

    # update the streamlines and masking
    for i in range(len(x_v)): 
        r = np.sqrt((X - x_v[i])**2 + (Y - y_v[i])**2)

        vel_x -= k_v[i] * (Y - y_v[i]) / r**2
        vel_y += k_v[i] * (X - x_v[i]) / r**2

        vel_x[r < r_mask] = np.nan
        vel_y[r < r_mask] = np.nan

    
    ## update plot
    # clear out the previous streamlines
    for collection in ax.collections:
        collection.remove()
    for patch in ax.patches:
        patch.remove()

    # update position of vortices
    p.set_xdata(x_v)
    p.set_ydata(y_v)

    # update velocity feild
    ax.streamplot(X, Y, vel_x, vel_y, density=[0.8, 0.8])

    # plot
    fig.canvas.draw()
    pl.pause(0.001) 
    
    count += 1 #update step count