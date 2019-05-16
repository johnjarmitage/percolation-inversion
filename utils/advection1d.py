#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 15 22:09:35 2019

@author: armitage
"""
import numpy as np

def advection_1d(x,var,vel,w,dz,dt,NN):

    """
    A TVD scheme for the advection of fluid 
    
    """
        
    # ghost cells required of my artificial boundary conditions:
    # non-reflecting Neumann type boundary conditions are implemented
    vargh = []
    vargh.append(var[0])
    vargh.append(var)
    vargh.append(var[NN-1])
    vargh = np.array(vargh)
    
    velgh = []
    velgh.append(vel[0])
    velgh.append(vel)
    velgh.append(vel[NN-1])
    velgh = np.array(velgh)
    
    theta = np.ones(NN+2,1)
    theta[np.where(velgh<0)] = -1
    
    # calculate slopes for the flux limiter (phi)
    TVD_r = vargh[1:NN-1]
    TVD_r2 = []
    TVD_r2.append(vargh[2:NN-1])
    TVD_r2.append(vargh[NN-1])
    TVD_r2 = np.array(TVD_r2)
    TVD_m = vargh[:NN-2]
    TVD_l = []
    TVD_l.append(vargh[0])
    TVD_l.append(vargh[:NN-3])
    TVD_l = np.array(TVD_l)
        
    r_TVDup = (TVD_r2-TVD_r)/(TVD_r-TVD_m)
    r_TVDdown = (TVD_m-TVD_l)/(TVD_r-TVD_m)
    
    r_TVD = r_TVDdown
    r_TVD[np.where(theta[:NN-2]<0)] = r_TVDup[np.where(theta[1:NN-1]<0)]
    r_TVD[np.where(np.diff(TVD_m)==0)] = 1
    r_TVD[0] = 1
    r_TVD[NN-1] = 1
                   
    # define Flux Limiter function (Van Leer)
    phi = (r_TVD + np.abs(r_TVD))/(1 + np.abs(r_TVD))
    phi_r = phi[1:NN-1]
    phi_l = phi[0:NN-2]
    
    # think about my ghost cells
    TVD_r = vargh[2:NN-1]
    TVD_l = vargh[0:NN-3]
    
    # compute fluxes for TVD
    F_rl = .5*((1+theta[1:NN-2])*vel*var + (1-theta[1:NN-2])*vel*TVD_r)
    F_rh = .5*vel*(var + TVD_r) - .5*vel*vel*dt/dz*(TVD_r-var)
    
    F_ll = .5*((1+theta[1:NN-2])*vel*TVD_l + (1-theta[1:NN-2])*vel*var)
    F_lh = .5*vel*(TVD_l+var) - .5*vel*vel*dt/dz*(var-TVD_l)
    
    # do the job
    F_right = F_rl + phi_r*(F_rh - F_rl)
    F_left  = F_ll + phi_l*(F_lh - F_ll)
    
    vari = var - dt*(F_right-F_left)/dz
        
    # might want to add a check for imaginary numbers...

    return vari

def interpolate_to_forward_model(y,layers,N,dz,lz,nz):
    """
    Interpolate the layer model onto the forward model grid
    
    """
    for j in range(N):
        x = []
        z = []
        z_ = 0
        for i in range(layers):
            z.append(z_)
            z_ += dz
            x.append(y.numpy()[j][i])
            z.append(z_)
            x.append(y.numpy()[j][i])

    zz = []
    k = []
    zz_ = 0
    ddz = lz/nz
    i = 1
    for j in range(nz):
        if zz_ < z[i]:
            k.append(k[i-1])
        
