#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 15 22:09:35 2019

@author: armitage
"""
import numpy as np

def interpolate_model(y,layers,N,dz,lz,nz):
    
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
    
    zz =[]
    zz_ = 0
    dz = lz/nz
    k = []
    i = 1
    
    for j in range(nz):
        zz.append(zz_)
        if zz_ < z[i]:
            k.append(x[i])
        else:
            i += 2
            k.append(x[i])
        zz_ += dz
    
    zz = np.array(zz)
    k = np.array(k)
    
    return zz,k

def advection_1d(var,vel,dz,dt,NN):

    """
    A TVD scheme for the advection of fluid 
    
    """
    
    # check cfl for advection and diffusion
    cfl = 0.5
    dtc = cfl*dz/(np.max(np.abs(vel)))
    dt = np.min([dt, dtc])
    
    # ghost cells required of my artificial boundary conditions:
    # non-reflecting Neumann type boundary conditions are implemented
    vargh = np.insert(var, [0,NN], [var[0],var[-1]])  
    velgh = np.insert(vel, [0,NN], [vel[0],vel[-1]])
        
    theta = np.ones(NN+2)
    theta[np.where(velgh<0)] = -1
    
    # calculate slopes for the flux limiter (phi)
    TVD_r = vargh[1:]
    TVD_r2 = np.insert(vargh[2:],np.shape(vargh[2:])[0],vargh[-1])
    TVD_m = vargh[:-1]
    TVD_l = np.insert(vargh[:-2],0,vargh[0])
            
    r_TVDup = (TVD_r2-TVD_r)/(TVD_r-TVD_m)
    r_TVDdown = (TVD_m-TVD_l)/(TVD_r-TVD_m)
    
    r_TVD = r_TVDdown
    r_TVD[np.where(theta[1:]<0)] = r_TVDup[np.where(theta[1:]<0)]
    r_TVD[np.where(np.diff(TVD_m)==0)] = 1
    r_TVD[0] = 1
    r_TVD[-1] = 1
                   
    # define Flux Limiter function (Van Leer)
    phi = (r_TVD + np.abs(r_TVD))/(1 + np.abs(r_TVD))
    phi_r = phi[1:]
    phi_l = phi[:-1]
    
    # think about my ghost cells
    TVD_r = vargh[2:]
    TVD_l = vargh[:-2]
    
    # compute fluxes for TVD
    F_rl = .5*((1+theta[1:-1])*vel*var + (1-theta[1:-1])*vel*TVD_r)
    F_rh = .5*vel*(var + TVD_r) - .5*vel*vel*dt/dz*(TVD_r-var)
    
    F_ll = .5*((1+theta[1:-1])*vel*TVD_l + (1-theta[1:-1])*vel*var)
    F_lh = .5*vel*(TVD_l+var) - .5*vel*vel*dt/dz*(var-TVD_l)
    
    # do the job
    F_right = F_rl + phi_r*(F_rh - F_rl)
    F_left  = F_ll + phi_l*(F_lh - F_ll)
    
    vari = var - dt*(F_right-F_left)/dz
        
    # might want to add a check for imaginary numbers...
    
    return vari,dt