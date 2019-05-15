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
    
    velgh = []
    velgh.append(vel[0])
    velgh.append(vel)
    velgh.append(vel[NN-1])
    
    theta = np.ones(NN+2,1)
    theta[velgh<0] = -1
    
    # calculate slopes for the flux limiter (phi)
    TVD_r = vargh[1:NN-1]
    TVD_r2 = []
    TVD_r2.append(vargh[2:NN-1])
    TVD_r2.append(vargh[NN-1])
    TVD_m = vargh[0:NN-2]
    TVD_l = []
    TVD_l.append(vargh(0))
    TVD_l = [vargh(1); vargh(1:end-2)];
    
    r_TVDup   = (TVD_r2-TVD_r)./(TVD_r-TVD_m);
    r_TVDdown = (TVD_m-TVD_l)./(TVD_r-TVD_m);
    
    r_TVD = r_TVDdown;
    r_TVD(theta(2:end)<0) = r_TVDup(theta(2:end)<0);
    r_TVD(diff(TVD_m)==0) = 1;
    r_TVD(1) = 1;
    r_TVD(end) = 1;
    
%     l_TVDup    = (TVD_m-TVD_l)./(TVD_r-TVD_m);
%     l_TVDdown  = (TVD_r2-TVD_r)./(TVD_r-TVD_m);
%     
%     l_TVD = l_TVDdown;
%     l_TVD(theta(1:end-1)<0) = l_TVDup(theta(1:end-1)<0);
%     l_TVD(diff(TVD_r)==0) = 1;
               
    % define Flux Limiter function (Van Leer)
    phi = (r_TVD + abs(r_TVD))./(1 + abs(r_TVD));
%    phi_l = (l_TVD + abs(l_TVD))./(1 + abs(l_TVD));
    phi_r = phi(2:end);
    phi_l = phi(1:end-1);
    
    % think about my ghost cells
    TVD_r = vargh(3:end);
    TVD_l = vargh(1:end-2);
    
    % compute fluxes for TVD
    F_rl = .5*((1+theta(2:end-1)).*vel.*var + (1-theta(2:end-1)).*vel.*TVD_r);
    F_rh = .5*vel.*(var + TVD_r) - .5*vel.*vel.*dt/dz.*(TVD_r-var);
    
    F_ll = .5*((1+theta(2:end-1)).*vel.*TVD_l + (1-theta(2:end-1)).*vel.*var);
    F_lh = .5*vel.*(TVD_l+var) - .5*vel.*vel.*dt/dz.*(var-TVD_l);
    
    % do the job
    F_right = F_rl + phi_r.*(F_rh - F_rl);
    F_left  = F_ll + phi_l.*(F_lh - F_ll);
    
    vari = var - dt*(F_right-F_left)/dz;
    
    
    if any(~isreal(vari))
        disp('imaginary number = time step problem?')
        params.iterate = 0;
        return
    end

    return vari
