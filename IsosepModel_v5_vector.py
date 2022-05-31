#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 10:11:39 2020

@author: elizaharris
"""
# new = input an N conc for the grid cell

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math as math

#d15Nsoil_vec = d15N_vec; fignum=100; N_processes = N_processes_vec; fNH3_vec=fNH3_vec

#%% Function to iteratively solve for f_gas to match d15Nsoil

def IsoModel(d15Nsoil_vec,N_processes,fNH3_vec,fignum=np.nan,d15N_inat = -1.5,E_NH3 = -17.9,E_L	= [-1,0,5],E_nit = [-56.6,7.3],E_denit = [-31.3,6.1],E_denit_N2O = [-14.9,6.7],
             E_red = [-6.6,2.7],E_SP_nit = [29.9,2.9],E_SP_denit = [-1.6,3.0],E_SP_red = [-5,-8,-2],f_soilpoolloss=0.5) :
    
    # details for timesteps
    spin_up = 10 # time to reach steady state
    mtimes = range(0,spin_up)
    soil_NO3 = np.zeros((spin_up,3,len(d15Nsoil_vec))) # 3 columns: 14N, 15N, d15N
    soil_NO3[0,0,:] = 1 # define a soil pool size to begin initialization (only changes the length of mtimes until equilibration)
    soil_NO3[0,1,:] = soil_NO3[0,0,:] # initial del of 0 permil so 15N = 14N
    soil_NO3_steady = d15Nsoil_vec.copy()*0-1000
    N2O = np.zeros((spin_up,4,len(d15Nsoil_vec))) # 3 columns: 14N, 15Na, 15Nb, d15N
    N2O[0,0,:] = 0.1 # define a "soil N2O" pool size to begin initialization (only changes the length of mtimes until equilibration)
    N2O[0,1,:] = N2O[0,0,:] # initial del of 0 permil
    N2O[0,2,:] = N2O[0,0,:] # initial SP of 0 permil
    
    # Start with natural inputs
    k_inat = d15Nsoil_vec.copy()*0+1 # natural input rate (currently unitless) for 14N to the soil system
    k15_inat = k_inat*(d15N_inat/1000+1) # input rate for 15N to the soil system
    
    # Steady state for natural inputs with outputs
    k_Rtot = k_inat # removal rate is equal to input rate
    
    # Removal by volatilization
    k_NH3 = k_Rtot*fNH3_vec # volatilization from Bai et al.
    k15_NH3 = k_NH3*(E_NH3/1000+1) # 15N removal rate with volatilization frac
       
    # Gaseous loss params
    E_G = E_nit[0]*N_processes[:,4]+E_denit[0]*N_processes[:,3] # frac factor for gas losses
    k_G = d15Nsoil_vec.copy()*0+0.1 # set an initial k_G for each
    for its in range(0,4): # run 5 iterations; 3 are sufficient for almost all d15N values anyway...
        # Removal by microbial gas production
        if its > 0 : k_G = k_G-(d15Nsoil_vec-soil_NO3_steady)/E_G
        k_G = k_G.clip(0,1)
        k15_G = k_G*(E_G/1000+1) # 15N removal rate with gas loss frac
        
        # Removal by leaching
        k_L = k_Rtot - k_NH3 - k_G # let all inputs be removed
        k_L = k_L.clip(0,1)
        k15_L = k_L*(E_L[0]/1000+1) # 15N removal rate with leaching frac
        
        # N2O production by nitrification
        k_G_nN2O = k_G*N_processes[:,4]*N_processes[:,0]
        k15_G_nN2O = k_G_nN2O*(E_nit[0]/1000+1)
        k15a_G_nN2O = k15_G_nN2O*(0.5*E_SP_nit[0]/1000+1)
        k15b_G_nN2O = k15_G_nN2O*(0.5*-E_SP_nit[0]/1000+1)
        
        # N2O production by denitrification
        k_G_dN2O = k_G*N_processes[:,3]*N_processes[:,0]
        k15_G_dN2O = k_G_dN2O*((E_denit[0]+E_denit_N2O[0])/1000+1)
        k15a_G_dN2O = k15_G_dN2O*(0.5*E_SP_denit[0]/1000+1)
        k15b_G_dN2O = k15_G_dN2O*(0.5*-E_SP_denit[0]/1000+1)
        
        # N2O consumption by reduction during denitrification
        k_red = k_G*N_processes[:,5]*N_processes[:,0]
        k15_red = k_red*(E_red[0]/1000+1) # 15N removal rate with leaching frac
        k15a_red = k15_red*(0.5*E_SP_red[0]/1000+1)
        k15b_red = k15_red*(0.5*-E_SP_red[0]/1000+1)
        
        ### Run spin up to achieve steady state
        for n in range(1,spin_up) :
            # add inputs
            soil_NO3[n,0,:] = soil_NO3[n-1,0,:]+k_inat
            soil_NO3[n,1,:] = soil_NO3[n-1,1,:]+k15_inat
            # leaching
            soil_NO3[n,0,:] = soil_NO3[n,0,:]-k_L*soil_NO3[n-1,0,:]/soil_NO3[0,0,:] # adjust removal rate for pool size (relative pool size from prev. step compared to initial)
            soil_NO3[n,1,:] = soil_NO3[n,1,:]-k15_L*soil_NO3[n-1,1,:]/soil_NO3[0,1,:]
            # volatilization
            soil_NO3[n,0,:] = soil_NO3[n,0,:]-k_NH3*soil_NO3[n-1,0,:]/soil_NO3[0,0,:] 
            soil_NO3[n,1,:] = soil_NO3[n,1,:]-k15_NH3*soil_NO3[n-1,1,:]/soil_NO3[0,1,:]
            # gas production
            soil_NO3[n,0,:] = soil_NO3[n,0,:]-k_G*soil_NO3[n-1,0,:]/soil_NO3[0,0,:]
            soil_NO3[n,1,:] = soil_NO3[n,1,:]-k15_G*soil_NO3[n-1,1,:]/soil_NO3[0,1,:]
            
            # nitrification and denitrification: produce N2O
            N2O[n,0,:] = N2O[n-1,0,:]+(k_G_nN2O+k_G_dN2O)*soil_NO3[n-1,0,:]/soil_NO3[0,0,:]
            N2O[n,1,:] = N2O[n-1,1,:]+(k15a_G_nN2O+k15a_G_dN2O)*soil_NO3[n-1,1,:]/soil_NO3[0,1,:] # alpha
            N2O[n,2,:] = N2O[n-1,2,:]+(k15b_G_nN2O+k15b_G_dN2O)*soil_NO3[n-1,1,:]/soil_NO3[0,1,:] # beta
            # reduction
            N2O[n,0,:] = N2O[n,0,:]-k_red*N2O[n-1,0,:]/N2O[0,0,:] 
            N2O[n,1,:] = N2O[n,1,:]-k15a_red*N2O[n-1,1,:]/N2O[0,1,:]
            N2O[n,2,:] = N2O[n,2,:]-k15b_red*N2O[n-1,2,:]/N2O[0,2,:]
            # remove soil pool to atm without fractionation
            N2O[n,0,:] = N2O[n,0,:]*f_soilpoolloss
            N2O[n,1,:] = N2O[n,1,:]*f_soilpoolloss
            N2O[n,2,:] = N2O[n,2,:]*f_soilpoolloss
            
        soil_NO3[:,2,:] = (soil_NO3[:,1,:]/soil_NO3[:,0,:]-1)*1000 # del values
        N2O[:,3,:] = ((N2O[:,1,:]+N2O[:,2,:])/2/N2O[:,0,:]-1)*1000 
        SP = (N2O[:,1,:]/N2O[:,0,:]-1)*1000 - (N2O[:,2,:]/N2O[:,0,:]-1)*1000
        soil_NO3_steady = soil_NO3[spin_up-1,2,:]    
        
    # Make a final plot
    if ~np.isnan(fignum):
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex='col', sharey='row')
        ax1.plot(mtimes,soil_NO3[:,0,fignum])
        ax1.set_ylabel("soil NO3 (14N)")
        ax2.plot(mtimes,soil_NO3[:,2,fignum])
        ax2.set_ylabel("soil NO3 d15N")
        fig.show()
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex='col', sharey='row')
        ax1.plot(mtimes,N2O[:,0,fignum])
        ax1.set_ylabel("N2O (14N)")
        ax2.plot(mtimes,N2O[:,3,fignum])
        ax2.set_ylabel("N2O d15N")
        ax3.plot(mtimes,SP[:,fignum])
        ax3.set_ylabel("N2O SP")
        fig.show()
    
    # Produce outputs: matrix with rows = d15N values and columns =
    # 0:5   k_L k_NH3 k_G soil_d15NNO3_final N2O_d15N N2O_SP
    result = np.array(np.vstack((k_L,k_NH3,k_G,soil_NO3_steady,N2O[spin_up-1,3,:],SP[spin_up-1,:]))).transpose()
    # check for convergence
    diffs = abs(d15Nsoil_vec-soil_NO3_steady)
    diffs[np.where(np.isnan(diffs) | np.isinf(diffs))] = 9999
    noconverg = np.where(diffs>0.1) 
    result[noconverg,:] = np.nan
    if sum(sum(noconverg))>0:
        print("model does not converge for d15Nsoil < "+str(np.max(d15Nsoil_vec[noconverg])))
    
    return result

#%% Note: frac factors from Bai et al. (+ve means 14N reacts faster)
E_L_Bai = 1 # leaching losses
E_NH3_Bai = 29 # amm vol losses
E_G_Bai = 16 # mean for gaseous losses (no diff for nitrif/denitrif)

