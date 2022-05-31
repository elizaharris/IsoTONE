#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 15:07:54 2020

@author: elizaharris
"""

import numpy as np

def atmos_model(Nsummary,MW_air = 28.9647,MW_N2O = 44,
                T_S = np.array([5.37e17,1.26e17]),m_trop = 0.85*1.77e20,m_strat = 0.15*1.77e20,ltPI = [123,10],ltPD = [116,9],
                c_prea = [250,7.5],d15_prea = [8.9,2.0],d18_prea = [46.1,2.0],SP_prea = [19.05,2.0],
                F_ocean = [4,1],d15_ocean = [5.1,1.9],d18_ocean = [44.8,3.6],SP_ocean = [15.8,7.1]) :
    
    # Get inputs from d15Nsoil model
    d15_terr = Nsummary[0,5] 
    SP_terr = Nsummary[0,6] 
    d15_anth = Nsummary[:,5] 
    SP_anth = Nsummary[:,6] 
    
    # model timeframe
    t0 = 1850 # Year to start the anthropocene
    t = Nsummary[:,0] # model run years
    
    # get lifetime for modelled years
    T_Sm = T_S*1000/MW_air
    lt_t = np.zeros((len(t),1))
    lt_t[t<=t0,0] = ltPI[0] # PI before anthropocene
    lt_t[t>t0,0] = np.interp(t[t>t0],np.array([t0,2010]),np.array([ltPI[0],ltPD[0]])) # interpolate once anthropocene starts
    
    #%% Set up steady state for preindustrial time period
    
    ### currently the Bai model N2O fluxes give a F_terr which is lower than the Sowers model best predictions,
    # necessitating an increased Focean and decreased strat trop exchange to achieve steady state
    
    # assume steady state to find the conc of N2O in the stratosphere in the PI using eq. 3
    #T_Sm[0]*(c_prea[0]-240)*1e-9*28/1e12 # LHS: inputs to strat from trop (strat = 240 in this)
    #(c_prea[0]*m_trop*1e-9*28 + 240*m_strat*1e-9*28)/lt_t[0]/1e12 # RHS: total sink (strat = 240 in this)
    cstrat_prea = -(c_prea[0]*m_trop*1e-9*28 - T_Sm[0]*c_prea[0]*1e-9*28*lt_t[0])/(T_Sm[0]*1e-9*28*lt_t[0] + m_strat*1e-9*28)  # solve
    
    # use Bai model data for terrestrial flux
    F_terr_B = Nsummary[0,1] 
    # assume steady state to find F_terr using eq. 2
    F_terr = - F_ocean[0] - T_Sm[0]*(cstrat_prea - c_prea[0])*1e-9*28/1e12 
    diff = F_terr_B - F_terr
    scale = diff/4 if diff/4 < 1 else 1
    F_ocean_new = F_ocean[0] - F_ocean[1]*scale
    T_Sm_new = T_Sm[0] + (T_S[1]*scale)*1000/MW_air
    F_terr = - F_ocean_new - T_Sm_new*(cstrat_prea - c_prea[0])*1e-9*28/1e12
    # iterate other params to best match terr flux estimates
    n_max = 20; n = 0
    match_crit = 0.0001
    cstrat_prea_new = cstrat_prea
    while abs(F_terr_B - F_terr) > match_crit:
        diff = F_terr_B - F_terr; 
        scale = diff/10 if diff/10 < 0.2 else 0.2
        F_ocean_new = F_ocean_new - F_ocean[1]*scale
        if F_ocean_new < F_ocean[0] - 2*F_ocean[1]: # Reverse the adjustment on F_ocean if it gets too low
            F_ocean_new = F_ocean_new + F_ocean[1]*scale
        T_Sm_new = T_Sm_new + (T_S[1]*scale)*1000/MW_air
        if T_Sm_new < T_Sm[0] - 3*T_Sm[1]: # Reverse the adjustment on T_S if it gets too low
            T_Sm_new = T_Sm_new - (T_S[1]*scale)*1000/MW_air
        F_terr = - F_ocean_new - T_Sm_new*(cstrat_prea_new - c_prea[0])*1e-9*28/1e12
        # find cstrat_prea again
        cstrat_prea_new = -(c_prea[0]*m_trop*1e-9*28 - T_Sm_new*c_prea[0]*1e-9*28*lt_t[0])/(T_Sm_new*1e-9*28*lt_t[0] + m_strat*1e-9*28) 
        n = n+1
        if n==n_max : 
            match_crit = match_crit*10
            n=0
    # look how different they are... (std devs difference)
    F_ocean_offset = (F_ocean_new-F_ocean[0])/F_ocean[1]; #print(F_ocean_offset)
    T_Sm_offset = (T_Sm_new-T_Sm[0])/(T_S[1]*1000/MW_air)
    
    ### add isotopes
    # note: tried to model isotopes separately... this doesn't seem to work for the moment but just propagates terms from trop 
    def preanth_del(del_ocean,del_prea,del_terr) :
        # Find strat fractionation and preanth del value assuming steady state
        os = F_ocean_new/(c_prea[0]*m_trop*1e-9*28*1e-12)*(del_ocean - del_prea) + F_terr/(c_prea[0]*m_trop*1e-9*28*1e-12)*(del_terr - del_prea)
        fs = T_Sm_new*cstrat_prea*1e-9*28*1e-12 / (c_prea[0]*m_trop*1e-9*28*1e-12)
        res1 = -os/fs+del_prea # del value of the preanth stratosphere, eq. 6
        ts = T_Sm_new*c_prea[0]*1e-9*28*1e-12 / (cstrat_prea*m_strat*1e-9*28*1e-12) * (del_prea-res1)
        fs = (c_prea[0]*m_trop*1e-9*28 + cstrat_prea*m_strat*1e-9*28)/lt_t[0]/1e12
        res2 = ts/(fs/(cstrat_prea[0]*m_strat*1e-9*28*1e-12)) # fractionation of the sink for del
        return [res1,res2]
      
    # d15N
    res = preanth_del(del_ocean = d15_ocean[0],del_prea = d15_prea[0],del_terr = d15_terr)
    d15strat_prea = res[0]
    E15_sink = res[1]
    
    # SP
    res = preanth_del(del_ocean = SP_ocean[0],del_prea = SP_prea[0],del_terr = SP_terr)
    SPstrat_prea = res[0]
    ESP_sink = res[1]
    
    #%% Move though time period
    
    ### WORK FROM HERE! deal with the anth source signature issue, and the model instability in the current config...
    
    # Concentration array
    results = np.zeros((len(t),6))  
    # colnames(results) = c("year","Fanth TgN y-1","c_trop","c_strat","M_trop TgN","rate.change_trop")
    results[:,0] = t
    results[:,1] = Nsummary[:,1]-Nsummary[0,1] # total flux - first year flux - Done with exp factor in the Bai model!
    results[0,2:6] = [c_prea[0],cstrat_prea,c_prea[0]*m_trop*1e-9*28,0] # prea details
    
    for n in range(1,len(t)) :
        rateofchange = F_ocean_new + F_terr + results[n,1] + T_Sm_new*(results[n-1,3]-results[n-1,2])*1e-9*28/1e12
        # rate of change of the tropospheric concentration due to anthrop in TgN/y; convert to ppb/y:
        results[n,5] = rateofchange*1e12/28/m_trop*1e9
        results[n,2] = results[n-1,2]+results[n,5]
        results[n,4] = results[n,2]*m_trop*1e-9*28
        FTtoS = T_Sm_new*(results[n-1,2] - results[n-1,3])*1e-9*28/1e12 # flux from the trop to the strat
        Fsink = (results[n-1,2]*m_trop*1e-9*28 + results[n-1,3]*m_strat*1e-9*28)/lt_t[n-1]/1e12
        dSdt = (FTtoS - Fsink)*1e12/28/m_strat*1e9
        results[n,3] = results[n-1,3]+dSdt
      
    #%% d15N
    
    results_d15 = np.zeros((len(t),3))  # d15 trop, d15 strat, rate of change of trop
    results_d15[0,0:3] = [d15_prea[0],d15strat_prea,0]
    for n in range(1,len(t)) :
        os = F_ocean_new/(results[n-1,2]*m_trop*1e-9*28*1e-12)*(d15_ocean[0] - results_d15[n-1,0])
        ts = F_terr/(results[n-1,2]*m_trop*1e-9*28*1e-12)*(d15_terr - results_d15[n-1,0])
        ans = results[n,1]/(results[n-1,2]*m_trop*1e-9*28*1e-12)*(d15_anth[n] - results_d15[n-1,0])
        fs = T_Sm_new*results[n-1,3]*1e-9*28*1e-12 / (results[n-1,2]*m_trop*1e-9*28*1e-12) * (results_d15[n-1,1] - results_d15[n-1,0])
        results_d15[n,2] = os+ts+ans+fs
        results_d15[n,0] = results_d15[n-1,0]+results_d15[n,2] # tropospheric d15N value
        ddSdt_trop = T_Sm_new*results[n-1,2]*1e-9*28*1e-12 / (results[n-1,3]*m_strat*1e-9*28*1e-12) * (results_d15[n-1,0] - results_d15[n-1,1])
        ddSdt_sink = (results[n-1,2]*m_trop*1e-9*28 + results[n-1,3]*m_strat*1e-9*28)/lt_t[n-1]/1e12/ (results[n-1,3]*m_strat*1e-9*28*1e-12) *E15_sink
        ddSdt = ddSdt_trop - ddSdt_sink
        results_d15[n,1] = results_d15[n-1,1]+ddSdt # tropospheric d15N value

    #%% SP
    
    results_SP = np.zeros((len(t),3))  # d15 trop, d15 strat, rate of change of trop
    results_SP[0,0:3] = [SP_prea[0],SPstrat_prea,0]
    for n in range(1,len(t)) :
        os = F_ocean_new/(results[n-1,2]*m_trop*1e-9*28*1e-12)*(SP_ocean[0] - results_SP[n-1,0])
        ts = F_terr/(results[n-1,2]*m_trop*1e-9*28*1e-12)*(SP_terr - results_SP[n-1,0])
        ans = results[n,1]/(results[n-1,2]*m_trop*1e-9*28*1e-12)*(SP_anth[n] - results_SP[n-1,0])
        fs = T_Sm_new*results[n-1,3]*1e-9*28*1e-12 / (results[n-1,2]*m_trop*1e-9*28*1e-12) * (results_SP[n-1,1] - results_SP[n-1,0])
        results_SP[n,2] = os+ts+ans+fs
        results_SP[n,0] = results_SP[n-1,0]+results_SP[n,2] # tropospheric d15N value
        ddSdt_trop = T_Sm_new*results[n-1,2]*1e-9*28*1e-12 / (results[n-1,3]*m_strat*1e-9*28*1e-12) * (results_SP[n-1,0] - results_SP[n-1,1])
        ddSdt_sink = (results[n-1,2]*m_trop*1e-9*28 + results[n-1,3]*m_strat*1e-9*28)/lt_t[n-1]/1e12/ (results[n-1,3]*m_strat*1e-9*28*1e-12) *ESP_sink
        ddSdt = ddSdt_trop - ddSdt_sink
        results_SP[n,1] = results_SP[n-1,1]+ddSdt # tropospheric d15N value
        
    full_res = np.hstack((results,results_d15,results_SP))
    return(full_res,T_Sm_new,F_ocean_new)
