#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Tue Mar  3 09:09:23 2020

Combined soil and atm model for optimisation of parameters

Main to do list: 
    - add comparison of post_obs and finish assessment figures and so on for the new flux data
    - check model and obs unc for flux data looks okay
    - once finalised, run lots more its
    - look at final EFs per climate zone and the amount of N input into each for different year ranges: is mean EF increasing?

@author: elizaharris
"""
%reset

#%% Preamble 

# basic packages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math as math
import xarray as xr
from datetime import datetime as dt
import cartopy.crs as ccrs
import os
import netCDF4 as nc4
import time
import datetime

# my functions for this model
import parameterisations_v3 as para # partition nitrif and denitrif
import IsosepModel_v5_vector as isomodel # function to estimate N cycle based on d15N of soil
import AtmosIsoModel_function_v3 as atmodel

#%% Define some quick functions and params

# define function for plotting    
def plot_map(longi,lati,gridval,title="title",vminmax=(np.nan,np.nan),cmap="plasma") :
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1,projection=ccrs.PlateCarree())
    ax.coastlines()
    ax.set_title(title)
    if np.isnan(vminmax[0]) :
        pr = plt.contourf(longi,lati,gridval,cmap=cmap)
    else : 
        #gridval = gridval.clip(vminmax[0],vminmax[1])
        pr = plt.contourf(longi,lati,gridval,vmin=vminmax[0],vmax=vminmax[1],cmap=cmap)
    cbar = plt.colorbar(pr,fraction=0.016, pad=0.04)
    fig.show()

# function for mean by "climate zones" defined by 2 variables
def climzone_means(var1_grid, var2_grid, datavar1,datavar2,data,bins=4,plotfigs="N"):  
    if (bins % 2) != 0: 
        print("bins must be even number!! ")
    else: 
        # create bins based on mean +- standard deviations and calc bin means for data
        #var1_bins_orig = np.linspace(np.nanmean(var1_grid)-bins/2*np.nanstd(var1_grid),np.nanmean(var1_grid)+bins/2*np.nanstd(var1_grid),num=bins+1)
        #var2_bins_orig = np.linspace(0,np.nanmean(var2_grid)+bins/2*np.nanstd(var2_grid),num=bins+1)
        # create bins based on percentiles of the data
        var1_bins_orig = np.percentile(datavar1[~np.isnan(datavar1)],(0,25,50,75,100))
        var2_bins_orig = np.percentile(datavar2[~np.isnan(datavar2)],(0,25,50,75,100))
        var1_bins = var1_bins_orig.copy(); var1_bins[0] = -100000; var1_bins[-1] = +100000 # make sure edge bins should span full range
        var2_bins = var2_bins_orig.copy(); var2_bins[0] = -100000; var2_bins[-1] = +100000 # edge bins should span full range
        res = np.zeros((bins**2,7))*np.nan # var1 bin #, var1 bin mid, var2 bin #, var2 bin mid, data mean, data std, data n
        l = 0
        for n in range(0,bins):
            for i in range(0,bins):
                res[l,0] = n
                res[l,1] = (var1_bins_orig[n]+var1_bins_orig[n+1])/2
                res[l,2] = i
                res[l,3] = (var2_bins_orig[i]+var2_bins_orig[i+1])/2        
                tmp = np.where((datavar1>=var1_bins[n]) & (datavar1<=var1_bins[n+1]) & (datavar2>=var2_bins[i]) & (datavar2<=var2_bins[i+1]))
                res[l,6] = len(tmp[0])
                if len(tmp[0])>1:
                    res[l,4] = np.nanmean(data[tmp])
                    res[l,5] = np.nanstd(data[tmp])
                l = l+1
        # plot the results
        if plotfigs == "Y":
            fig, ax = plt.subplots(1,1)
            for i in range(0,bins):
                tmp = (res[:,2] == i) & ~np.isnan(res[:,4])
                ax.plot(res[tmp,1],res[tmp,4],"-",c="C"+str(i))
            tmp = (res[:,0] == 0)
            ax.legend(res[tmp,3].astype(str))  
            for i in range(0,bins):
                tmp = (res[:,2] == i) & ~np.isnan(res[:,4])
                ax.scatter(res[tmp,1],res[tmp,4],c="C"+str(i))
                ax.plot(res[tmp,1],res[tmp,4]-res[tmp,5],":",c="C"+str(i))
                ax.plot(res[tmp,1],res[tmp,4]+res[tmp,5],":",c="C"+str(i))
            for i, txt in enumerate(res[:,6]):
                ax.annotate(int(txt), (res[i,1]+0.01,res[i,4]+0.01))
            ax.set_xlabel("var1")
            ax.set_ylabel("N2O EF")
        # plot the zones if needed
        if plotfigs == "Y":
            climzones = var1_grid.copy()*np.nan
            l = 0
            for n in range(0,bins):
                for i in range(0,bins):
                    tmp = np.where((var1_grid>=var1_bins[n]) & (var1_grid<=var1_bins[n+1]) & (var2_grid>=var2_bins[i]) & (var2_grid<=var2_bins[i+1]))
                    climzones[tmp] = l       
                    l = l+1
            plot_map(LON,LAT,climzones,"Climate zones - n(bins) = "+str(bins))
        return(res)

# set time res
years = np.arange(1800,2021)

# set grid for calcs
resolution = 0.5
lat_out = np.arange(-60, 85, resolution) # min max spacing
lon_out = np.arange(-180,180, resolution)
(LON,LAT) = np.meshgrid(lon_out,lat_out)

# find the area of grid cells
lats = np.deg2rad(np.arange(-60-0.5*resolution, 85+0.5*resolution, resolution)) # Calculations needs to be in radians
r_sq = 6371000**2 # earth's radius squared
n_lats = int(360./resolution) # how many longitudes 
area_grid = r_sq*np.ones(n_lats)[:, None]*np.deg2rad(resolution)*(np.sin(lats[1:]) - np.sin(lats[:-1])) # area in m2
area_grid = np.transpose(area_grid)

#%% Import and plot the input data
    
inputs = nc4.Dataset('climate_input_data.nc','r')
C_grid = inputs.variables["soilC"][:,:]; plot_map(LON,LAT,C_grid,"soil C (kg m-2)")
MAP_grid = inputs.variables["MAPrecip"][:,:]; plot_map(LON,LAT,MAP_grid,"MAP (mm)")
MAT_grid = inputs.variables["MATemp"][:,:]; plot_map(LON,LAT,MAP_grid,"MAT (degC)",vminmax=(0,3000))
pH_grid = inputs.variables["soilpH"][:,:]; plot_map(LON,LAT,pH_grid,"pH")
N_grid = inputs.variables["soilN"][:,:]; plot_map(LON,LAT,N_grid,"N (g m-2)")
AI_grid = inputs.variables["AridIndex"][:,:]; plot_map(LON,LAT,AI_grid,"Aridity Index")
BulkD_grid = inputs.variables["BulkD"][:,:]; plot_map(LON,LAT,BulkD_grid,"Bulk Density (g cm-3)")
WFPS_grid = inputs.variables["WFPS"][:,:]*100; plot_map(LON,LAT,WFPS_grid,"WFPS (%)")
d15N_grid = inputs.variables["soild15N"][:,:]; plot_map(LON,LAT,d15N_grid,"d15N")
d15Nerr_grid = inputs.variables["soild15N_BSUnc"][:,:]; plot_map(LON,LAT,d15Nerr_grid,"d15N uncertainty")

f = nc4.Dataset('fNH3_data.nc','r')
fNH3_grid = f.variables["fNH3"][:,:]; plot_map(LON,LAT,fNH3_grid,"fNH3")
f = nc4.Dataset('EDGAR_data_extrap.nc','r')
AGS_grid = f.variables["AGS"][:,:,:]; plot_map(LON,LAT,AGS_grid[150,:,:],"EDGAR agriculture (g N2O-N/m2/year): 2000")
TRO_grid = f.variables["TRO"][:,:,:]; plot_map(LON,LAT,TRO_grid[150,:,:],"EDGAR transport (g N2O-N/m2/year): 2000")
IDE_grid = f.variables["IDE"][:,:,:]; plot_map(LON,LAT,IDE_grid[150,:,:],"EDGAR indirect (g N2O-N/m2/year): 2000")
WWT_grid = f.variables["WWT"][:,:,:]; plot_map(LON,LAT,WWT_grid[150,:,:],"EDGAR wastewater (g N2O-N/m2/year): 2000")

# import and plot the Temp anomaly data
Tanom = nc4.Dataset('Tanom_data.nc','r')
# normalise to the beginning year
Tnorm = np.zeros(Tanom.variables["Tanom"].shape)
Tstart = Tanom.variables["Tanom"][0,:,:].data
Tstart[np.isnan(Tstart)] = 0
for n in range(0,len(Tanom.variables["time"])) :
    Tnorm[n,:,:] = Tanom.variables["Tanom"][n,:,:].data - Tstart
    Tnorm[np.isnan(Tnorm)] = 0 
plot_map(LON,LAT,Tnorm[150,:,:],"Temp anomaly: "+str(Tanom.variables["time"][150]))
mean_T_rise = np.nanmean(np.nanmean(Tnorm,axis=1),axis=1)

#%% Observation data for comparison

# atmospheric time series data (combined sources)
N2O_atmos = pd.read_csv("SummAtmosData.csv", sep=',') 

# flux data from Chris Dorich, divided into climate regions
N2O_fluxes = pd.read_csv("globaln2o_sites_filled_ChrisDorich.csv")
# cut to: MAT, MAP, BulkD, C, N, pH, lat, lon, flux, flux_se, EF, fert rate for easy comparison
# fix NAs and string values also
def fix_nas(data,NA_value): # function to make different nans to true nans
    for n in range(0,len(NA_value)):
        tmp = np.where(new == NA_value[n])
        data[tmp] = np.nan
    return(data)
def fix_strings(data,chars):
    for n in range(0,len(chars)):
        tmp = np.where(np.char.find(data.astype("str"),chars[n])!=-1)
        data[tmp] = np.nan
    return(data)
N2O_flux_params = (("Temp_C","Precip_mm","Soil_BD_gcm3","Soil_SOC_perc","Soil_SON_perc","pH","Lat","Long",'N2O_kgN2O-N_ha', 'N2O_se_kgN2O-N_ha',"EF",'N_App_Rate_kgN_ha'))
N2O_fluxes_short = np.array(N2O_fluxes[N2O_flux_params[0]])
for n in range(1,len(N2O_flux_params)):
    new = np.array(N2O_fluxes[N2O_flux_params[n]])
    new = fix_nas(new,("*","na","nan"))
    new = fix_strings(new,(">","-"))
    new = new.astype(np.float)
    N2O_fluxes_short = np.vstack((N2O_fluxes_short,new))
N2O_fluxes_short = N2O_fluxes_short.transpose()
tmp = np.where(~np.isnan(N2O_fluxes_short[:,8]))
# Match the anc data to each flux point
flux_ancdata = np.zeros((N2O_fluxes.shape[0],8))*np.nan # MAT, MAP, BulkD, C, N, pH, AI, d15N
for n in range(0,N2O_fluxes.shape[0]):
    if (~np.isnan(N2O_fluxes["Lat"][n])) & (~np.isnan(N2O_fluxes["Long"][n])):
        r = np.where(np.nanmin(abs(LAT[:,0] - N2O_fluxes["Lat"][n])) == abs(LAT[:,0] - N2O_fluxes["Lat"][n])) # match flux lat long to gridcells
        c = np.where(np.nanmin(abs(LON[0,:] - N2O_fluxes["Long"][n])) == abs(LON[0,:] - N2O_fluxes["Long"][n]))
        if len(r[0])>1: r=r[0][0] # if two grid cells match lat/lon equally, take first
        if len(c[0])>1: c=c[0][0]
        flux_ancdata[n,0] = MAT_grid[r,c]
        flux_ancdata[n,1] = MAP_grid[r,c]
        flux_ancdata[n,2] = BulkD_grid[r,c]
        flux_ancdata[n,3] = C_grid[r,c]/10 # change to %
        flux_ancdata[n,4] = N_grid[r,c]/10 # change to %
        flux_ancdata[n,5] = pH_grid[r,c]
        flux_ancdata[n,6] = AI_grid[r,c]
        flux_ancdata[n,7] = d15N_grid[r,c]
N2O_fluxes_short = N2O_fluxes_short[tmp[0],:]
flux_ancdata = flux_ancdata[tmp[0],:]
# calc mean EFs by climate zone        
N2O_fluxes_zones = climzone_means(var1_grid = MAT_grid, var2_grid = MAP_grid, datavar1 = flux_ancdata[:,0],
                                  datavar2 = flux_ancdata[:,1],data = N2O_fluxes_short[:,10],bins=4,plotfigs="Y")
# final output
print(str(N2O_fluxes_short.shape[0])+" flux measurements from "+str(len(set(N2O_fluxes["Location"][tmp[0]])))+" locations")

#%% Collect input variables that may be optimised
 
# dates for model output, to compare to obs
tstarts = np.hstack((np.arange(1740,1940,25),np.arange(1940,2022,2)))
 
# 1. For soil model
# Input vars: Fractionation and partitioning
fracex = 0.4
d15N_inat = -0.5 # inputs d15N; lower = more gas loss by 1 permil = approx 1%
E_NH3 = -17.9 # amm vol losses; lower = less gas loss by 5 permil = approx 1%
E_L	= [-1,0,5] # from Bai et al. (value, low, high); higher = less gas loss; model breaks down with much deviation
E_nit = [-56.6,7.3] # frac for "exiting" nitrate pool for nitrification; lower = loss gas loss
E_denit	= [-31.3,6.1] # frac for "exiting" nitrate pool for denitification (NO3- to NO2-); lower = loss gas loss, 10 permil = approx 1%
E_denit_N2O 	= [-14.9,6.7] # frac factor for N2O production during denitrification (NO2- to N2O); minor impact on d15N emitted
E_red = [-6.6,2.7] # frac factor for N2O reduction; minor impact on d15N emitted
E_SP_nit = [29.9,2.9]
E_SP_denit = [-1.6,3.0]
E_SP_red = [-5,-8,-2] 
f_soilpoolloss = 0.5 # fraction of soil pool lost to atm in each timestep; no impact
scale_fitNO = np.array((1.,1.,1.,1.)) # scaling factors for the four parameters a,b,c,d of the sigmoid fit for NO/(N2O+NO) where: a + d = RHS plateau; b = mid point of rise; c = slope (larger = steeper); d = LHS plateau
scale_fitN2 = np.array((1.,1.,1.,1.)) # scaling factors for N2/(N2O+N2) sigmoid fit
scale_fitnit = np.array((1.,1.,1.,1.)) # scaling factors for nit/(nit+denit) sigmoid fit
#gaspart = para.gp_gaspart(20,scale_fitNO = scale_fitNO,scale_fitN2 = scale_fitN2,plotYN="Y")
N2Opart = para.gp_nitdenit(20,scale_fitnit = scale_fitnit,plotYN="Y")

# 2. For emissions
fertEFred = 0.4 #1 # reduction in EF for N2O emitted from fertiliser relative to natural conditions 
scaleDep = 1 # scale for deposition N inputs
scaleFix = 1 # scale for fixation N inputs
temp_sens = 1.1 # check this in excel against 1800 flux
Ninputs = nc4.Dataset('N_input_data_v2.nc','r')
datarange = Ninputs.variables["time"][:].data
# wwt source
d15_WWT = [-11.6,12.7] # harris et al, 2017, SI synthesis
SP_WWT = [10.5,5.7] # harris et al, 2017, SI synthesis
# tro source
d15_TRO = [-7.2,1.2] # harris et al, 2017, SI synthesis
SP_TRO = [10.0,4.3] # harris et al, 2017, SI synthesis

# 3. For atm model
# atmospheric vars
MW_air = 28.9647 # g/mol
MW_N2O = 44 # g/mol
T_S = np.array([5.37e17,1.26e17]) # Troposphere-stratosphere exchange in kg yr-1 (val, err; from Schilt 2014)
T_Sm = T_S*1000/MW_air # mol/year
m_trop = 0.85*1.77e20; # mol of the troposphere - check source: Schilt?
m_strat = 0.15*1.77e20; # estimate moles of the stratosphere 
F_ocean = [5.1,1.8] # Ocean flux in Tg N y-1 (Tian 2020)
ltPD = [116,9] # N2O lifetime in years for PD
ltPD_ltPI = [1.06,0.02]
ltPI = [ltPD[0]*ltPD_ltPI[0],ltPD[1]*(ltPD_ltPI[1]+1)] # N2O lifetime in years for PI (from Prather et al; 118 in Sowers et al.)
# preanth trop
c_prea = [270,7.5] # preanthropogenic N2O concentration (ppb) - 260-270
d15_prea = [8.9,2.0] # del values for the preanth troposphere, from Toyoda et al. 2013 - average from Bernard, Rockmann, Sowers
d18_prea = [46.1,2.0] 
SP_prea = [19.05,2.0]
# ocean source
d15_ocean = [5.1,1.9] # del values for the ocean source, from Schilt et al.
d18_ocean = [44.8,3.6] # del values for the ocean source, from Schilt et al.
SP_ocean = [15.8,7.1] # SP values from Snider dataset ("marine")
