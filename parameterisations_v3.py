#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 14:36:56 2020

@author: elizaharris
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#%% 1. Partitioning of N gas losses into NO, N2O and N2

# get the Bai model
gaspart = pd.read_csv("para_Bai_WFPS.csv",delimiter=",") 

# import extended gas partitioning data
data = pd.read_csv("parameterisations_N_gases.csv",delimiter=",") 
data_vals = np.array(data)
# 5 6 7 =  WFPS(%)	 N2O/(N2O+NO)	 N2O/(N2O+N2)

# fit to the N2O/(N2O+NO) data
from scipy.optimize import curve_fit
xdata = data_vals[:,5].astype("float")
ydata = data_vals[:,6].astype("float")
xdata = xdata[np.where(~np.isnan(ydata))]
ydata = ydata[np.where(~np.isnan(ydata))]
xdata = np.append(xdata,np.array((0.,90.,100.)))
ydata = np.append(ydata,np.array((0.,1.,1.)))
def sigmoid(x, L ,x0, k, b):
    y = L / (1 + np.exp(-k*(x-x0)))+b
    return (y)
p0 = [np.nanmax(ydata),np.median(xdata),0.1,np.nanmin(ydata)] # mandatory initial guess
popt1, pcov1 = curve_fit(sigmoid, xdata, ydata,p0, method='dogbox',bounds=(0,100)) # fit the curve
# L + b = RHS plateau; x0 = mid point of rise; k = slope (larger = steeper); b = LHS plateau

# fit to the N2O/(N2O+N2) data
xdata = data_vals[:,5].astype("float")
ydata = data_vals[:,7].astype("float")
xdata = xdata[np.where(~np.isnan(ydata))]
ydata = ydata[np.where(~np.isnan(ydata))]
xdata = np.append(xdata,np.array((0.)))
ydata = np.append(ydata,np.array((1.)))
def sigmoid2(x, L ,x0, k, b): # opposite sign on k this time!
    y = L / (1 + np.exp(k*(x-x0)))+b
    return (y)
p0 = [np.nanmax(ydata),np.median(xdata),0.1,np.nanmin(ydata)] # mandatory initial guess
popt2, pcov2 = curve_fit(sigmoid2, xdata, ydata,p0, method='dogbox',bounds=(0,100)) # fit the curve

def calc_gases(w,popt1,popt2) : 
    yfit1 = sigmoid(w,popt1[0],popt1[1],popt1[2],popt1[3]) # fit to the N2O/(N2O+NO) data
    yfit2 = sigmoid2(w,popt2[0],popt2[1],popt2[2],popt2[3]) # fit to the N2O/(N2O+N2) data
    N2O_all = 1/( (1-yfit1)/yfit1 + (1-yfit2)/yfit2 + 1 ) 
    NO_all = ((1-yfit1)/yfit1) /( (1-yfit1)/yfit1 + (1-yfit2)/yfit2 + 1 ) 
    N2_all = ((1-yfit2)/yfit2) /( (1-yfit1)/yfit1 + (1-yfit2)/yfit2 + 1 ) 
    return np.vstack((N2O_all,NO_all,N2_all)).transpose()

# final parameterisation = N2O/(N2+NO+N2O): derivation:
# a = N2O/(NO+N2O) and b = N2O/(N2+N2O)
# NO = (1-a)/a*N2O and N2 = (1-b)/b*N2O
# N2O/(N2+NO+N2O) = N2O/( (1-a)/a*N2O + (1-b)/b*N2O + N2O ) ... flip both upside down
# (N2+NO+N2O)/N2O = (1-b)/b + (1-a)/a + 1
# therefore N2O/(N2+NO+N2O) = 1/( (1-b)/b + (1-a)/a + 1 )
WFPSrng = np.arange(0,100)
yfit1 = sigmoid(WFPSrng,popt1[0],popt1[1],popt1[2],popt1[3])
yfit2 = sigmoid2(WFPSrng,popt2[0],popt2[1],popt2[2],popt2[3])
res = calc_gases(WFPSrng,popt1,popt2)
fig,ax = plt.subplots(3,1)
ax[0].plot(data_vals[:,5],data_vals[:,6],"bo")
ax[0].plot(np.arange(0,100),yfit1,"b-")
ax[0].set_ylabel("N2O/(N2O+NO)")
ax[1].plot(data_vals[:,5],data_vals[:,7],"bo")
ax[1].plot(np.arange(0,100),yfit2,"b-")
ax[1].set_ylabel("N2O/(N2O+N2)")
ax[2].plot(np.arange(0,100),res[:,0],"r-")
ax[2].plot(np.arange(0,100),res[:,1],"g-")
ax[2].plot(np.arange(0,100),res[:,2],"c-")
ax[2].plot(gaspart["WFPS"]*100,gaspart["%N2O"],"r:")
ax[2].plot(gaspart["WFPS"]*100,gaspart["%NO"],"g:")
ax[2].plot(gaspart["WFPS"]*100,gaspart["%N2"],"c:")
plt.legend(("N2O/(N2O+N2+NO)","NO/(N2O+N2+NO)","N2/(N2O+N2+NO)"))
fig.show()

# create function
def gp_gaspart(WFPS,scale_fitNO = (1,1,1,1),scale_fitN2 = (1,1,1,1),plotYN="N"): # function to partition N2O, NO, N2 over the sum of the three
    popt1_scale = popt1.copy()*scale_fitNO
    popt2_scale = popt2.copy()*scale_fitN2
    res = calc_gases(WFPS,popt1_scale,popt2_scale)
    tmp = {}
    tmp["N2O_all"] = res[:,0]
    tmp["NO_all"] = res[:,1]
    tmp["N2_all"] = res[:,2]
    if plotYN=="Y": # plot to check impact of scaling factors if needed
        WFPSrng = np.arange(0,100)
        res = calc_gases(WFPSrng,popt1_scale,popt2_scale)
        res_orig = calc_gases(WFPSrng,popt1,popt2)
        fig,ax = plt.subplots(1,1)
        ax.plot(WFPSrng,res[:,0],"r-")
        ax.plot(WFPSrng,res[:,1],"g-")
        ax.plot(WFPSrng,res[:,2],"c-")
        ax.plot(WFPSrng,res_orig[:,0],"r:")
        ax.plot(WFPSrng,res_orig[:,1],"g:")
        ax.plot(WFPSrng,res_orig[:,2],"c:")
        plt.legend(("N2O/(N2O+N2+NO)","NO/(N2O+N2+NO)","N2/(N2O+N2+NO)"))
    return tmp

# check scale changing
gp_gaspart(np.arange(0,100),scale_fitNO = (0.5,1,1,1),scale_fitN2 = (1,0.5,1,1),plotYN="Y")

#%% 2. Partitioning of denitrification/nitrification contribution to N turnover based on WFPS
  
# import extended gas partitioning data
dataND = pd.read_csv("parameterisations_N2O_nit_denit.csv",delimiter=",") 
dataND_vals = np.array(dataND)
# 4 5 6 =  WFPS(%)	 f_nit	 f_denit

# fit to the data
xdata = dataND_vals[:,5].astype("float")
ydata = dataND_vals[:,6].astype("float")
xdata = xdata[np.where(~np.isnan(ydata))]
ydata = ydata[np.where(~np.isnan(ydata))]
p0 = [np.nanmax(ydata),np.median(xdata),0.1,np.nanmin(ydata)] # mandatory initial guess
popt, pcov = curve_fit(sigmoid, xdata, ydata,p0, method='dogbox',bounds=(0,100)) # fit the curve
nit_N2O = sigmoid(np.arange(0,100),popt[0],popt[1],popt[2],popt[3])
denit_N2O = 1-nit_N2O

# plot to check
fig,ax = plt.subplots(3,1)
ax[0].plot(dataND_vals[:,5],dataND_vals[:,7],"ro")
ax[0].plot(np.arange(0,100),denit_N2O,"r-")
ax[0].set_ylabel("f_denit")
ax[0].set_ylim(0,1)
ax[1].plot(np.arange(0,100),denit_N2O,"r-")
ax[1].plot(np.arange(0,100),nit_N2O,"r--")
ax[1].set_ylim(0,1)
ax[2].plot(np.arange(0,100),res[:,0]*denit_N2O,"r:")
ax[2].plot(np.arange(0,100),res[:,0]*nit_N2O,"r--")
ax[2].plot(np.arange(0,100),res[:,0],"r")
ax[2].plot(np.arange(0,100),res[:,1],"g-")
ax[2].plot(np.arange(0,100),res[:,2],"c-")
ax[2].set_ylim(0,1)
plt.legend(("N2Od/(N2O+N2+NO)","N2On/(N2O+N2+NO)","N2O/(N2O+N2+NO)","NO/(N2O+N2+NO)","N2/(N2O+N2+NO)"))
plt.show()

# create function
def gp_nitdenit(WFPS,scale_fitnit = (1,1,1,1),plotYN="N"): # function to partition N2O, NO, N2 over the sum of the three
    popt_scale = popt*scale_fitnit
    tmp = {}
    tmp["nit_N2O"] = sigmoid(WFPS,popt_scale[0],popt_scale[1],popt_scale[2],popt_scale[3])
    tmp["denit_N2O"] = 1-tmp["nit_N2O"]
    if plotYN=="Y": # plot to check impact of scaling factors if needed
        WFPSrng = np.arange(0,100)
        res = sigmoid(WFPSrng,popt_scale[0],popt_scale[1],popt_scale[2],popt_scale[3])
        res_orig = sigmoid(WFPSrng,popt[0],popt[1],popt[2],popt[3])
        fig,ax = plt.subplots(1,1)
        ax.plot(WFPSrng,res,"r-")
        ax.plot(WFPSrng,1-res,"g-")
        ax.plot(WFPSrng,res_orig,"r:")
        ax.plot(WFPSrng,1-res_orig,"g:")
        plt.legend(("N2On/(N2O+N2+NO)","N2Od/(N2O+N2+NO)"))
    return tmp

#%% plot final figure
    
def gp_finalfig(scale_fitNO = (1,1,1,1),scale_fitN2 = (1,1,1,1)): # function to partition N2O, NO, N2 over the sum of the three
    popt1_scale = popt1.copy()*scale_fitNO
    popt2_scale = popt2.copy()*scale_fitN2
    WFPSrng = np.arange(0,100)
    res = calc_gases(WFPSrng,popt1_scale,popt2_scale)
    res_orig = calc_gases(WFPSrng,popt1,popt2)
    yfit1_post = sigmoid(WFPSrng,popt1_scale[0],popt1_scale[1],popt1_scale[2],popt1_scale[3]) # fit to the N2O/(N2O+NO) data
    if min(yfit1_post)<0: print("N2O/(N2O+NO) goes too low! min=",str(min(yfit1_post)))
    if max(yfit1_post)>=1.05: print("N2O/(N2O+NO) goes too high! max=",str(max(yfit1_post)))
    yfit2_post = sigmoid2(WFPSrng,popt2_scale[0],popt2_scale[1],popt2_scale[2],popt2_scale[3]) # fit to the N2O/(N2O+N2) data
    if min(yfit2_post)<0: print("N2O/(N2O+N2) goes too low! min=",str(min(yfit2_post)))
    if max(yfit2_post)>=1.05: print("N2O/(N2O+N2) goes too high! max=",str(max(yfit2_post)))
    fig,ax = plt.subplots(3,1)
    ax[0].plot(WFPSrng,res[:,0],"r-") # posterior results
    ax[0].plot(WFPSrng,res[:,1],"g-")
    ax[0].plot(WFPSrng,res[:,2],"c-")
    ax[0].plot(WFPSrng,res_orig[:,0],"r:") # prior results
    ax[0].plot(WFPSrng,res_orig[:,1],"g:")
    ax[0].plot(WFPSrng,res_orig[:,2],"c:")
    ax[0].plot(gaspart["WFPS"]*100,gaspart["%N2O"],"r--",) # Bai parameterisation
    ax[0].plot(gaspart["WFPS"]*100,gaspart["%NO"],"g--")
    ax[0].plot(gaspart["WFPS"]*100,gaspart["%N2"],"c--")
    plt.legend(("N2O/(N2O+N2+NO)","NO/(N2O+N2+NO)","N2/(N2O+N2+NO)"))
    ax[1].set_ylabel("N2O/(N2O+NO)")
    ax[1].plot(data_vals[:,5],data_vals[:,6],"bo") 
    ax[1].plot(np.arange(0,100),yfit1,"b:")
    ax[1].plot(np.arange(0,100),res[:,0]/(res[:,0]+res[:,1]),"b")
    ax[2].set_ylabel("N2O/(N2O+N2)")
    ax[2].plot(data_vals[:,5],data_vals[:,7],"bo") 
    ax[2].plot(np.arange(0,100),yfit2,"b:")
    ax[2].plot(np.arange(0,100),res[:,0]/(res[:,0]+res[:,2]),"b")
    
def gp_finalfig_error(scale_fitNO = (1,1,1,1),scale_fitN2 = (1,1,1,1),err_NO=(0,0,0,0),err_N2=(0,0,0,0)): # function to partition N2O, NO, N2 over the sum of the three
    popt1_scale = popt1.copy()*scale_fitNO
    popt1_scalemin = popt1.copy()*np.sum((scale_fitNO,-np.array(err_NO)),axis=0)
    popt1_scalemax = popt1.copy()*np.sum((scale_fitNO,err_NO),axis=0)
    popt2_scale = popt2.copy()*scale_fitN2
    popt2_scalemin = popt2.copy()*np.sum((scale_fitN2,-np.array(err_N2)),axis=0)
    popt2_scalemax = popt2.copy()*np.sum((scale_fitN2,err_N2),axis=0)
    WFPSrng = np.arange(0,100)
    res = calc_gases(WFPSrng,popt1_scale,popt2_scale)
    res_lo = calc_gases(WFPSrng,popt1_scalemin,popt2_scalemin)
    res_hi = calc_gases(WFPSrng,popt1_scalemax,popt2_scalemax)
    res_orig = calc_gases(WFPSrng,popt1,popt2)
    fig,ax = plt.subplots(3,1)
    ax[0].plot(WFPSrng,res[:,0],"r-") # posterior results
    ax[0].plot(WFPSrng,res_hi[:,0],"r:")
    ax[0].plot(WFPSrng,res_lo[:,0],"r:")
    ax[0].plot(WFPSrng,res[:,1],"g-") # posterior results
    ax[0].plot(WFPSrng,res_hi[:,1],"g:")
    ax[0].plot(WFPSrng,res_lo[:,1],"g:")
    ax[0].plot(WFPSrng,res[:,2],"c-") # posterior results
    ax[0].plot(WFPSrng,res_hi[:,2],"c:")
    ax[1].set_ylabel("N2O/(N2O+NO)")
    ax[1].plot(np.arange(0,100),res[:,0]/(res[:,0]+res[:,1]),"b")
    ax[1].plot(np.arange(0,100),res_hi[:,0]/(res_hi[:,0]+res_hi[:,1]),"b:")
    ax[1].plot(np.arange(0,100),res_lo[:,0]/(res_lo[:,0]+res_lo[:,1]),"b:")
    ax[2].set_ylabel("N2O/(N2O+N2)")
    ax[2].plot(np.arange(0,100),res[:,0]/(res[:,0]+res[:,2]),"b")   
    ax[2].plot(np.arange(0,100),res_hi[:,0]/(res_hi[:,0]+res_hi[:,2]),"b:")
    ax[2].plot(np.arange(0,100),res_lo[:,0]/(res_lo[:,0]+res_lo[:,2]),"b:")
    
    