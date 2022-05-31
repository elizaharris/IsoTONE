#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Import and plot detailed MCMC results

Created on Tue Apr  7 10:19:28 2020

@author: elizaharris
"""

# Before using this file run the preamble, model and MCMC files!

#%% Import results

# MC results are in the netcdf:
# accept_MC_res = 0 for not accept, 1 for accept
# obs_solutions: observations posterior: D1 = N2O, d15N, SP end on end, D2 = n_its
# prior_solutions: D1 = the params, D2 = n_its
# atmos_opt: D1 = n_its, D2 = T_Sm_new, F_ocean_new (optimised within atm model)
# soil_results: D1 = n_its, D2 = datarng (non-na indices of d15N in global grid), D3 = k_L, k_NH3, k_G, soil_NO3_steady, d15N, SP )))
# Nproc_results: D1 = n_its, D2 = datarng (non-na indices of d15N in global grid), D3 = N2O_all, N2_all, NO_all, denit_N2O, nit_N2O, red_N2O

# find all files
import glob
filenames = glob.glob('MCMC_results/*_MCMC_*')
dates = glob.glob('MCMC_results/AtmosSolutions_MCMC_*')
dates_as_num = np.zeros(len(dates))
for n in range(0,len(dates)) :
    tmp = dates[n].split("_")
    dates[n] = (tmp[3]+"_"+tmp[4].split(".")[0])
    dates_as_num[n] = int(tmp[3]+tmp[4].split(".")[0])
# make sure ordered right and add step sizes to match 
tmp = dates_as_num.argsort()
dates_o = dates.copy()
for n in range(0,len(dates)) : dates_o[n] = dates[tmp[n]]
# first get the full prior solutions
def fix_acceptMC(accept_MC_res): # sometimes dimensions of this file are wrong...
    if len(accept_MC_res.shape) == 1: 
        new = np.zeros((int(accept_MC_res.shape[0]/2),2))
        new[:,0] = accept_MC_res[0:int(accept_MC_res.shape[0]/2)]
        new[:,1] = accept_MC_res[int(accept_MC_res.shape[0]/2):]
    else : new = accept_MC_res
    return(new)
for n in range(0,len(dates_o)) :
    name = ("MCMC_"+dates_o[n])
    if n==0: # read files in
        accept_MC_res = np.loadtxt(fname = "MCMC_results/AcceptMC_"+name+".txt")
        accept_MC_res = fix_acceptMC(accept_MC_res)
        stepsize = accept_MC_res[:,1].copy()
        accept_MC_res = accept_MC_res[:,0]
        prior_solutions = np.loadtxt(fname = "MCMC_results/PriorSolutions_"+name+".txt")
        atmos_opt = np.loadtxt(fname = "MCMC_results/AtmosSolutions_"+name+".txt")
    else :
        tmp = np.loadtxt(fname = "MCMC_results/AcceptMC_"+name+".txt")
        tmp = fix_acceptMC(tmp)
        accept_MC_res = np.append(accept_MC_res,tmp[:,0])
        stepsize = np.append(stepsize,tmp[:,1])
        tmp = np.loadtxt(fname = "MCMC_results/PriorSolutions_"+name+".txt"); prior_solutions = np.hstack((prior_solutions,tmp))
        tmp = np.loadtxt(fname = "MCMC_results/AtmosSolutions_"+name+".txt"); atmos_opt = np.concatenate((atmos_opt,tmp),0)

### then get and combine the obs data
name = ("MCMC_"+dates_o[0]); ncres = nc4.Dataset("MCMC_results/ModelResults_"+name+".nc",'r')
datarng = np.where(~np.isnan(d15N_grid) & ~np.isinf(d15N_grid))
n_acc = np.zeros(len(dates))
n_its = np.zeros(len(dates))
obs_res_msd = np.zeros((len(dates),ncres.dimensions["n_obs"].size,2))
soilmodres_m = np.zeros((len(dates),ncres.dimensions["datarng"].size,ncres.dimensions["soilparams"].size))
Nprocesses_m = np.zeros((len(dates),ncres.dimensions["datarng"].size,ncres.dimensions["Nprocparams"].size))
soilmodres_s = np.zeros((len(dates),ncres.dimensions["datarng"].size,ncres.dimensions["soilparams"].size))
Nprocesses_s = np.zeros((len(dates),ncres.dimensions["datarng"].size,ncres.dimensions["Nprocparams"].size))
for n in range(0,len(dates)) :
    name = ("MCMC_"+dates_o[n])
    ncres = nc4.Dataset("MCMC_results/ModelResults_"+name+".nc",'r')
    n_acc[n] = ncres.variables["n_accepted"][:]
    n_its[n] = ncres.variables["n_its"][:]
    obs_res_msd[n,:,:] = ncres.variables["obs_res"][:,:]
    soilmodres_m[n,:,:] = ncres.variables["soil_res_m"][:,:]
    Nprocesses_m[n,:,:] = ncres.variables["Nproc_res_m"][:,:]
    soilmodres_s[n,:,:] = ncres.variables["soil_res_s"][:,:]
    Nprocesses_s[n,:,:] = ncres.variables["Nproc_res_s"][:,:]
# find combined mean and sd
def combined_mean(data,std,n) :
    data_dim = len(data.shape)
    Sx = data*0
    for i in range(0,len(n)) :
        if data_dim == 3: Sx[i,:,:] = data[i,:,:]*n[i]
        if data_dim == 2: Sx[i,:] = data[i,:]*n[i]
    tx = np.nansum(Sx,axis=0)
    m = tx/np.nansum(n) # combined mean
    Sxsd = data*0   
    for i in range(0,len(n)) :
        if data_dim == 3: Sxsd[i,:,:] = std[i,:,:]*std[i,:,:]*(n[i]-1) + (Sx[i,:,:]*Sx[i,:,:]/n[i])
        if data_dim == 2: Sxsd[i,:] = std[i,:]*std[i,:]*(n[i]-1) + Sx[i,:]*Sx[i,:]/n[i]
    txx = np.nansum(Sxsd,axis=0)
    tn = np.nansum(n)
    s = abs((txx-(tx*tx)/tn)/(tn-1))**0.5 ### combined SD
    nans = np.isnan(data).all(axis=0)
    m[nans] = np.nan
    s[nans] = np.nan
    return(m,s) 
post_obs, post_obs_sd = combined_mean(data = obs_res_msd[:,:,0], std = obs_res_msd[:,:,1], n = n_acc)
post_kG_m, post_kG_sd = combined_mean(data = soilmodres_m[:,:,2], std = soilmodres_s[:,:,2], n = n_acc)
post_fN2O_m, post_fN2O_sd = combined_mean(data = Nprocesses_m[:,:,0], std = Nprocesses_s[:,:,0], n = n_acc)

#%% MCMC diagnostics

# all values
full_params =  np.hstack((prior_solutions.transpose(),atmos_opt))
full_names = ["c_prea","N2 fit scale","frac express","fert EF_red","temp sens","lifetime PD/PI","lifetime PD","d15N ocean","SP ocean","d15N prea","NO fit scale","SP prea","TtoS mass", "F ocean"]
full_params[:,-2] = full_params[:,-2]/1000*MW_air
# post values
go = np.where(accept_MC_res==1)[0]
post = np.nanmean(full_params[go,:],axis=0)
post_sd = np.nanstd(full_params[go,:],axis=0)
# compile all prior values
full_prior = np.zeros((full_params.shape[1],4)) 
full_prior[0:params.shape[0],:] = params_orig.copy()  
full_prior[-2,:] = [T_S[0],T_S[1],T_S[1],1]
full_prior[-1,:] = [F_ocean[0],F_ocean[1],F_ocean[1],1]
# select to plot only every 10th point of the full data (figure too large otherwise)
subset = np.arange(0,full_params.shape[0],4)

### check the coverage of the parameter space 
fig, ax = plt.subplots(2,3,figsize=(12,10))
i1 = 0; i2 = 0;
col_pairs = [[0,2],[3,4],[5,6],[7,9],[8,11],[1,10]] 
for n in range(0,6) :
    cols = col_pairs[n]
    #ax[i1,i2].scatter(full_params[:,cols[0]],full_params[:,cols[1]],c=range(0,full_params.shape[0]),marker=".") # plot all points coloured by iteration
    cp = ax[i1,i2].scatter(full_params[subset,cols[0]],full_params[subset,cols[1]],c=stepsize[subset],marker=".",s=4) # plot all points coloured by stepsize
    ax[i1,i2].scatter(full_params[go,cols[0]],full_params[go,cols[1]],c="silver",marker=".",s=1) # outline accepted points
    ax[i1,i2].plot([post[cols[0]]]*3,[ post[cols[1]]-post_sd[cols[1]],post[cols[1]],post[cols[1]]+post_sd[cols[1]] ],"r")
    if full_prior[cols[1],3] == 1: # gaussian error
        ax[i1,i2].plot([full_prior[cols[0],0]]*3,[ full_prior[cols[1],0]-full_prior[cols[1],1],full_prior[cols[1],0],full_prior[cols[1],0]+full_prior[cols[1],1] ],"b")
    else: ax[i1,i2].plot([full_prior[cols[0],0]]*3,[ full_prior[cols[1],1],full_prior[cols[1],0],full_prior[cols[1],2] ],"b--") # uniform
    ax[i1,i2].plot([ post[cols[0]]-post_sd[cols[0]],post[cols[0]],post[cols[0]]+post_sd[cols[0]] ],[post[cols[1]]]*3,"r")
    if full_prior[cols[0],3] == 1: # gaussian error
        ax[i1,i2].plot([ full_prior[cols[0],0]-full_prior[cols[0],1],full_prior[cols[0],0],full_prior[cols[0],0]+full_prior[cols[0],1] ],[full_prior[cols[1],0]]*3,"b")
    else: ax[i1,i2].plot([ full_prior[cols[0],1],full_prior[cols[0],0],full_prior[cols[0],2] ],[full_prior[cols[1],0]]*3,"b--") # uniform
    #if ((i1==0) & (i2==0)): ax[i1,i2].legend((["post mean","prior mean"]))
    #if ((i1==0) & (i2==0)): plt.colorbar(cp)
    #ax[i1,i2].set_xlabel(full_names[cols[0]])
    #ax[i1,i2].set_ylabel(full_names[cols[1]])
    i2 = i2+1
    if i2 == 3: i2 = 0; i1 = i1+1; 
### check the means and stds change with diff selections of the data (shouldn't change once finished!)
    
#%% Compare results from different step sizes and look at significance
    
import scipy.stats as stats
nsteps = len(np.unique(stepsize))
step_means = np.zeros((nsteps+1,full_params.shape[1])) # means for 0.3, 0.5, 0.75, all
step_stds = np.zeros((nsteps+1,full_params.shape[1]))
step_ps = np.zeros((nsteps+1,full_params.shape[1]))*np.nan
for n in range(0,nsteps+1):
    if n<nsteps: 
        total = np.where((stepsize == np.unique(stepsize)[n]))
        tmp = np.where((stepsize == np.unique(stepsize)[n]) & (accept_MC_res==1))
        tmp2 = np.where((stepsize != np.unique(stepsize)[n]) & (accept_MC_res==1))
        t,p = stats.ttest_ind(full_params[tmp,:],full_params[tmp2,:],axis=1) # are results from this step different to all other accepted results?
        step_ps[n,:] = p # p of whether this group of stepsize results are sig diff to the rest of the accepted results
        print("stepsize = "+str(np.unique(stepsize)[n])+": runs = "+str(len(total[0]))+", accepted = "+str(len(tmp[0]))+", rate = "+str(len(tmp[0])/len(total[0])*100))
    if n==nsteps+1: 
        tmp = np.where((accept_MC_res==1))
        print("all steps: runs = "+str(len(accept_MC_res))+", accepted = "+str(len(tmp[0]))+", rate = "+str(len(tmp[0])/len(accept_MC_res)*100))
    step_means[n,:] = np.nanmean(full_params[tmp,:],axis=1)
    step_stds[n,:] = np.nanstd(full_params[tmp,:],axis=1)
    
# test the mean and stdev for all values against the prior
for n in range(0,full_params.shape[1]):
    if full_prior[n,3] == 1: # gaussian priors
        t,p = stats.ttest_ind_from_stats(step_means[3,n],step_stds[3,n],len(go),full_prior[n,0],full_prior[n,1],len(go))
        step_ps[nsteps,n] = p # p of whether this group of stepsize results are sig diff to the rest of the accepted results
        print(p)
    #if full_prior[n,3] == 0: # gaussian priors
step_ps_summ = step_ps.copy()*0
step_ps_summ[np.where(step_ps<0.01)]= 1
step_ps_summ = pd.DataFrame(step_ps_summ,columns=full_names,index=np.append(np.unique(stepsize),"all_to_prior")).transpose()
print(step_ps_summ)
print("1 = significantly different")
### note: all gaussian parameters are significantly different to priors

labels = np.append(np.unique(stepsize),"all")
w = 0.7
fig, ax = plt.subplots(5,3)
x,y = 0,0
for n in range(0,full_params.shape[1]):
    ax[x,y].bar(range(0,4),step_means[:,n],yerr=step_stds[:,n],width=w,linewidth=0,color="r",tick_label=labels)
    low = min(step_means[:,n]-step_stds[:,n])
    high = max(step_means[:,n]+step_stds[:,n])
    ax[x,y].set_ylim([(low-1*(high-low)), (high+1*(high-low))])
    ax[x,y].set_title(full_names[n],fontsize=8)
    ax[x,y].set_xlim([-0.5,3.5])
    ax[x,y].plot((-0.5,3.5),(0,0),"k") # plot 0 line
    ax[x,y].plot((-0.5,3.5),(full_prior[n,0],full_prior[n,0]),"b")
    if full_prior[n,3] == 0:  # uniform error in prior
        ax[x,y].plot((-0.5,3.5),(full_prior[n,1],full_prior[n,1]),"b--")
        ax[x,y].plot((-0.5,3.5),(full_prior[n,2],full_prior[n,2]),"b--")
    if full_prior[n,3] == 1:  # gaussian error in prior
        ax[x,y].plot((-0.5,3.5),(full_prior[n,0]-full_prior[n,1],full_prior[n,0]-full_prior[n,1]),"b:")
        ax[x,y].plot((-0.5,3.5),(full_prior[n,0]+full_prior[n,1],full_prior[n,0]+full_prior[n,1]),"b:")
    x = x+1
    if x==5: x=0; y = y+1
    
#%% Look at correlations between the different parameters
    
# select only accepted params and reorder params for this figure
accepted = np.zeros((len(go),full_params.shape[1]))
a_names = ["c_prea","lifetime PD/PI","lifetime PD","d15N ocean","SP ocean","d15N prea","SP prea","frac express","fert EF_red","temp sens","N2 fit scale","NO fit scale","TtoS mass","F ocean"]
for n in range(0,len(a_names)):
    for i in range(0,len(a_names)):
        if full_names[i] == a_names[n]: accepted[:,n] = full_params[go,i]
# look at correlations
import scipy.stats as stats
correlations_pearson = np.zeros((len(a_names),len(a_names),2)) # corr, p
correlations_spearman = np.zeros((len(a_names),len(a_names),2)) # corr, p
for n in range(0,len(a_names)):
    for i in range(0,len(a_names)):
        r,p = stats.pearsonr(accepted[:,n],accepted[:,i])
        if n!=i: correlations_pearson[n,i,0:2] = [r,p]
        r,p = stats.spearmanr(accepted[:,n],accepted[:,i])
        if n!=i: correlations_spearman[n,i,0:2] = [r,p]
corrs_p = correlations_pearson[:,:,0].copy()
corrs_p[correlations_pearson[:,:,1]>0.01]=0
corrs_p_df = pd.DataFrame(corrs_p, index=a_names,columns=a_names)
corrs_s = correlations_spearman[:,:,0].copy()
corrs_s[correlations_spearman[:,:,1]>0.01]=0
corrs_s_df = pd.DataFrame(corrs_s, index=a_names,columns=a_names)
# plot the results
import seaborn as sns
fig, ax = plt.subplots(1,1)
sns.heatmap(corrs_p_df, vmin=-1, vmax=1, center=0,cmap=sns.diverging_palette(220, 20, n=200),square=True)
plt.title("Pearson", fontsize =20); fig.show()
fig, ax = plt.subplots(1,1)
sns.heatmap(corrs_s_df, vmin=-1, vmax=1, center=0,cmap=sns.diverging_palette(220, 20, n=200),square=True)
plt.title("Spearman", fontsize =20); fig.show()

#%% PCA to look at multiparam correlations

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# standardise the data scaling
full_params_norm = StandardScaler().fit_transform(full_params[go,:])
# perform PCA
pca = PCA(n_components=len(full_names))
principalComponents = pca.fit_transform(full_params_norm)
colnames = ["PC"]*len(full_names)
for n in range(0,len(full_names)): colnames[n] = colnames[n]+str(n+1)
principalDf = pd.DataFrame(data = principalComponents, columns = colnames)
exvar = pca.explained_variance_ratio_ # tells how much each PC explains (21, 17, 11, 10%...): cumsum(exvar) for cumulative sum: 8 PCs for 90% of variance
components = pca.components_
# visualise
fig, ax = plt.subplots(2,1)
for n in (0,1):
    ax[n].set_xlabel('Principal Component 1: '+str(exvar[0].round(3)*100)+"%", fontsize = 8)
    ax[n].set_ylabel('Principal Component '+str(n+2)+": "+str(exvar[n+1].round(3)*100)+"%", fontsize = 8)
    #ax.scatter(principalDf.loc[:, 'PC1'], principalDf.loc[:, 'PC2'], s = 50) plot all data if needed
    for i, txt in enumerate(full_names):
        ax[n].annotate(txt, (components[0,i]+0.01, components[n+1,i]+0.01))
        ax[n].plot((0,components[0,i]), (0,components[n+1,i]),"k")
    ax[n].scatter(components[0,:],components[n+1,:])

#%% Figures
        
print(str(sum(n_acc)/sum(n_its)*100)+"% of solutions accepted; ",str(sum(n_acc))+" solutions")
go = np.where(accept_MC_res==1)[0]
post = np.nanmean(prior_solutions[:,go],axis=1)
post_sd = np.nanstd(prior_solutions[:,go],axis=1)
# post model
tmp = model(post)
modres_post = expand_modres(tmp)
k_G = np.zeros(d15N_grid.shape)+np.nan; k_G[datarng] = tmp[3][:,2]
f_N2O = np.zeros(d15N_grid.shape)+np.nan; f_N2O[datarng] = tmp[4][:,0]
EF = k_G*f_N2O*100
EF_zones_post = utils.climzone_means(var1_grid = MAT_grid, var2_grid = BulkD_grid, datavar1 = MAT_grid,
                              datavar2 = BulkD_grid,data = EF,bins=4,plotfigs="N",LON=LON,LAT=LAT)
# prior model
tmp = model(x=params_orig[:,0])
modres_prior = expand_modres(tmp)

# N gas parameterisations
para.gp_finalfig(scale_fitNO = (1,post[10],1,1),scale_fitN2 = (1,post[1],1,1))

# Plot conc and isotopes with time
labels = ("N2O trop (ppb)","d15N (permil)","SP (permil)","EF")
fig, ax = plt.subplots(4,2)
for n in range(0,4):
    tmp = obs[:,2]==n+1
    if n<3:
        ax[n,0].plot(tstarts,obs[tmp,0],"bo")
        ax[n,0].plot(tstarts,post_obs[tmp],"cx")
        ax[n,0].plot(tstarts,modres_prior[tmp],"ro")
        ax[n,0].plot(tstarts,modres_post[tmp],"mx")
    if n==3:
        ax[n,0].errorbar(np.arange(1,17),obs[tmp,0],yerr=obs[tmp,1],color="blue")
        ax[n,0].plot(np.arange(1,17),obs[tmp,0],"bo")
        ax[n,0].errorbar(np.arange(1,17),post_obs[tmp],yerr=post_obs_sd[tmp],color="cyan")
        ax[n,0].plot(np.arange(1,17),post_obs[tmp],"cx")
        ax[n,0].plot(np.arange(1,17),modres_prior[tmp],"ro")
        ax[n,0].plot(np.arange(1,17),modres_post[tmp],"mx")
    if n==0: ax[n,0].legend((["obs_prior","obs_post","mod_prior","mod_post"]))
    ax[n,0].set_ylabel(labels[n])
    # plot the 1:1
    ymin = np.nanmin(np.append(obs[tmp,0],modres_prior[tmp]))
    ymax = np.nanmax(np.append(obs[tmp,0],modres_prior[tmp]))
    ax[n,1].plot(obs[tmp,0],modres_prior[tmp],"go")
    ax[n,1].plot(post_obs[tmp],modres_post[tmp],"yx")
    ax[n,1].plot((ymin-(ymax-ymin)/10,ymax+(ymax-ymin)/10),(ymin-(ymax-ymin)/10,ymax+(ymax-ymin)/10),"k-")
    ax[n,1].set_xlim(ymin-(ymax-ymin)/10,ymax+(ymax-ymin)/10)
    ax[n,1].set_ylim(ymin-(ymax-ymin)/10,ymax+(ymax-ymin)/10)
    ax[n,1].set_xlabel("Obs: "+labels[n])
    ax[n,1].set_ylabel("Mod: "+labels[n])
    if n==0: ax[n,1].legend((["prior","post"]))
    # agreement stats
    print("RMSE prior, "+labels[n]+" = "+str((np.nanmean((obs[tmp,0] - modres_prior[tmp])**2))**0.5))
    print("RMSE post, "+labels[n]+" = "+str((np.nanmean((post_obs[tmp] - modres_post[tmp])**2))**0.5))
fig.show()

#%%
# plot the clim zones data also
fig, ax = plt.subplots(4,2)
for i in range(0,4):
    tmp = N2O_fluxes_zones[:,2] == i
    eb = ax[i,0].errorbar(N2O_fluxes_zones[tmp,1],N2O_fluxes_zones[tmp,4],yerr=N2O_fluxes_zones[tmp,5],color=("b"),fmt=":") # data (prior)
    eb[-1][0].set_linestyle(":")
    ax[i,0].plot(EF_zones_post[tmp,1],EF_zones_post[tmp,4],"mx") # model
    ax[i,0].plot(N2O_fluxes_zones[tmp,1],N2O_fluxes_zones[tmp,4],"d",color=("b")) # data (prior)
    tmp1 = np.where(obs[:,2]==4)[0][tmp]
    ax[i,0].plot(EF_zones_post[tmp,1],post_obs[tmp1],"cx") # data (post)
    ax[i,0].set_ylabel("N2O EF")
    ax[i,0].set_ylim((0,5))
    # compare errors also
    ax[i,1].plot(N2O_fluxes_zones[tmp,1],N2O_fluxes_zones[tmp,5],"bd")
    ax[i,1].plot(N2O_fluxes_zones[tmp,1],post_obs_sd[tmp1],"cx")
    ax[i,1].set_ylabel("Stdev")
    ax[i,1].set_ylim((0,2.75))
ax[3,0].set_xlabel("MAT")
ax[3,1].set_xlabel("MAT")

# Params in order for table
table_order = [0,2,3,4,5,6,7,9,8,11,1,10]
for n in table_order:
    print(post[n],"+-",post_sd[n])


