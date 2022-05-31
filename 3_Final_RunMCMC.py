
#%% Some set up

from utils import interval_random 
from utils import prob_u
from utils import prob_g
from utils import prob_g_multi
step_lengths = [0.5,0.25,0.75]

# define the observations to be used for param optimisation: obs = [ value, unc, group number ] (obs with the same group number are not varied independently in the MCMC)   
obs_conc = np.hstack((np.array(N2O_atmos[["N2O_ppb","sd_N2O_ppb"]]),np.zeros((N2O_atmos[["N2O_ppb"]].shape[0],1))+1))
obs_d15 = np.hstack((np.array(N2O_atmos[["d15Nbulk_permil","sd_d15Nbulk_permil"]]),np.zeros((N2O_atmos[["N2O_ppb"]].shape[0],1))+2))
obs_SP = np.hstack((np.array(N2O_atmos[["SP_permil","sd_SP_permil"]]),np.zeros((N2O_atmos[["N2O_ppb"]].shape[0],1))+3))
obs_flux = np.vstack((N2O_fluxes_zones[:,4],N2O_fluxes_zones[:,5],(np.zeros((N2O_fluxes_zones[:,4].shape[0],1))+4).transpose())).transpose()
obs = np.vstack((obs_conc,obs_d15,obs_SP,obs_flux))
obs_2016 = np.cumsum(np.array((obs_conc.shape[0],obs_d15.shape[0],obs_SP.shape[0],obs_flux.shape[0])))-2 # 2016 obs for probabilities

# define model uncertainty
mod_uncert = obs[:,1].copy()
mod_uncert_base = [0.5,0.1,0.1,0.5] # uncertainty for "2020" data, gets larger with time in past
mod_uncert_scale = (abs(tstarts-2020)/280*2+1) # scale error by 1 (2020) to 3 (1840) to account for higher uncertainty in older data
for n in range(0,3):   
    mod_uncert[obs[:,2]==(n+1)] = mod_uncert_scale*mod_uncert_base[n] # model uncertainty in conc is 3% of the obs value
mod_uncert[obs[:,2]==4] = mod_uncert_base[3]

# function to calc EFs per clim zone from model run and add to model results
def expand_modres(tmp):
    k_G = np.zeros(d15N_grid.shape)+np.nan; k_G[datarng] = tmp[3][:,2]
    f_N2O = np.zeros(d15N_grid.shape)+np.nan; f_N2O[datarng] = tmp[4][:,0]
    EF = k_G*f_N2O*100
    EF_zones = utils.climzone_means(var1_grid = MAT_grid, var2_grid = MAP_grid, datavar1 = MAT_grid,
                                  datavar2 = MAP_grid,data = EF,bins=4,plotfigs="N",LON=LON,LAT=LAT)[:,4]
    res = np.hstack((tmp[0],EF_zones))
    return(res)

runs = 0

#%% Start the MCMC        

for runs in range(0,8): # NOTE: should be a multiple of 3 to go evenly through steps!
    n_iterations = 5000
    step_length = step_lengths[runs % 3] # this is a fixed steplength - proportion of standard deviation
    # can be used for one or many obs!
    
    # est: [285,1.7,0.9,1.2,1.1,132,131,5.4,7.1,10]
    # define the parameter list to be optimised: x = [ value, unc/lo, hi, dist ] where dist = 0 = uniform and dist = 1 = gaussian
    params = np.zeros((12,4))
    params[0,:] = (265, c_prea[1], c_prea[1], 1)#(c_prea[0], c_prea[1], c_prea[1], 1) # preanth N2O conc
    params[1,:] = (scale_fitN2[1],0.7,2,0) # N2 emission scaling factor
    params[2,:] = (fracex,0.3,1.001,0) # frac expression factor
    params[3,:] = (fertEFred,0,1.001,0) # fert EF red
    params[4,:] = (temp_sens,0.04,0.04,1)
    params[5,:] = (ltPD_ltPI[0],ltPD_ltPI[1],ltPD_ltPI[1],1)
    params[6,:] = (121,ltPD[1],ltPD[1],1) #(ltPD[0],ltPD[1],ltPD[1],1)
    params[7,:] = (d15_ocean[0],d15_ocean[1],d15_ocean[1],1)
    params[8,:] = (SP_ocean[0],SP_ocean[1],SP_ocean[1],1)
    params[9,:] = (d15_prea[0],d15_prea[1],d15_prea[1],1) 
    params[10,:] = (scale_fitNO[1],0.7,1.3,0) # NO emission scaling factor: restictions keep it in the range of obs
    params[11,:] = (SP_prea[0],SP_prea[1],SP_prea[1],1) 
    params_orig = params.copy()
    #params[:,0] = [280,1.7,0.4,0.3,1.1,1.06,121,5.4,7.1,10,1,SP_prea[0]]
    UseMostRecent = "Y" # select if the most recent model final priors should be used? Use N if major changes have been made
    if UseMostRecent == "Y" :
        import glob
        filenames = glob.glob('MCMC_results/ModelResults_MCMC_*.nc')
        if len(filenames)>0: 
            order = np.zeros((len(filenames))) # make sur file names ordered by date
            for n in range(0,len(filenames)):
                tmp = filenames[n].replace('.', '_').replace('2020', '_').split("_")
                order[n] = int(tmp[-3]+tmp[-2])
            r = np.where(order==np.max(order))[0][0]
            ncres = nc4.Dataset(filenames[r],'r')
            params[:,0] = ncres.variables["last_prior"][:]  
            if params[0,0] > 9e36: params[:,0] = ncres.variables["prior_res"][:][:,0]
    
    # give an initial value for prior params and obs
    prior_solutions = np.zeros((len(params[:,0]),n_iterations))+np.nan
    prior_solutions[:,0] = params[:,0]
    tmp = model(x=prior_solutions[:,0])
    datarng = np.where(~np.isnan(d15N_grid) & ~np.isinf(d15N_grid))
    N_summary = tmp[5]
    Ntot_prev = np.array((N_summary[N_summary[:,0] == 1860,1],N_summary[N_summary[:,0] == 2010,1]))
    
    modres_prior = expand_modres(tmp) 
    modres_prev = modres_prior
    obs_solutions = np.zeros((len(obs[:,0]),n_iterations))+np.nan
    obs_solutions[:,0] = obs[:,0]
    
    # create space for the other model results
    accept_MC = np.zeros((n_iterations,1)); accept_MC[0] = 1
    atmos_opt = np.zeros((n_iterations,2))
    atmos_opt[0,:] = (tmp[1],tmp[2])
    soil_results = np.zeros((n_iterations,tmp[3].shape[0],tmp[3].shape[1]))
    soil_results[0,:,:] = tmp[3]
    Nproc_results = np.zeros((n_iterations,tmp[4].shape[0],tmp[4].shape[1]))
    Nproc_results[0,:,:] = tmp[4]
    
    # run the MCMC
    i_prev = 0 
    for i in range(1,n_iterations):
        prior_i = prior_solutions[:,i_prev].copy()
        prior_i_stepscale = params[:,1].copy()
        prior_i_stepscale[params[:,3]==0] = abs(params[params[:,3]==0,2]-params[params[:,3]==0,1])*0.25
        prior_i += step_length * prior_i_stepscale * interval_random(-1., 1., nn=len(params[:,0])) # vary all parameters independently
        prior_solutions[:,i] = prior_i.copy() # save the results
        obs_i = obs_solutions[:,i_prev].copy()
        tmp = interval_random(-1., 1., nn=int(max(obs[:,2])))
        obs_i += step_length * obs[:,1] * tmp[(obs[:,2]-1).astype(int)] # vary different observation groups independently from each other
        obs_solutions[:,i] = obs_i.copy()
        #fig, ax = plt.subplots(3,1); 
        #for n in range(1,4): ax[n-1].errorbar(tstarts,obs[obs[:,2]==n,0],obs[obs[:,2]==n,1],marker="o"); ax[n-1].plot(tstarts,obs_solutions[obs[:,2]==n,i],"rx")
        ## now test if it passes the acceptance test for priors
        # first check uniform priors
        tmp = np.where(params[:,3] == 0)
        if interval_random() > prob_u(prior_i[tmp],params_orig[tmp,1],params_orig[tmp,2]) / prob_u(prior_solutions[tmp,i_prev],params_orig[tmp,1],params_orig[tmp,2]) : continue # don't accept
        # then check gaussian priors
        tmp = np.where(params[:,3] == 1)
        if interval_random() > prob_g_multi(prior_i[tmp],params_orig[tmp,0],params_orig[tmp,1]) / prob_g_multi(prior_solutions[tmp,i_prev],params_orig[tmp,0],params_orig[tmp,1]) : continue # don't accept
        # now do it for the data (all normally dist.)  (use only one from each group, as all in each group vary the same)  
        if interval_random() > prob_g_multi(obs_i[obs_2016],obs[obs_2016,0],obs[obs_2016,1]) / prob_g_multi(obs_solutions[obs_2016,i_prev],obs[obs_2016,0],obs[obs_2016,1]) : continue # don't accept
        # now do the same for the model, with randomisation selected for d15N
        tmp = model(prior_i,d15Nrandomise=0.05)
        print(i)
        N_summary = tmp[5]
        # check total flux in 1860 and 2010 approx right acc. Tian 2019: in TgN y-1: 1860 6.3pm1.1, 2010 10pm2.2... much higher in Tian 2020??
        Ntot = np.array((N_summary[N_summary[:,0] == 1860,1],N_summary[N_summary[:,0] == 2010,1]))
        #if interval_random() > prob_g(Ntot[0],6.3,1.1*1) / prob_g(Ntot_prev[0],6.3,1.1*1) : continue # don't accept
        #if interval_random() > prob_g(Ntot[1],10,2.2*1) / prob_g(Ntot_prev[1],10,2.2*1) : continue # don't accept
        Ntot_prev = Ntot
        # then check the rest of the observations
        modres = expand_modres(tmp)
        atmos_opt[i,:] = (tmp[1],tmp[2])
        soil_results[i,:,:] = tmp[3]
        Nproc_results[i,:,:] = tmp[4]
        tmp_c = abs((modres-obs_i)/mod_uncert) # current model prob
        tmp_p = abs((modres_prev-obs_solutions[:,i_prev])/mod_uncert) # prev model prob
        mod_summ_c = np.zeros(int(max(obs[:,2]))); mod_summ_p = np.zeros(int(max(obs[:,2]))) # summarise model by observation type
        for n in range(0,int(max(obs[:,2]))) :
            mod_summ_c[n] = np.nanmean(tmp_c[ obs[:,2]==(n+1) ])
            mod_summ_p[n] = np.nanmean(tmp_p[ obs[:,2]==(n+1) ])
        current_mod_prob = prob_g_multi(mod_summ_c,mod_summ_c*0,mod_summ_c*0+1)
        prev_mod_prob = prob_g_multi(mod_summ_p,mod_summ_p*0,mod_summ_p*0+1)
        if (prev_mod_prob==0) : prev_mod_prob=1e-300
        if interval_random() > current_mod_prob/prev_mod_prob : continue # don't accept
        # it's passed all tests, append it to the solutions and make it the new starting point
        i_prev = i
        modres_prev = modres
        accept_MC[i] = 1
        go = sum(accept_MC==1)
        print("found a solution! n="+str(int(go))+" - rate="+str(int(100*go/(i+1) ))+"%" )
        
    ### save MCMC results
    
    # take means of results to reduce data size  
    go = np.where(accept_MC==1)[0]
    soil_res_mean = np.nanmean(soil_results[go,:,:],axis=0)
    soil_res_sd = np.nanstd(soil_results[go,:,:],axis=0)
    Nproc_mean = np.nanmean(Nproc_results[go,:,:],axis=0)
    Nproc_sd = np.nanstd(Nproc_results[go,:,:],axis=0)
    prior_solutions_meansd = np.vstack((np.nanmean(prior_solutions[:,go],axis=1),np.nanstd(prior_solutions[:,go],axis=1))).transpose()
    atmos_opt_meansd = np.vstack((np.nanmean(atmos_opt[go,:],axis=0),np.nanstd(atmos_opt[go,:],axis=0))).transpose()
    obs_solutions_meansd = np.vstack((np.nanmean(obs_solutions[:,go],axis=1),np.nanstd(obs_solutions[:,go],axis=1))).transpose()
        
    name = ("MCMC_"+datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    ncout = nc4.Dataset("MCMC_results/ModelResults_"+name+".nc",'w','NETCDF4'); # using netCDF3 for output format 
    ncout.createDimension('points',1);
    ncout.createDimension('rows',2);
    ncout.createDimension('soilparams',soil_results.shape[2]);
    ncout.createDimension('Nprocparams',Nproc_results.shape[2]);
    ncout.createDimension('datarng',soil_results.shape[1]);
    ncout.createDimension('n_params',12);
    ncout.createDimension('n_obs',obs_solutions_meansd.shape[0]);
    n_its = ncout.createVariable('n_its','f4',('points'))
    n_its[:] = i+1
    n_accepted = ncout.createVariable('n_accepted','f4',('points'))
    n_accepted[:] = len(go)
    soil_res_m = ncout.createVariable('soil_res_m','f4',('datarng','soilparams'))
    soil_res_m.setncattr('cols','D1 = datarng (non-na indices of d15N in global grid), D2 = k_L, k_NH3, k_G, soil_NO3_steady, d15N, SP')
    soil_res_m[:,:] = soil_res_mean[:,:]
    soil_res_s = ncout.createVariable('soil_res_s','f4',('datarng','soilparams'))
    soil_res_s[:,:] = soil_res_sd[:,:]
    Nproc_res_m = ncout.createVariable('Nproc_res_m','f4',('datarng','Nprocparams'))
    Nproc_res_m.setncattr('cols','D1 = datarng (non-na indices of d15N in global grid), D2 = N2O_all, N2_all, NO_all, denit_N2O, nit_N2O, red_N2O')
    Nproc_res_m[:,:] = Nproc_mean[:,:]
    Nproc_res_s = ncout.createVariable('Nproc_res_s','f4',('datarng','Nprocparams'))
    Nproc_res_s[:,:] = Nproc_sd[:,:]
    prior_res = ncout.createVariable('prior_res','f4',('n_params',"rows"))
    prior_res[:,:] = prior_solutions_meansd[:,:]
    atmos_opt_res = ncout.createVariable('atmos_opt_res','f4',('rows',"rows"))
    atmos_opt_res[:,:] = atmos_opt_meansd[:,:]
    obs_res = ncout.createVariable('obs_res','f4',('n_obs',"rows"))
    obs_res[:,:] = obs_solutions_meansd[:,:]
    last_prior = ncout.createVariable('last_prior','f4',('n_params'))
    last_prior[:] = prior_solutions[:,go[-1]]
    print("last prior is "+str(last_prior[:]))
    # full prior res as text also to check coverage of parameter space
    np.savetxt("MCMC_results/AcceptMC_"+name+".txt", np.vstack((accept_MC.astype("int"),accept_MC.astype("int")*0+step_length)).transpose())
    np.savetxt("MCMC_results/PriorSolutions_"+name+".txt", prior_solutions)
    np.savetxt("MCMC_results/AtmosSolutions_"+name+".txt", atmos_opt)
    
#%% model check  

#tmp = model(x=[265,1.,0.4,0.4,1.1,1.06,121,5.4,15.8,10,1,SP_prea[0]])
tmp = model(x= params[:,0],d15Nrandomise=0.05)
res = tmp[0]
res_conc = res[0:int(len(res)/3)]
res_d15 = res[int(len(res)/3):int(2*len(res)/3)]
res_SP = res[int(2*len(res)/3):int(len(res))]
fig, ax = plt.subplots(3,1)
ax[0].plot(tstarts,obs_conc[:,0],"bo")
ax[0].plot(tstarts,res_conc,"cx")
ax[0].legend((["obs","mod"]))
ax[0].set_ylabel("N2O trop (ppb)")
ax[1].plot(tstarts,obs_d15[:,0],"bo")
ax[1].plot(tstarts,res_d15,"cx")
ax[1].set_ylabel("d15N (permil)")
ax[2].plot(tstarts,obs_SP[:,0],"bo")
ax[2].plot(tstarts,res_SP,"cx")
ax[2].set_ylabel("SP (permil)")

#%% not currently functioning code to plot post obs also...
obs_post = "Y"
if obs_post == "Y" :
    obs_post = obs_solutions[:,r[i]]
    obs_post_conc = obs_post[0:int(len(obs_post)/3)]
    obs_post_d15 = obs_post[int(len(obs_post)/3):int(2*len(obs_post)/3)]
    obs_post_SP = obs_post[int(2*len(obs_post)/3):int(len(obs_post))]
    ax[0].plot(np.arange(1940,2018,2),obs_post_conc,"mx")
    ax[1].plot(np.arange(1940,2018,2),obs_post_d15,"mx")
    ax[2].plot(np.arange(1940,2018,2),obs_post_SP,"mx")
    ax[0].legend((["obs","mod","obs_post"]))
soil = tmp[3]; print(str(round(np.nanmean(soil[:,2])*100,2)),"% lost to gas; base = 7.6%")
Nproc = tmp[4]; print(str(round(np.nanmean(Nproc[:,0])*100,2)),"% of gas is N2O; base = 25%")
print(str(round(np.nanmean(Nproc[:,3])*100,2)),"% denit; base = 70%")



