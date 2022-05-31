#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 13:17:17 2020

@author: elizaharris
"""

#%% Define the model

#x = params[:,0]
def model(x,fullres="N",d15Nrandomise="N"):
    c_prea_new=[x[0],7.5]
    scale_fitN2_new = np.array((1.,x[1],1.,1.))
    fertEFred_new = x[3]
    temp_sens_new = x[4]
    ltPD_new = [x[6],9]
    ltPI_new = [ltPD_new[0]*x[5],9]
    d15_ocean_new = [x[7],1.9]
    SP_ocean_new = [x[8],7.1]
    d15_prea_new = [x[9],2.0]
    scale_fitNO_new = np.array((1.,x[10],1.,1.))
    SP_prea_new = [x[11],2.0]
    
    # reduce the "expression" of fractionation by a certain factor
    frac_expression = x[2]
    E_nit_new = [E_nit[0]*frac_expression,E_nit[1]]
    E_denit_new = [E_denit[0]*frac_expression,E_denit[1]]
    E_denit_N2O_new = [E_denit_N2O[0]*frac_expression,E_denit_N2O[1]]
    E_red_new = [E_red[0]*frac_expression,E_red[1]]
    E_SP_nit_new = [E_SP_nit[0]*frac_expression,E_SP_nit[1]]
    E_SP_denit_new = [E_SP_denit[0]*frac_expression,E_SP_denit[1]]
    E_SP_red_new = [E_SP_red[0]*frac_expression,E_SP_red[1]]
    
    #%% Run the soil model
    
    # check time: if too slow, would be possible to use "multiproc" library in python, which runs the loop but each bit gets sent to a different processer to speed up
    start_time = time.time()
    
    # get non-nan d15N values as a vector
    datarng = np.where(~np.isnan(d15N_grid) & ~np.isinf(d15N_grid))
    d15N_vec = d15N_grid[datarng]
    # if d15N randomise option is selected, randomise using gridded uncertainty
    if d15Nrandomise!="N":
        tmp = np.random.randn(len(d15N_vec))*d15Nrandomise
        d15N_vec = d15N_vec + tmp*d15Nerr_grid[datarng]
    # get the partitioning for each d15N point; rows = d15N points; cols = N2O_all, N2_all, NO_all, denit_N2O, nit_N2O, red_N2O
    WFPS_vec = WFPS_grid[datarng]
    gaspart = para.gp_gaspart(WFPS_vec,scale_fitNO = scale_fitNO_new,scale_fitN2 = scale_fitN2_new,plotYN="N")
    N2Opart = para.gp_nitdenit(WFPS_vec,scale_fitnit = scale_fitnit,plotYN="N")
    N2Opart["red_N2O"] = 1 - gaspart["N2O_all"]/(gaspart["N2O_all"]+gaspart["N2_all"])
    N_processes_vec = np.array(np.vstack((gaspart["N2O_all"],gaspart["N2_all"],gaspart["NO_all"],
                                 N2Opart["denit_N2O"],N2Opart["nit_N2O"],N2Opart["red_N2O"])).transpose())
    fNH3_vec = fNH3_grid[datarng]
    fNH3_vec[np.isnan(fNH3_vec)] = 0.04
    # run model on the vector (output = k_L k_NH3 k_G soil_d15NNO3_final N2O_d15N N2O_SP, rows = d15N values)
    tmp = isomodel.IsoModel(d15N_vec,N_processes_vec,fNH3_vec=fNH3_vec,fignum=np.nan,d15N_inat=d15N_inat,E_NH3=E_NH3,E_L=E_L,
                            E_nit=E_nit_new,E_denit=E_denit_new,E_denit_N2O=E_denit_N2O_new,E_red=E_red_new,E_SP_nit=E_SP_nit_new,E_SP_denit=E_SP_denit_new,
                            E_SP_red=E_SP_red_new,f_soilpoolloss=f_soilpoolloss)
    noconverg = np.isnan(tmp[:,0])
    tmp[noconverg,2] = 0.0 # no gas production where model doesn't converge
    tmp[noconverg,1] = fNH3_vec[noconverg]
    tmp[noconverg,0] = 1 - tmp[noconverg,1]
    
    # allocate results
    k_L = np.zeros(d15N_grid.shape)+np.nan; k_L[datarng] = tmp[:,0]
    k_G = np.zeros(d15N_grid.shape)+np.nan; k_G[datarng] = tmp[:,2]
    f_denit = np.zeros(d15N_grid.shape)+np.nan; f_denit[datarng] = N_processes_vec[:,3]
    f_N2O = np.zeros(d15N_grid.shape)+np.nan; f_N2O[datarng] = N_processes_vec[:,0]
    d15N_modoutput = np.zeros(d15N_grid.shape)+np.nan; d15N_modoutput[datarng] = tmp[:,3]
    N2O_d15N = np.zeros(d15N_grid.shape)+np.nan; N2O_d15N[datarng] = tmp[:,4]
    N2O_SP = np.zeros(d15N_grid.shape)+np.nan; N2O_SP[datarng] = tmp[:,5]
    soilmodres = tmp
    
    t_s = time.time()
    
    #%% 2. Calculate N emissions by multiplying soil model with inputs
    
    Nsummary = np.zeros((years.shape[0],10)) 
    Nsummary[:,0] = years
    Nsummary_full = np.zeros((years.shape[0],23)) 
    Nsummary_full[:,0] = years
    
    # unit/area conversion factor
    factor = (k_G*f_N2O*area_grid)/1000/1000/1000/1000 
    factor = factor[datarng]
    factor_area = (area_grid/1000/1000/1000/1000)[datarng] # from g m-2 y-1 to Tg y-1 per gridcell
    
    # find the years to use - nearest covered by the dataset
    datayears = years.copy()
    datayears[years<min(datarange)] = min(datarange)
    datayears[years>max(datarange)] = max(datarange)
    # cycle through the years to find integrated fluxes etc. for each year
    for n in range(0,years.shape[0]) :
        # get values for year n for fert, dep, fix
        index = int(np.where(datarange-datayears[n]==0)[0]) # find the "row" of the N input data for this year
        fix = Ninputs.variables["fixation"][index,:,:].data 
        fix = fix[datarng]*scaleFix
        dep = Ninputs.variables["deposition"][index,:,:].data
        dep = dep[datarng]*scaleDep
        fert = Ninputs.variables["fertilisation"][index,:,:].data
        fert = fert[datarng] 
        if n == 0: fix0 = fix; dep0 = dep; fert0 = fert # save initial year values for calculating temp sensitivity
        
        # calculate temperature sens
        tindex = int(np.where(abs(Tanom.variables["time"]-years[n]) == np.min(abs(Tanom.variables["time"]-years[n])))[0])
        tanomaly = Tnorm[tindex,:,:]
        tanomaly = tanomaly[datarng]
        tanomaly_sens = (temp_sens_new-1)*tanomaly + 1 # multiply gas production in each grid cell by this number
        #fix = fix + fix0*(temp_sens_new-1)*tanomaly
        #dep = dep + dep0*(temp_sens_new-1)*tanomaly
        #fert = fert + fert0*(temp_sens_new-1)*tanomaly
        
        # get EDGAR direct anthrop emissions (g N2O-N m-2 y-1)
        index = int(np.where(f.variables["time"][:]-datayears[n]==0)[0]) # find the "row" of the N input data for this year
        TRO = TRO_grid[index,:,:][datarng]
        WWT = WWT_grid[index,:,:][datarng]
        CHE = CHE_grid[index,:,:][datarng]
        ENE = ENE_grid[index,:,:][datarng]
    
        # 0-6: year, total emissions (Tg N-N2O a-1), emissions from fert, dep, fix, mean d15N, mean SP, EDGAR total, mean d15N, mean SP for soil only
        Nsummary[n,2] = np.nansum(factor*fert*tanomaly_sens*fertEFred_new) # Tg N /year, fert emissions (fertEFred lost emissions assumed to be harvested)
        Nsummary[n,3] = np.nansum(factor*dep*tanomaly_sens) # Tg N /year, dep emissions
        Nsummary[n,4] = np.nansum(factor*fix*tanomaly_sens) # Tg N /year, fix emissions
        Nsummary[n,7] = np.nansum(factor_area*TRO)+np.nansum(factor_area*WWT)+np.nansum(factor_area*CHE)+np.nansum(factor_area*ENE) # Tg N /year, EDGAR total emissions
        factor1 = factor * tanomaly_sens * (fix+dep) # summed fert emissions
        factor2 = factor * tanomaly_sens * (fert*fertEFred_new)
        factor12sum = np.nansum(factor1+factor2) # total soil emissions for the year
        Nsummary[n,8] = np.nansum(factor1*N2O_d15N[datarng] + factor2*(N2O_d15N[datarng]+4.5))/factor12sum # d15N for soil emissions only; account for fert N inputs being 4.5 permil heavier!
        Nsummary[n,9] = np.nansum(factor1*N2O_SP[datarng] + factor2*N2O_SP[datarng])/factor12sum # SP
        Nsummary[n,5] = ( np.nansum(factor1*N2O_d15N[datarng]) + np.nansum(factor2*(N2O_d15N[datarng]+4.5)) + np.nansum(factor_area*TRO)*d15_TRO[0] + np.nansum(factor_area*WWT)*d15_WWT[0] + np.nansum(factor_area*CHE)*d15_CHE[0] + np.nansum(factor_area*ENE)*d15_ENE[0] ) / (factor12sum+np.nansum(factor_area*TRO)+np.nansum(factor_area*WWT)+np.nansum(factor_area*CHE)+np.nansum(factor_area*ENE)) # d15N for total emissions (soil + EDGAR)
        Nsummary[n,6] = ( np.nansum(factor1*N2O_SP[datarng]) + np.nansum(factor2*N2O_SP[datarng]) + np.nansum(factor_area*TRO)*SP_TRO[0] + np.nansum(factor_area*WWT)*SP_WWT[0] + np.nansum(factor_area*CHE)*SP_CHE[0] + np.nansum(factor_area*ENE)*SP_ENE[0] ) / (factor12sum+np.nansum(factor_area*TRO)+np.nansum(factor_area*WWT)+np.nansum(factor_area*CHE)+np.nansum(factor_area*ENE)) # SP
        
        # break down emissions more closely
        if fullres == "Y":
            Nsummary_full[n,1] = np.nansum(factor_area*fert) # inputs from fert dep fix
            Nsummary_full[n,2] = np.nansum(factor_area*dep)
            Nsummary_full[n,3] = np.nansum(factor_area*fix)
            Nsummary_full[n,4] = np.nansum(factor*fert*tanomaly_sens*fertEFred_new*f_denit[datarng]) # N2O Tg n/year from denit, fert
            Nsummary_full[n,5] = np.nansum(factor*dep*tanomaly_sens*f_denit[datarng]) # Tg N /year, from denit, dep 
            Nsummary_full[n,6] = np.nansum(factor*fix*tanomaly_sens*f_denit[datarng]) # Tg N /year, from denit, fix 
            Nsummary_full[n,7] = np.nansum(factor_area*fert*fertEFred_new*tanomaly_sens*k_G[datarng]*N_processes_vec[:,1]) # N2 from fert dep fix
            Nsummary_full[n,8] = np.nansum(factor_area*dep*tanomaly_sens*k_G[datarng]*N_processes_vec[:,1])
            Nsummary_full[n,9] = np.nansum(factor_area*fix*tanomaly_sens*k_G[datarng]*N_processes_vec[:,1])
            Nsummary_full[n,10] = np.nansum(factor_area*fert*tanomaly_sens*fertEFred_new*k_G[datarng]*N_processes_vec[:,2]) # NO from fert dep fix
            Nsummary_full[n,11] = np.nansum(factor_area*dep*tanomaly_sens*k_G[datarng]*N_processes_vec[:,2])
            Nsummary_full[n,12] = np.nansum(factor_area*fix*tanomaly_sens*k_G[datarng]*N_processes_vec[:,2])
            Nsummary_full[n,13] = np.nansum(factor_area*fert*fertEFred_new*tanomaly_sens*fNH3_vec) # NH3 from fert dep fix
            Nsummary_full[n,14] = np.nansum(factor_area*dep*tanomaly_sens*fNH3_vec)
            Nsummary_full[n,15] = np.nansum(factor_area*fix*tanomaly_sens*fNH3_vec)
            Nsummary_full[n,16] = np.nansum(factor_area*fert*fertEFred_new* (k_L[datarng]-(tanomaly_sens-1)*(1-k_L[datarng])) ) # leaching; reduced by temp sens of other processes
            Nsummary_full[n,17] = np.nansum(factor_area*dep* (k_L[datarng]-(tanomaly_sens-1)*(1-k_L[datarng])) )
            Nsummary_full[n,18] = np.nansum(factor_area*fix* (k_L[datarng]-(tanomaly_sens-1)*(1-k_L[datarng])) )
            
        # 0-6 : year, N input fert, dep, fix (Tg N a-1), N2O denit only fert, dep, fix (Tg N-N2O a-1), 
        # 7-12: N2 fert dep fix, NO fert dep fix, 
        # 13-18: NH3 fert dep fix, leach fert dep fix
    
    Nsummary[:,1] = Nsummary[:,2]+Nsummary[:,3]+Nsummary[:,4]+Nsummary[:,7]
    t_e = time.time()
        
    #%% 3. Atmosphere model
     
    tmp = atmodel.atmos_model(Nsummary,c_prea=c_prea_new,d15_prea=d15_prea_new,SP_prea=SP_prea_new,d15_ocean=d15_ocean_new,SP_ocean=SP_ocean_new,ltPI =ltPI_new,ltPD = ltPD_new)
    T_Sm_new=tmp[1]; F_ocean_new=tmp[2]; tmp=tmp[0]
    results = tmp[:,0:5]
    results_d15 = tmp[:,6:8]
    results_SP = tmp[:,9:11]
    t_a = time.time()
    
    #print("Total run time: %s seconds" % (time.time() - start_time))
    #print("Soil model: %s seconds" % (t_s - start_time))
    #print("Emissions calc: %s seconds" % (t_e - t_s))
    #print("Atmos model: %s seconds" % (t_a - t_e))
    
    res_conc = np.interp(tstarts,results[:,0],results[:,2])
    res_d15 = np.interp(tstarts,results[:,0],results_d15[:,0])
    res_SP = np.interp(tstarts,results[:,0],results_SP[:,0])
    res = np.hstack((res_conc,res_d15,res_SP)).transpose()
    
    #%% 4. Return results to match observations
    if fullres=="N": return(res,T_Sm_new,F_ocean_new,soilmodres,N_processes_vec,Nsummary)
    if fullres=="Y": return(res,T_Sm_new,F_ocean_new,soilmodres,N_processes_vec,Nsummary,Nsummary_full)
    # fullres = concisores, T_Sm