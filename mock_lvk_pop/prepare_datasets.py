import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import InterpolatedUnivariateSpline as spline
from astropy.cosmology import Planck15
import utils
from scipy.interpolate import RegularGridInterpolator
import scipy.stats as stats
from scipy.stats import gaussian_kde
from popsummary.popresult import PopulationResult
import os


thetas_draw=utils.draw_thetas(10000)
kde_theta=gaussian_kde(thetas_draw,bw_method='scott')

seeds=[5]


no_selection=0
small_errors=0
sample_in_q=1
sample_in_mtot=0
sample_in_z=0
sample_in_source=0
nsamples_ev=5000
#nevs=153
nevs=259
# large_snr=1
very_large_snr=0
snr_th=8
delta_pop=0
log_normal_pop=0
fixed_error=0
no_error=0
draw_flat_z=0

q_no_gap=1


m_min,m_max=2.,400.

snr_th0=8

theta_max=1.

sigma0_chirpm=0.08*snr_th0
sigma0_mtot=0.1*snr_th0
sigma0_eta=0.022*snr_th0
sigma0_q=0.15*snr_th0
sigma0_theta=0.21*snr_th0

zmax=2.

settings={}

settings['zmin'],settings['zmax']=0,zmax
zs_grid=np.linspace(0,zmax,1000)
dVc=np.copy(Planck15.differential_comoving_volume(zs_grid))*4.*np.pi
dVc_spl=spline(zs_grid,dVc)

settings['dVc_spl']=dVc_spl

dls=np.copy(Planck15.luminosity_distance(zs_grid))
zs_dl_spline=spline(dls,zs_grid)
dl_zs_spline=spline(zs_grid,dls)
ddl_zs_spline=dl_zs_spline.derivative()

dl_max=np.amax(dls.value)

#filename = "BBHMassSpinRedshift_BrokenPowerLawTwoPeaks_GaussianComponentSpins_PowerLawRedshift.h5"
#result = PopulationResult(fname=filename)
gwtc5 = '/home/tristan-bruel/Documents/Science/LVK/gwtc5'
result = PopulationResult(fname=f'{gwtc5}/popsummary_files/gwtc5_updated_default_mmax_mass_TwoPeakBrokenPowerLawSmoothedMassDistribution_redshift_PowerLawRedshift_magnitude_iid_spin_magnitude_gaussian_tilt_iid_spin_orientation_popsummary_result.h5')


hyper_samples=result.get_hyperparameter_samples(hyperparameters=['alpha_1', 'alpha_2' , 'break_mass', 'delta_m_1', 'lam_0', 'lam_1', 'lamb', 'mlow_1', 'mpp_1', 'sigpp_1' , 'mpp_2', 'sigpp_2', 'beta'])

# breakpoint()
params={}

id_sample=-3

params['alpha_1']=hyper_samples[id_sample,0]
params['alpha_2']=hyper_samples[id_sample,1]
params['m_break']=hyper_samples[id_sample,2]
params['m1_low']=hyper_samples[id_sample,7]
params['m1_high']=150.
params['mu_1']=hyper_samples[id_sample,8]
params['sigma_1']=hyper_samples[id_sample,9] 
params['mu_2'] = hyper_samples[id_sample,10] 
params['sigma_2'] = hyper_samples[id_sample,11] 
params['lamda_0'] = hyper_samples[id_sample,4] 
params['lamda_1'] = hyper_samples[id_sample,5] 
params['delta_m1'] = hyper_samples[id_sample,3] 

params['kappa']=hyper_samples[id_sample,6] 

params['betaq']=hyper_samples[id_sample,12] 
# params['betaq']=4

params['mg_low']=45
params['mg_up']=115





# dl_max=np.inf


file=np.load('grid_snr.npz')
m1s_grid=file['m1s_grid']
m2s_grid=file['m2s_grid']
# if large_snr==1:
#     snrs_grid=file['snrs_grid']*np.sqrt(3)
# elif very_large_snr==1:
#     snrs_grid=file['snrs_grid']*100
# else:
snrs_grid=file['snrs_grid']
    
snr_interp=RegularGridInterpolator([m1s_grid,m2s_grid],snrs_grid)
Dfix=500

print(np.amin(m1s_grid),np.amax(m1s_grid),np.amin(m2s_grid),np.amax(m2s_grid))


for seed in seeds:
    np.random.seed(seed)

    batch_size = 500  # number of events drawn at once
    events = np.zeros((100 * nevs, 3))
    snrs = np.zeros(100 * nevs)
    

    events_obs0 = np.zeros((nevs, 3))
    theta_events_obs0 = np.zeros(nevs)
    snrs_events_obs0 = np.zeros(nevs)
    snrs_obs0 = np.zeros(nevs)

    ndraw = 0
    nobs = 0
    num_events_1 = 0

    # Optional: for consistent plotting color
    color = None  

    while nobs < nevs:
        print(nobs, nevs)

        # --- Expand memory if needed ---
        if ndraw + batch_size >= len(events):
            events_copy = np.copy(events)
            events = np.zeros((2 * len(events_copy), 3))
            events[:len(events_copy)] = events_copy

            snrs_copy=np.copy(snrs)
            snrs=np.zeros(2*len(snrs))
            snrs[:len(snrs_copy)]=snrs_copy

        # --- Draw batch of events ---
        event_batch = utils.draw_events(batch_size, params, settings,q_no_gap=q_no_gap )
       

        nbatch = len(event_batch)
        events[ndraw:ndraw + nbatch] = event_batch

        # --- Compute derived quantities ---
        z_i = event_batch[:, 2]
        dl_i = dl_zs_spline(z_i)
        m1_i = event_batch[:, 0]
        q_i = event_batch[:, 1]
        m2_i=q_i*m1_i
        theta_i = utils.draw_thetas(nbatch)

        snr_i = snr_interp(np.column_stack([m1_i * (1. + z_i), m2_i * (1. + z_i)])) \
                * theta_i * Dfix / dl_i

        # --- Add measurement noise ---
        if no_error == 0:
            if sample_in_z == 1 or no_selection == 0:
                snr_obs_i = snr_i + np.random.normal(0, 1, nbatch)
            else:
                snr_obs_i = snr_i + np.random.normal(0, 1, nbatch)
                mask_snr = snr_obs_i <= 0
                while mask_snr.any():
                    snr_obs_i[mask_snr] = snr_i[mask_snr] + np.random.normal(0, 1, mask_snr.sum())
                    mask_snr = snr_obs_i <= 0
        else:
            snr_obs_i = snr_i

        # try:
        snrs[ndraw:ndraw + nbatch] = snr_obs_i
        # except:
            # breakpoint()
        # --- Selection ---
        if no_selection == 0:
            mask_obs = snr_obs_i > snr_th
        else:
            mask_obs = np.ones(nbatch, dtype=bool)

        nsel = np.sum(mask_obs)
        if nsel > 0:
            # prevent overflow if we’re close to nevs
            nadd = min(nsel, nevs - nobs)

            events_obs0[nobs:nobs + nadd] = event_batch[mask_obs][:nadd]
            theta_events_obs0[nobs:nobs + nadd] = theta_i[mask_obs][:nadd]
            snrs_events_obs0[nobs:nobs + nadd] = snr_i[mask_obs][:nadd]
            snrs_obs0[nobs:nobs + nadd] = snr_obs_i[mask_obs][:nadd]

            nobs += nadd

        ndraw += nbatch

        # --- Optional: live plotting (consistent color) ---
       

    # --- Trim arrays to actual number of draws ---
    events = events[:ndraw]
    snrs = snrs[:ndraw]
    

    # breakpoint()

    zs_events0=events_obs0[:,2]
    m1s_events0=events_obs0[:,0]
    qs_events0=events_obs0[:,1]
    m2s_events0=m1s_events0*qs_events0
    mtots_events0=m1s_events0+m2s_events0
    chirpmz_events0=(1.+zs_events0)*((m1s_events0**3*m2s_events0**3)/(m1s_events0+m2s_events0))**(1./5.)
    mtotz_events0=mtots_events0*(1.+zs_events0)
    eta_events0=m1s_events0*m2s_events0/((m1s_events0+m2s_events0)**2)

    if no_error==0:
        chirpmz_obs0=chirpmz_events0+np.random.normal(0,1,len(snrs_obs0))*(sigma0_chirpm/snrs_obs0)
        q_obs0=qs_events0+np.random.normal(0,1,len(snrs_obs0))*(sigma0_q/snrs_obs0)
        theta_obs0=theta_events_obs0+np.random.normal(0,1,len(snrs_obs0))*(sigma0_theta/snrs_obs0)

        eta_obs0=q_obs0/((1.+q_obs0)**2)

        mtotz_obs0=chirpmz_obs0*eta_obs0**(-3./5.)
        
        if sample_in_z==0:

            m1z_obs0=mtotz_obs0/(1.+q_obs0)
            m2z_obs0=mtotz_obs0*q_obs0/(1.+q_obs0)

            dl_obs0=np.zeros_like(mtotz_obs0)

            inds_snr_comp=np.argwhere((m1z_obs0>np.amin(m1s_grid)) & (m1z_obs0<np.amax(m1s_grid)) & (m2z_obs0>np.amin(m2s_grid)) & (m2z_obs0<np.amax(m2s_grid)) & (theta_obs0>0.) & (theta_obs0<theta_max))[:,0]

            dl_obs0[inds_snr_comp]=(snr_interp(np.array([m1z_obs0[inds_snr_comp],m2z_obs0[inds_snr_comp]]).T)*theta_obs0[inds_snr_comp]/snrs_obs0[inds_snr_comp])*Dfix

            z_obs0=zs_dl_spline(dl_obs0)


        print('doing samples')
        if sample_in_z==0:
            
            snrs_obs=np.repeat(snrs_obs0,nsamples_ev)+np.random.normal(0,1,nsamples_ev*nevs)
            
        
        chirpmz_obs=np.repeat(chirpmz_obs0,nsamples_ev)+np.random.normal(0,1,nsamples_ev*nevs)*(sigma0_chirpm/np.repeat(snrs_obs0,nsamples_ev))
        q_obs=np.repeat(q_obs0,nsamples_ev)+np.random.normal(0,1,nsamples_ev*nevs)*(sigma0_q/np.repeat(snrs_obs0,nsamples_ev))
        theta_obs=np.repeat(theta_obs0,nsamples_ev)+np.random.normal(0,1,nsamples_ev*nevs)*(sigma0_theta/np.repeat(snrs_obs0,nsamples_ev))

        eta_obs=q_obs/((1.+q_obs)**2)

        mtotz_obs=chirpmz_obs*eta_obs**(-3./5.)
        
        if sample_in_z==0:
            m1z_obs=mtotz_obs/(1.+q_obs)
            m2z_obs=mtotz_obs*q_obs/(1.+q_obs)

            dl_obs=np.zeros_like(mtotz_obs)

            inds_snr_comp=np.argwhere((m1z_obs>np.amin(m1s_grid)) & (m1z_obs<np.amax(m1s_grid)) & (m2z_obs>np.amin(m2s_grid)) & (m2z_obs<np.amax(m2s_grid)) & (theta_obs>0.) & (theta_obs<theta_max) & (snrs_obs>0))[:,0]

            dl_obs[inds_snr_comp]=(snr_interp(np.array([m1z_obs[inds_snr_comp],m2z_obs[inds_snr_comp]]).T)*theta_obs[inds_snr_comp]/snrs_obs[inds_snr_comp])*Dfix

            nfix=np.sum((m1z_obs<np.amin(m1s_grid)) | (m1z_obs>np.amax(m1s_grid)) | (m2z_obs<np.amin(m2s_grid)) | (m2z_obs>np.amax(m2s_grid)) | (q_obs>1.) | (q_obs<0.) | (theta_obs<0.) | (theta_obs>theta_max) | (dl_obs>dl_max) | (snrs_obs<0))

        else:

            if sample_in_source==0:
                nfix=np.sum((q_obs>1.) | (q_obs<0.) | (chirpmz_obs<0) | (z_obs<0) | (z_obs>zmax))
            else:
                nfix=np.sum((q_obs>1.) | (q_obs<0.) | (mtot_obs<0) | (z_obs<0) | (z_obs>zmax))

        while nfix>0:

            print(nfix)

            if sample_in_z==0:
                inds_fix=np.argwhere((m1z_obs<np.amin(m1s_grid)) | (m1z_obs>np.amax(m1s_grid)) | (m2z_obs<np.amin(m2s_grid)) | (m2z_obs>np.amax(m2s_grid))| (q_obs>1.) | (q_obs<0.) | (theta_obs<0.) | (theta_obs>theta_max) | (dl_obs>dl_max) | (snrs_obs<0))[:,0]

                # print('low m1:',np.sum(m1z_obs<np.amin(m1s_grid)))
                # print('large m1:',np.sum(m1z_obs>np.amax(m1s_grid)))
                # print('low m2:',np.sum(m2z_obs<np.amin(m2s_grid)))
                # print('large m2:',np.sum(m2z_obs>np.amax(m2s_grid)))
                # print('large q:',np.sum(q_obs>1.))
                # print('low q:',np.sum(q_obs<0))
                # print('low theta:',np.sum(theta_obs<0.))
                # print('large theta:',np.sum(theta_obs>theta_max))
                # print('large_dl:',np.sum((dl_obs>dl_max)))
                # print('low snr:',np.sum(snrs_obs<0))
                # (m1z_obs<np.amin(m1s_grid)) | (m1z_obs>np.amax(m1s_grid)) | (m2z_obs<np.amin(m2s_grid)) | (m2z_obs>np.amax(m2s_grid))| (q_obs>1.) |  (theta_obs<0.) | (theta_obs>1.) | (dl_obs>dl_max) | (snrs_obs<0)
            else:
                if sample_in_source==0:
                    inds_fix=np.argwhere((q_obs>1.) | (q_obs<0.) | (chirpmz_obs<0) | (z_obs<0) | (z_obs>zmax))[:,0]
                else:
                    inds_fix=np.argwhere((q_obs>1.) | (q_obs<0.) | (mtot_obs<0) | (z_obs<0) | (z_obs>zmax))[:,0]

            if sample_in_z==0:
                
                snrs_obs[inds_fix]=np.repeat(snrs_obs0,nsamples_ev)[inds_fix]+np.random.normal(0,1,nfix)
                

            chirpmz_obs[inds_fix]=np.repeat(chirpmz_obs0,nsamples_ev)[inds_fix]+np.random.normal(0,1,nfix)*(sigma0_chirpm/snrs_obs[inds_fix])
            q_obs[inds_fix]=np.repeat(q_obs0,nsamples_ev)[inds_fix]+np.random.normal(0,1,nfix)*(sigma0_q/snrs_obs[inds_fix])
            theta_obs[inds_fix]=np.repeat(theta_obs0,nsamples_ev)[inds_fix]+np.random.normal(0,1,nfix)*(sigma0_theta/snrs_obs[inds_fix])
            
            eta_obs[inds_fix]=q_obs[inds_fix]/((1.+q_obs[inds_fix])**2)

            mtotz_obs[inds_fix]=chirpmz_obs[inds_fix]*eta_obs[inds_fix]**(-3./5.)

            if sample_in_z==0:
                m1z_obs[inds_fix]=mtotz_obs[inds_fix]/(1.+q_obs[inds_fix])
                m2z_obs[inds_fix]=mtotz_obs[inds_fix]*q_obs[inds_fix]/(1.+q_obs[inds_fix])

                inds_fix_snr_comp=np.argwhere((m1z_obs[inds_fix]>np.amin(m1s_grid)) & (m1z_obs[inds_fix]<np.amax(m1s_grid)) & (m2z_obs[inds_fix]>np.amin(m2s_grid)) & (m2z_obs[inds_fix]<np.amax(m2s_grid)) & (theta_obs[inds_fix]>0.) & (theta_obs[inds_fix]<theta_max) & (snrs_obs[inds_fix]>0))[:,0]

                dl_obs_fix=np.copy(dl_obs[inds_fix])
                dl_obs_fix[inds_fix_snr_comp]=(snr_interp(np.array([m1z_obs[inds_fix][inds_fix_snr_comp],m2z_obs[inds_fix][inds_fix_snr_comp]]).T)*theta_obs[inds_fix][inds_fix_snr_comp]/snrs_obs[inds_fix][inds_fix_snr_comp])*Dfix
                dl_obs[inds_fix]=np.copy(dl_obs_fix)

                nfix=np.sum((m1z_obs<np.amin(m1s_grid)) | (m1z_obs>np.amax(m1s_grid)) | (m2z_obs<np.amin(m2s_grid)) | (m2z_obs>np.amax(m2s_grid))| (q_obs>1.) | (q_obs<0.) | (theta_obs<0.) | (theta_obs>theta_max) | (dl_obs>dl_max) | (snrs_obs<0))

            else:

                if sample_in_source==0:
                    nfix=np.sum((q_obs>1.) | (q_obs<0.) | (chirpmz_obs<0) | (z_obs<0) | (z_obs>zmax))
                else:
                    nfix=np.sum((q_obs>1.) | (q_obs<0.) | (mtot_obs<0) | (z_obs<0) | (z_obs>zmax))


        if sample_in_z==0:
            print(np.amin(m1z_obs),np.amax(m1z_obs),np.amin(m2z_obs),np.amax(m2z_obs))
        # breakpoint()
        
        if sample_in_z==0:
            z_obs=zs_dl_spline(dl_obs)

            z_samples=z_obs.reshape(nevs,nsamples_ev)
            # print('stds:',np.std(z_samples,axis=1))

        if sample_in_source==0:
            mtot_obs=mtotz_obs/(1+z_obs)
        
        print(np.amin(z_obs),np.amax(z_obs))

        # breakpoint()

        print('computing prior')
       
            
    
        priors=np.absolute((snrs_obs/dl_obs)*ddl_zs_spline(z_obs))
          
        priors/=kde_theta.evaluate(theta_obs)

        priors*=q_obs*(chirpmz_obs/(m1z_obs*m2z_obs))*(1+z_obs)**2
        


        # print(priors)

        if sample_in_z==1:
            theta_obs0=np.repeat(theta_events_obs0,nsamples_ev)
            snrs_obs=np.repeat(snrs_obs0,nsamples_ev)

    else:

        mtot_obs=np.copy(mtots_events0)
        q_obs=np.copy(qs_events0)
        z_obs=np.copy(zs_events0)
        snrs_obs=np.copy(snrs_obs0)
        priors=np.ones_like(mtot_obs)

        mtotz_obs0=np.copy(mtot_obs*(1+z_obs))
        q_obs0=np.copy(q_obs)
        z_obs0=np.copy(z_obs)
        theta_obs0=np.copy(theta_events_obs0)


        nsamples_ev=1


    
    m1_obs=mtot_obs/(1.+q_obs)
    m2_obs=mtot_obs*q_obs/(1.+q_obs)

    if q_no_gap==0:
        name_out='samples_mass_gap_pop'
    elif q_no_gap==1:
        name_out='samples_no_mass_gap_pop'

    if draw_flat_z==1:
        name_out+='_draw_flat_z'

    name_out+='_%d_evs'%nevs

    if no_error==0:

        name_out+='_%d_samples'%nsamples_ev

    else:

        name_out+='_no_error'

    if no_selection==0 and snr_th!=8:
        name_out+='_snr_th_%d'%snr_th

    if no_selection==1:
        name_out+='_no_selection'

    if sample_in_z==1:
        name_out+='_sample_in_z'

    if sample_in_q==1:
        name_out+='_sample_in_q'

    if sample_in_mtot==1:
        name_out+='_sample_in_mtot'

    if sample_in_source==1:
        name_out+='_sample_in_source'

    # if no_selection==0 and no_error==0:
    #     if large_snr==1:
    #         name_out+='_large_snr'

    #     if very_large_snr==1:
    #         name_out+='_very_large_snr'

    if fixed_error==1:
        name_out+='_fixed_error'

    if small_errors==1:
        name_out+='_small_errors'

    

    if zmax!=5:
        name_out+='_zmax_%1.1f'%zmax

    name_out+='_seed_%d'%seed

    print(name_out)

    if not os.path.exists('datasets/'):
        os.mkdir('datasets/')
    np.savez('datasets/'+name_out+'.npz',m1_obs=m1_obs,m2_obs=m2_obs,z_obs=z_obs,priors=priors,nsamples_ev=nsamples_ev)
    np.savez('datasets/'+name_out+'_aux.npz',events=events,events_obs0=events_obs0,theta_obs0=theta_obs0,snrs_events_obs0=snrs_events_obs0,snrs_obs0=snrs_obs0,snrs_obs=snrs_obs,num_events_1=num_events_1,mtotz_obs0=mtotz_obs0,q_obs0=q_obs0,z_obs0=z_obs0)

breakpoint()
