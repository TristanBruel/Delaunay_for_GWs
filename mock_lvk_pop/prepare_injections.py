import numpy as np
import matplotlib.pyplot as plt
import utils

from scipy.interpolate import InterpolatedUnivariateSpline as spline
from astropy.cosmology import Planck15

from tqdm import tqdm

import seaborn as sns
from popsummary.popresult import PopulationResult
from scipy.interpolate import RegularGridInterpolator

import os


file=np.load('grid_snr.npz')
m1s_grid=file['m1s_grid']
m2s_grid=file['m2s_grid']
snrs_grid=file['snrs_grid']


Dfix=500

snr_interp=RegularGridInterpolator([m1s_grid,m2s_grid],snrs_grid)


#filename = "BBHMassSpinRedshift_BrokenPowerLawTwoPeaks_GaussianComponentSpins_PowerLawRedshift.h5"
#result = PopulationResult(fname=filename)
gwtc5 = '/home/tristan-bruel/Documents/Science/LVK/gwtc5'
result = PopulationResult(fname=f'{gwtc5}/popsummary_files/gwtc5_updated_default_mmax_mass_TwoPeakBrokenPowerLawSmoothedMassDistribution_redshift_PowerLawRedshift_magnitude_iid_spin_magnitude_gaussian_tilt_iid_spin_orientation_popsummary_result.h5')


hyper_samples=result.get_hyperparameter_samples(hyperparameters=['alpha_1', 'alpha_2' , 'break_mass', 'delta_m_1', 'lam_0', 'lam_1', 'lamb', 'mlow_1', 'mpp_1', 'sigpp_1' , 'mpp_2', 'sigpp_2', 'beta'])

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

# params['betaq']=hyper_samples[id_sample,12] 
params['betaq']=4



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



snr_th=8


ninj=3000000

zs_grid=np.linspace(0,zmax,1000)
dVc=np.copy(Planck15.differential_comoving_volume(zs_grid))*4.*np.pi
dVc_spl=spline(zs_grid,dVc)

dls=np.copy(Planck15.luminosity_distance(zs_grid))
zs_dl_spline=spline(dls,zs_grid)
dl_zs_spline=spline(zs_grid,dls)
ddl_zs_spline=dl_zs_spline.derivative()


seed=0

np.random.seed(seed)



print('drawing')

print('drawing m1')
m1s_inj0=utils.draw_pl(ninj,-params['alpha_1'],params['m1_low'],params['m1_high'])
print('drawing q')
qs_inj0=utils.draw_q_no_gap(ninj,m1s_inj0,params['m1_low'],params['betaq'])
m2s_inj0=m1s_inj0*qs_inj0

print('drawing z')
zs_inj0=utils.draw_z_cdf(ninj,params['kappa'],settings['dVc_spl'],settings['zmax'])


thetas_inj0=utils.draw_thetas(ninj)

dls_injs=dl_zs_spline(zs_inj0)

print(np.amax(m1s_inj0*(1+zs_inj0)))
snrs_inj0=snr_interp(np.array([m1s_inj0*(1+zs_inj0),m2s_inj0*(1+zs_inj0)]).T)*thetas_inj0*Dfix/dls_injs


snrs_inj=snrs_inj0+np.random.normal(0,1,ninj)

mtots_inj0=m1s_inj0+m2s_inj0
qs_inj0=m2s_inj0/m1s_inj0

zs_inj=zs_inj0[snrs_inj>snr_th]
m1s_inj=m1s_inj0[snrs_inj>snr_th]
qs_inj=qs_inj0[snrs_inj>snr_th]

print('computing priors')


pdf_m1=utils.power_law_pdf(m1s_inj,-params['alpha_1'],params['m1_low'],params['m1_high'])

pdf_z=utils.z_pdf(zs_inj,params['kappa'],settings['dVc_spl'],settings['zmax'])

pdf_q=utils.pdf_q_no_gap(qs_inj,m1s_inj,params['m1_low'],params['betaq'])



priors_inj=pdf_m1*pdf_z*pdf_q/m1s_inj


if not os.path.exists('injections/'):
    os.mkdir('injections/')
name_file_injections='injections/injections'


np.savez(name_file_injections+'.npz',m1s_inj=m1s_inj,m2s_inj=m1s_inj*qs_inj,zs_inj=zs_inj,priors_inj=priors_inj,ninj=ninj)

breakpoint()
