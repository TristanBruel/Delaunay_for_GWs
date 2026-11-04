import numpy as np
import matplotlib.pyplot as plt
import utils

from scipy.interpolate import InterpolatedUnivariateSpline as spline
from astropy.cosmology import Planck15

from tqdm import tqdm

import seaborn as sns
from popsummary.popresult import PopulationResult


from gwbench import antenna_pattern_np as antpat
from gwbench.wf_models import lal_bbh_np as lal_bbh
from gwbench import psd
from gwbench import snr
import gwbench.utils


f_lo = 16
f_hi = 2048
df = 2**-4
f = np.arange(f_lo, f_hi, df)

ra_s = gwbench.utils.det_specs_built_in['H']['longitude'] + gwbench.utils.tc_offset * 2*np.pi / gwbench.utils.det_specs_built_in['H']['period']
dec_s = gwbench.utils.det_specs_built_in['H']['latitude']

m1_interp_min,m1_interp_max=2.,500.
m2_interp_min,m2_interp_max=2.,500.

Dfix=500


ngrid=200
m1s_grid=np.linspace(m1_interp_min,m1_interp_max,ngrid)
m2s_grid=np.linspace(m2_interp_min,m2_interp_max,ngrid)


snrs_grid=np.zeros((len(m1s_grid),len(m2s_grid)))

Dfix=500

for i in tqdm(range(ngrid)):
    m1i=m1s_grid[i]
    for j in tqdm(range(0,i+1)):

        # print(j)

        m2j=m2s_grid[j]

        # print(m1i,m2j)

        chirpmij=((m1i**3*m2j**3)/(m1i+m2j))**(1./5.)

        etaij=m1i*m2j/((m1i+m2j)**2)


        Mc = chirpmij #chirp mass
        eta = etaij #symmetric mass ratio
        chi1x = 0. #spin 1 x
        chi1y = 0. #spin 1 y
        chi1z = 0. #spin 1 z
        chi2x = 0. #spin 2 x
        chi2y = 0. #spin 2 y
        chi2z = 0. #spin 2 z
        DL = Dfix #distance in Mpc
        iota = 0. #inclination
        tc = 0 #coalescence time
        phic = 0 #phase at coalescence
        ra = ra_s #right ascension
        dec = dec_s #declination
        psi = 0 #polarisation
        gmst0 = 0
        use_rot = 0

        approximant = 'IMRPhenomXPHM'
        hfp, hfc = lal_bbh.hfpc(f, Mc, eta, chi1x, chi1y, chi1z, chi2x, chi2y, chi2z, DL, tc, phic, iota, approximant=approximant)
        det_psd_ET, ff = psd.psd('aLIGO', f, f_lo=f_lo, f_hi=f_hi)

        locs = ['H']
        snr_sq = 0
        for loc in locs:
            hf = antpat.detector_response(hfp, hfc, f, Mc, eta, tc, ra, dec, psi, loc, use_rot, user_locs=None)
            
            snr_sq += snr.snr_square(hf, det_psd_ET, ff)

        # print(i,j,np.sqrt(snr_sq))
        snrs_grid[i,j] = np.sqrt(snr_sq)
        snrs_grid[j,i] = np.sqrt(snr_sq)



np.savez('grid_snr.npz',m1s_grid=m1s_grid,m2s_grid=m2s_grid,snrs_grid=snrs_grid,Dfix=Dfix)