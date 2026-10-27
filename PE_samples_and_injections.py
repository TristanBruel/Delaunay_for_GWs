import numpy as np
import gwax.data as gwax
import gwax.cosmology as cosmo

import os
import argparse

def load_and_save_PE_samples(catalog='GWTC-5', chi_eff=False, save_dir='./'):
    print('Loading PE samples for catalog:', catalog)
    pe_samples = gwax.get_posteriors(path='/home/tristan.bruel', catalog=catalog, 
            downsample=True,
            chi_eff=chi_eff,
            )

    if catalog == 'GWTC-3':
        filename = 'gwtc3_samples'+chi_eff*'_chieff'+'.npz'
    elif catalog == 'GWTC-4':
        filename = 'gwtc4_samples'+chi_eff*'_chieff'+'.npz'
    elif catalog == 'GWTC-5':
        filename = 'gwtc5_samples'+chi_eff*'_chieff'+'.npz'
    else:
        print('catalog', catalog, 'not available')
    (
        m1s,
        m2s,
        z,
        jac,
    ) = cosmo.detector_to_source(
            pe_samples['mass_1'], 
            pe_samples['mass_2'], 
            pe_samples['luminosity_distance'], 
            67.9, 0.3065,
            )
    prior = 1/pe_samples['weight'] # p(m1,m2,dl) 
    prior *= jac # p(m1,m2,dl) -> p(m1s,m2s,z)
    prior *= m1s # p(m1s,m2s,z) -> p(m1s,q,z)
    if chi_eff:
        np.savez(os.path.join(save_dir, filename),
                m1s=np.ravel(m1s),
                zs=np.ravel(z),
                qs=np.ravel(m2s/m1s),
                chi_effs=np.ravel(pe_samples['chi_eff']),
                priors=np.ravel(prior),
                nevents=pe_samples['mass_1'].shape[0],
                nsamples=pe_samples['mass_1'].shape[1],
                )
    else:
        np.savez(os.path.join(save_dir, filename),
                m1s=np.ravel(m1s),
                zs=np.ravel(z),
                qs=np.ravel(m2s/m1s),
                chi1s=np.ravel(pe_samples['a_1']),
                chi2s=np.ravel(pe_samples['a_2']),
                cos_tilt_1s=np.ravel(pe_samples['cos_tilt_1']),
                cos_tilt_2s=np.ravel(pe_samples['cos_tilt_2']),
                priors=np.ravel(prior),
                nevents=pe_samples['mass_1'].shape[0],
                nsamples=pe_samples['mass_1'].shape[1],
                )
    return pe_samples


def load_and_save_injections(catalog='GWTC-5', chi_eff=False, save_dir='./'):
    print('Loading injections for catalog:', catalog)
    inj_gwtc = gwax.get_injections(path='/home/tristan.bruel', catalog=catalog, 
            extra_keys=['redshift'],
            chi_eff=chi_eff,
            )
    if catalog == 'GWTC-3':
        filename = 'gwtc3_injections_full'+chi_eff*'_chieff'+'.npz'
    elif catalog == 'GWTC-4':
        filename = 'gwtc4_injections_full'+chi_eff*'_chieff'+'.npz'
    elif catalog == 'GWTC-5':
        filename = 'gwtc5_injections_full'+chi_eff*'_chieff'+'.npz'
    else:
        print('catalog', catalog, 'not available')

    (
        m1s,
        m2s,
        z,
        jac,
    ) = cosmo.detector_to_source(
            inj_gwtc['mass_1'],
            inj_gwtc['mass_2'], 
            inj_gwtc['luminosity_distance'],
            67.9, 0.3065,
            )
    prior = 1/inj_gwtc['weight'] # p(m1,m2,dl) 
    prior *= jac # p(m1,m2,dl) -> p(m1s,m2s,z)
    prior *= m1s # p(m1s,m2s,z) -> p(m1s,q,z)
    if chi_eff:
        np.savez(os.path.join(save_dir,filename),
                m1s=np.ravel(m1s),
                zs=np.ravel(z),
                qs=np.ravel(m2s/m1s),
                chi_effs=np.ravel(inj_gwtc['chi_eff']),
                inj_priors=np.ravel(prior),
                ninjs=inj_gwtc['total'],
                analysis_time=inj_gwtc['time'],
                )
    else:
        np.savez(os.path.join(save_dir,filename),
                m1s=np.ravel(m1s),
                zs=np.ravel(z),
                qs=np.ravel(m2s/m1s),
                chi1s=np.ravel(inj_gwtc['a_1']),
                chi2s=np.ravel(inj_gwtc['a_2']),
                cos_tilt_1s=np.ravel(inj_gwtc['cos_tilt_1']),
                cos_tilt_2s=np.ravel(inj_gwtc['cos_tilt_2']),
                inj_priors=np.ravel(prior),
                ninjs=inj_gwtc['total'],
                analysis_time=inj_gwtc['time'],
                )
    return inj_gwtc


##################################################################
### Run it!
###
##################################################################
if __name__ == "__main__":
    # Define command line options
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", dest='catalog', 
            help="GWTC catalog for which to load PE samples and injections", 
            type=str, default='GWTC-5')
    parser.add_argument("--d", dest='save_dir', 
            help="Path to the directory where to save the files", 
            type=str, default='./')
    parser.add_argument("--chi_eff", dest='chi_eff', action='store_true',
            help="Include chi_eff in the PE samples, injections, and prior evalutations", 
            )
    args = parser.parse_args()

    pe_samples = load_and_save_PE_samples(catalog=args.catalog, chi_eff=args.chi_eff, save_dir=args.save_dir)
    inj_gwtc = load_and_save_injections(catalog=args.catalog, chi_eff=args.chi_eff, save_dir=args.save_dir)
