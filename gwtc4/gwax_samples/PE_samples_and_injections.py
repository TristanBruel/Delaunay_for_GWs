import numpy as np
import gwax.data as gwax

import os
import argparse

def load_and_save_PE_samples(catalog='GWTC-5', save_dir='./'):
    print('Loading PE samples for catalog:', catalog)
    pe_samples = gwax.get_posteriors(path='/home/tristan.bruel', catalog=catalog, 
            extra_keys=['redshift'], 
            downsample=True, stack=True,
            )

    if catalog == 'GWTC-3':
        filename = 'gwtc3_samples.npz'
    elif catalog == 'GWTC-4':
        filename = 'gwtc4_samples.npz'
    elif catalog == 'GWTC-5':
        filename = 'gwtc5_samples.npz'
    else:
        print('catalog', catalog, 'not available')
    np.savez(os.path.join(save_dir, filename),
            m1s=np.ravel(pe_samples['mass_1']),
            zs=np.ravel(pe_samples['redshift']),
            qs=np.ravel(pe_samples['mass_2']/pe_samples['mass_1']),
            chi1s=np.ravel(pe_samples['a_1']),
            chi2s=np.ravel(pe_samples['a_2']),
            cos_tilt_1s=np.ravel(pe_samples['cos_tilt_1']),
            cos_tilt_2s=np.ravel(pe_samples['cos_tilt_2']),
            priors=1/np.ravel(pe_samples['weight']),
            nevents=pe_samples['mass_1'].shape[0],
            nsamples=pe_samples['mass_1'].shape[1],
            )
    return pe_samples


def load_and_save_injections(catalog='GWTC-5', save_dir='./'):
    print('Loading injections for catalog:', catalog)
    inj_gwtc = gwax.get_injections(path='/home/tristan.bruel', catalog=catalog, 
            extra_keys=['redshift'],
            )
    if catalog == 'GWTC-3':
        filename = 'gwtc3_injections_full.npz'
    elif catalog == 'GWTC-4':
        filename = 'gwtc4_injections_full.npz'
    elif catalog == 'GWTC-5':
        filename = 'gwtc5_injections_full.npz'
    else:
        print('catalog', catalog, 'not available')
    np.savez(os.path.join(save_dir,filename),
            m1s=np.ravel(inj_gwtc['mass_1']),
            zs=np.ravel(inj_gwtc['redshift']),
            qs=np.ravel(inj_gwtc['mass_2']/inj_gwtc['mass_1']),
            chi1s=np.ravel(inj_gwtc['a_1']),
            chi2s=np.ravel(inj_gwtc['a_2']),
            cos_tilt_1s=np.ravel(inj_gwtc['cos_tilt_1']),
            cos_tilt_2s=np.ravel(inj_gwtc['cos_tilt_2']),
            inj_priors=1/np.ravel(inj_gwtc['weight']),
            ninjs=inj_gwtc['total'],
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
    args = parser.parse_args()

    pe_samples = load_and_save_PE_samples(catalog=args.catalog, save_dir=args.save_dir)
    inj_gwtc = load_and_save_injections(catalog=args.catalog, save_dir=args.save_dir)

