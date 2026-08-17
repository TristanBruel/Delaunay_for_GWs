import numpy as np
import pickle
from itertools import product
import matplotlib.pyplot as plt
import matplotlib as mpl

from scipy import stats, special
from scipy.spatial import Delaunay
from local_utils import delaunaytor
from generate_events import generate_pop, p_det

import os
import argparse
from tqdm import trange, tqdm



def plot_Nevents(prior, Nevents):
    """
    """
    estimated_num_events = np.zeros(len(prior))
    xgrid = np.linspace(-10,10,101)
    ygrid = np.linspace(-10,10,101)
    zgrid = np.linspace(-10,10,101)
    X, Y, Z = np.meshgrid(xgrid, ygrid, zgrid)
    grid = np.c_[X.ravel(), Y.ravel(), Z.ravel()]
    dx = xgrid[1] - xgrid[0]
    dy = ygrid[1] - ygrid[0]
    dz = zgrid[1] - zgrid[0]
    for ind in range(len(prior)):
        this_delo = delaunaytor.CPUDelaunayInterpolator()
        this_delo.triangulate(prior[ind])
        log_rate = this_delo.interpolate(grid).reshape(ygrid.shape[0], xgrid.shape[0], zgrid.shape[0])
        estimated_num_events[ind] = np.sum(np.exp(log_rate)) *dx *dy *dz
    
    #####################################
    # Plotting parameters
    fs = 12
    lw = 1.4
    plt.rcParams['font.size']=fs
    plt.rcParams['font.family']='serif'
    plt.rcParams['font.serif']='cmr10'
    plt.rcParams['mathtext.fontset']='cm'
    plt.rcParams['axes.unicode_minus']=False
    plt.rcParams['axes.formatter.use_mathtext']=True
    plt.rcParams['lines.linewidth']=lw
    plt.rcParams['xtick.labelsize']=fs
    plt.rcParams['ytick.labelsize']=fs
    plt.rcParams['legend.fontsize']=.9*fs

    fig, ax = plt.subplots(1, 1, figsize=(6,6))
    bins = np.logspace(0,6,50)
    hist, _ = np.histogram(estimated_num_events, bins=bins)
    ax.stairs(hist, bins, 
              color="forestgreen", alpha=0.25,
              )
    ax.axvline(Nevents, color='r')

    ax.set_xlabel('Estimated number of events')
    ax.set_xscale('log')
    #ax.set_xlim(xmin=1e-2,xmax=1e2)
    ax.set_ylabel('N')
    ax.set_box_aspect(1)



def plot_prior_marginals(astro_pop, prior, Nevents):
    """
    """
    xgrid = np.linspace(-10,10,101)
    ygrid = np.linspace(-10,10,101)
    zgrid = np.linspace(-10,10,101)
    X, Y, Z = np.meshgrid(xgrid, ygrid, zgrid)
    grid = np.c_[X.ravel(), Y.ravel(), Z.ravel()]
    dx = xgrid[1] - xgrid[0]
    dy = ygrid[1] - ygrid[0]
    dz = zgrid[1] - zgrid[0]

    ## Prior rate ##
    log10_dNdx_prior = np.zeros((len(prior), xgrid.shape[0]))
    log10_dNdy_prior = np.zeros((len(prior), ygrid.shape[0]))
    log10_dNdz_prior = np.zeros((len(prior), zgrid.shape[0]))
    for ind in range(len(prior)):
        this_delo = delaunaytor.CPUDelaunayInterpolator()
        this_delo.triangulate(prior[ind])
        log_rate = this_delo.interpolate(grid).reshape(ygrid.shape[0], xgrid.shape[0], zgrid.shape[0])
        log10_dNdx_prior[ind] = (special.logsumexp(log_rate, axis=(0,2)) + np.log(dy) + np.log(dz)) / np.log(10)
        log10_dNdy_prior[ind] = (special.logsumexp(log_rate, axis=(1,2)) + np.log(dx) + np.log(dz)) / np.log(10)
        log10_dNdz_prior[ind] = (special.logsumexp(log_rate, axis=(0,1)) + np.log(dx) + np.log(dy)) / np.log(10)

    ## Compute `astro' marginals
    astro_samples = astro_pop.rvs(100_000)
    x_pdf_astro, _ = np.histogram(astro_samples[:, 0], density=True, bins=xgrid)
    y_pdf_astro, _ = np.histogram(astro_samples[:, 1], density=True, bins=ygrid)
    z_pdf_astro, _ = np.histogram(astro_samples[:, 2], density=True, bins=zgrid)
    x_rate_astro = x_pdf_astro * Nevents
    y_rate_astro = y_pdf_astro * Nevents
    z_rate_astro = z_pdf_astro * Nevents


    #####################################
    # Plotting parameters
    fs = 12
    lw = 1.4
    plt.rcParams['font.size']=fs
    plt.rcParams['font.family']='serif'
    plt.rcParams['font.serif']='cmr10'
    plt.rcParams['mathtext.fontset']='cm'
    plt.rcParams['axes.unicode_minus']=False
    plt.rcParams['axes.formatter.use_mathtext']=True
    plt.rcParams['lines.linewidth']=lw
    plt.rcParams['xtick.labelsize']=fs
    plt.rcParams['ytick.labelsize']=fs
    plt.rcParams['legend.fontsize']=.9*fs


    x_labels = [r'x', r'y', r'z']
    grids = [xgrid,ygrid,zgrid]
    y_labels = ['$\\rm{log}_{10}\\rm{dN}/\\rm{d}x$', 
                '$\\rm{log}_{10}\\rm{dN}/\\rm{d}y$',
                '$\\rm{log}_{10}\\rm{dN}/\\rm{d}z$'
                ]
    prior_rates = [log10_dNdx_prior,log10_dNdy_prior,log10_dNdz_prior]
    astro_rates = [x_rate_astro,y_rate_astro,z_rate_astro]
    for n in range(len(prior_rates)):
        low_prior = np.quantile(prior_rates[n], 0.05, axis=0)
        median_prior = np.quantile(prior_rates[n], 0.5, axis=0)
        high_prior = np.quantile(prior_rates[n], 0.95, axis=0)
        fig, ax = plt.subplots(1, 1, figsize=(6,6))
        mid_bins = (grids[n][:-1]+grids[n][1:]) /2
        ax.plot(mid_bins, np.log10(astro_rates[n]), 
                color="black", ls="--",
                label=r"`Astro' pop",
                )
        ax.plot(grids[n], median_prior, color="forestgreen", alpha=0.5)
        ax.fill_between(grids[n], low_prior, high_prior,
                        color="forestgreen", alpha=0.25,
                        label='Prior',
                        )
        ax.set_xlabel(x_labels[n])
        ax.set_xlim(xmin=-10, xmax=10)
        ax.set_ylabel(y_labels[n])
        ax.legend(loc='best')
        ax.set_box_aspect(1)




##################################################################
### Run it!
###
##################################################################
if __name__ == "__main__":

    work_dir = './'
    plot_dir = os.path.join(work_dir,'plots')

    # Define command line options
    parser = argparse.ArgumentParser()
    # Set 'astro' population
    parser.add_argument("--mu1", dest='mu1', help="Mean of first distribution", default=np.array([3,5,-2]))
    parser.add_argument("--cov1", dest='cov1', help="Covariance matrix of first distribution",
                        default=np.array([[5,2,0],[2,1,0],[0,0,1.5]]),
                        )
    parser.add_argument("--mu2", dest='mu2', help="Mean of second distribution", default=np.array([-4.,2,6]))
    parser.add_argument("--cov2", dest='cov2', help="Covariance matrix of second distribution",
                        default=np.array([[1.7,0,2],[0,2.2,0],[2,0,3.5]]),
                        )
    # Events and samples
    parser.add_argument("--events", dest='Nevents', help="Number of events", type=int, default=1_000)
    args = parser.parse_args()

    # Load triangulations from prior
    filename = 'prior_triangulations.npy'
    if os.path.exists(filename):
        print('Loading prior triangulations from', filename)
        triangulations_prior = np.load(filename, allow_pickle=True)
    else:
        raise ValueError('File %s could not be found.' %filename)

    # Plot the estimated number of events
    plot_Nevents(prior=triangulations_prior, Nevents=args.Nevents)

    # Plot the marginal distributions and compare with `astro' pop
    astro_pop = generate_pop(args.mu1,args.cov1,args.mu2,args.cov2)
    plot_prior_marginals(astro_pop=astro_pop, prior=triangulations_prior, Nevents=args.Nevents)
    plt.show()
