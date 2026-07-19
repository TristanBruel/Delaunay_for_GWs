import numpy as np
import pickle
from itertools import product
import matplotlib.pyplot as plt
import matplotlib as mpl

from scipy import stats, special
from scipy.spatial import Delaunay
from generate_events import generate_pop, p_det
from triangulate import set_uniform_priors, SquareLogLikelihood, make_injections, initial_delaunay_proposal

import os
import argparse
from tqdm import trange, tqdm
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)




def plot_delaunay(event_barycenters, delaunay_proposal, corners,
                  title, outfile,
                  ):
    """
    Plot events and a proposal of Delaunay triangulation scheme.
    """
    vertices = np.vstack([delaunay_proposal['tri'][:,:2], corners])
    weights = np.concatenate([delaunay_proposal['tri'][:,2], delaunay_proposal['corners']])
    d = Delaunay(vertices)

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

    ax.scatter(event_barycenters[:,0], event_barycenters[:,1], 
               c='k', marker='*', s=10**2,
               )

    cmap = plt.get_cmap('magma')
    norm = mpl.colors.Normalize(vmin=-12,vmax=8)

    ax.scatter(
        vertices[:, 0], vertices[:, 1], 
        c=cmap(norm(weights)), zorder=10,
    )
    ax.triplot(
        vertices[:, 0], vertices[:, 1], d.simplices,
        color="fuchsia", ls="-",alpha=0.9,
    )

    ax.set_xlabel(r'x')
    ax.set_xlim(xmin=-10, xmax=10)
    ax.set_ylabel(r'y')
    ax.set_ylim(ymin=-10, ymax=10)
    ax.set_title(title)
    ax.set_box_aspect(1)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, ax=ax, 
                        cmap=cmap, norm=norm, 
                        fraction=0.086, pad=0.04, aspect=10,
                        )
    cbar.set_label(r'weight')
    cbar.ax.tick_params(labelsize=0.8*fs)

    plt.savefig(outfile, bbox_inches='tight', dpi=1200)
    return fig



def plot_diagnostics(backend, outfile):
    """
    """
    chain = backend.get_chain()

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

    ## Acceptance fractions ##
    for t in range(backend.ntemps):
        fig, ax = plt.subplots(1, 1, figsize=(6,6))
        xticks = np.arange(backend.nwalkers)+1
        for move in backend.get_move_info():
            acceptance_frac = backend.get_move_info()[move]['acceptance_fraction'][t]
            if 'RJ' in move:
                ax.scatter(xticks, 100*acceptance_frac,
                           s=7**2, edgecolor='k', facecolor='white',
                           )
            else:
                ax.scatter(xticks, 100*acceptance_frac, 
                           s=7**2,
                           )
        ax.scatter([], [], s=7**2, edgecolor='k', facecolor='white', label=r'rj move')
        ax.scatter([], [], s=7**2, edgecolor='k', facecolor='k', label=r'moves')
        ax.set_xlabel(r'walkers')
        ax.set_xticks(xticks)
        ax.set_ylabel(r'Acceptance fraction')
        ax.legend(loc='best')
        ax.set_box_aspect(1)
        filename = '/'.join(outfile.split('/')[:-1])+'/AcceptanceFraction_temp%i'%t+outfile.split('/')[-1]
        plt.savefig(filename, bbox_inches='tight', dpi=1200)

    ## Log-likelihood convergence ##
    fig, ax = plt.subplots(1, 1, figsize=(6,6))
    for w in range(backend.nwalkers):
        ax.plot(backend.get_log_like()[:, 0, w], label=w)
    ax.set_xlabel(r'Iterations')
    ax.set_ylabel(r'Log-likelihood')
    ax.legend(loc='best')
    ax.set_box_aspect(1)
    filename = '/'.join(outfile.split('/')[:-1])+'/LogLikeConvergence'+outfile.split('/')[-1]
    plt.savefig(filename, bbox_inches='tight', dpi=1200)


    ## Posterior checks ##
    # Number of vertices
    backend_nleaves = backend.get_nleaves()
    bins = np.linspace(0, backend.nleaves_max['tri'])
    fig, ax = plt.subplots(1, 1, figsize=(6,6))
    for t in range(backend.ntemps):
        nvertices = backend_nleaves['tri'][:,t,:].ravel()
        hist, _ = np.histogram(nvertices, bins=bins)
        ax.stairs(hist, bins, label='temp %i' %t)
    ax.axvline(x=4, color='gray', linestyle='--')
    ax.axvline(x=backend.nleaves_max['tri'], color='gray', linestyle='--')
    ax.set_xlabel(r'Number of vertices')
    ax.set_ylabel(r'')
    ax.legend(loc='best')
    ax.set_box_aspect(1)
    filename = '/'.join(outfile.split('/')[:-1])+'/Nvertices'+outfile.split('/')[-1]
    plt.savefig(filename, bbox_inches='tight', dpi=1200)

    # Corner weights
    chains = backend.get_chain()
    bins = np.linspace(-12,8,21)
    fig, ax = plt.subplots(1, 1, figsize=(6,6))
    for t in range(backend.ntemps):
        corner_weights = np.array([chains["corners"][step, t, walker]
                                   for step, walker in product(range(args.nsteps), range(backend.nwalkers))
                                   ]).ravel()
        hist, _ = np.histogram(corner_weights, bins=bins)
        ax.stairs(hist, bins, label='temp %i' %t)
    ax.axvline(x=-12, color='gray', linestyle='--')
    ax.axvline(x=8, color='gray', linestyle='--')
    ax.set_xlabel(r'Corner weights')
    ax.set_ylabel(r'')
    ax.legend(loc='best')
    ax.set_box_aspect(1)
    filename = '/'.join(outfile.split('/')[:-1])+'/CornerWeights'+outfile.split('/')[-1]
    plt.savefig(filename, bbox_inches='tight', dpi=1200)

    # Vertices weights
    inds = backend.get_inds()
    bins = np.linspace(-12,8,21)
    fig, ax = plt.subplots(1, 1, figsize=(6,6))
    for t in range(backend.ntemps):
        vertice_weights = np.concatenate([chains["tri"][step, t, walker][inds["tri"][step, t, walker]][:,-1]
                                   for step, walker in product(range(args.nsteps), range(backend.nwalkers))
                                   ])
        hist, _ = np.histogram(vertice_weights, bins=bins)
        ax.stairs(hist, bins, label='temp %i' %t)
    ax.axvline(x=-12, color='gray', linestyle='--')
    ax.axvline(x=8, color='gray', linestyle='--')
    ax.set_xlabel(r'Vertices weights')
    ax.set_ylabel(r'')
    ax.legend(loc='best')
    ax.set_box_aspect(1)
    filename = '/'.join(outfile.split('/')[:-1])+'/VerticesWeights'+outfile.split('/')[-1]
    plt.savefig(filename, bbox_inches='tight', dpi=1200)



def plot_maps(triangulations, selected_tris, outfile):
    """
    """
    n_triangulations = len(selected_tris)

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

    cmap = plt.get_cmap('magma')
    norm = mpl.colors.Normalize(vmin=-12,vmax=8)

    xgrid = np.linspace(-10,10,101)
    ygrid = np.linspace(-10,10,101)
    X, Y = np.meshgrid(xgrid, ygrid)
    grid = np.c_[X.ravel(), Y.ravel()]
    dx = xgrid[1] - xgrid[0]
    dy = ygrid[1] - ygrid[0]

    log_rate = np.zeros((n_triangulations, grid.shape[0]))
    for ind, tri_ind in enumerate(tqdm(selected_tris)):
        this_delo = delaunaytor.CPUDelaunayInterpolator()
        this_delo.triangulate(triangulations[tri_ind])
        log_rate[ind] = this_delo.interpolate(grid)
    square_rate = log_rate.reshape(n_triangulations, ygrid.shape[0], xgrid.shape[0])

    quantiles = [0.05, 0.5, 0.95]
    for q in quantiles:
        data = np.quantile(square_rate, q, axis=0)
        fig, ax = plt.subplots(1, 1, figsize=(6,6))
        c = ax.pcolormesh(xgrid, ygrid, data[:-1, :-1], vmin=-12, vmax=8)
        cbar = fig.colorbar(c, ax=ax,
                            cmap=cmap, norm=norm,
                            fraction=0.086, pad=0.04, aspect=10,
                            )
        cbar.set_label(r'Log pdf')
        cbar.ax.tick_params(labelsize=0.8*fs)

        ax.set_xlabel(r'x')
        ax.set_xlim(xmin=-10,xmax=10)
        ax.set_ylabel(r'y')
        ax.set_ylim(ymin=-10,ymax=10)
        ax.set_title(r'Reconstructed log-rate (%.2f quantile)' %q)
        ax.set_box_aspect(1)

        filename = '/'.join(outfile.split('/')[:-1])+'/lograte_q%.2f'%q+outfile.split('/')[-1]
        plt.savefig(filename, bbox_inches='tight', dpi=1200)



def compute_num_events(triangulation_points):
        tri = delaunaytor.CPUDelaunayInterpolator()
        tri.triangulate(triangulation_points)
        weights_in_vertices = tri.weights[tri.triangulation.simplices]
        weight_diffs = np.c_[
            (weights_in_vertices[:, 1] - weights_in_vertices[:, 2]),
            (weights_in_vertices[:, 2] - weights_in_vertices[:, 0]),
            (weights_in_vertices[:, 0] - weights_in_vertices[:, 1]),
        ]
        bar_integral = (np.exp(weights_in_vertices) * weight_diffs).sum(axis=-1) / (
            -weight_diffs
        ).prod(axis=-1)
        return (2 * tri.volumes() * bar_integral).sum()



def plot_Nevents(triangulations, Nevents, outfile):
    """
    """
    estimated_num_events = np.array([compute_num_events(tri) for tri in triangulations])

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

    bins = np.logspace(-2,2,41)
    hist, _ = np.histogram(estimated_num_events /Nevents, bins=bins)
    ax.stairs(hist, bins)

    ax.axvline(Nevents, color='r')

    ax.set_xlabel('Estimated number of events /Real number of events')
    ax.set_xscale('log')
    ax.set_xlim(xmin=1e-2, xmax=1e2)
    ax.set_ylabel('N')
    ax.set_box_aspect(1)

    filename = '/'.join(outfile.split('/')[:-1])+'/Nevents'+outfile.split('/')[-1]
    plt.savefig(filename, bbox_inches='tight', dpi=1200)



def plot_marginals(triangulations, selected_tris, astro_pop, prior, Nevents, outfile):
    """
    """
    n_triangulations = len(selected_tris)

    xgrid = np.linspace(-10,10,101)
    ygrid = np.linspace(-10,10,101)
    X, Y = np.meshgrid(xgrid, ygrid)
    grid = np.c_[X.ravel(), Y.ravel()]
    dx = xgrid[1] - xgrid[0]
    dy = ygrid[1] - ygrid[0]

    ## Inferred event rate ##
    log10_dNdx = np.zeros((n_triangulations, xgrid.shape[0]))
    log10_dNdy = np.zeros((n_triangulations, ygrid.shape[0]))
    estimated_num_events = np.array([compute_num_events(tri) for tri in triangulations])
    for ind, tri_ind in enumerate(tqdm(selected_tris)):
        this_delo = delaunaytor.CPUDelaunayInterpolator()
        this_delo.triangulate(triangulations[tri_ind])
        log_rate = this_delo.interpolate(grid).reshape(ygrid.shape[0], xgrid.shape[0])
        log10_dNdx[ind] = (special.logsumexp(log_rate, axis=0) + np.log(dy)) / np.log(10)
        log10_dNdy[ind] = (special.logsumexp(log_rate, axis=1) + np.log(dx)) / np.log(10)

    ## Prior rate ##
    log10_dNdx_prior = np.zeros((len(prior), xgrid.shape[0]))
    log10_dNdy_prior = np.zeros((len(prior), ygrid.shape[0]))
    for ind in range(len(prior)):
        this_delo = delaunaytor.CPUDelaunayInterpolator()
        this_delo.triangulate(prior[ind])
        log_rate = this_delo.interpolate(grid).reshape(ygrid.shape[0], xgrid.shape[0])
        log10_dNdx_prior[ind] = (special.logsumexp(log_rate, axis=0) + np.log(dy)) / np.log(10)
        log10_dNdy_prior[ind] = (special.logsumexp(log_rate, axis=1) + np.log(dx)) / np.log(10)

    ## Compute `astro' marginals
    pdf = astro_pop.pdf(np.array([X,Y]).T)
    x_pdf_astro = np.trapezoid(x=ygrid, y=pdf, axis=1)
    y_pdf_astro = np.trapezoid(x=xgrid, y=pdf, axis=0)
    x_rate_astro = x_pdf_astro * Nevents
    y_rate_astro = y_pdf_astro * Nevents


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


    x_labels = [r'x', r'y']
    grids = [xgrid,ygrid]
    y_labels = [r'$\mathrm{log}_{10}\mathrm{dN}/\mathrm{d}x$', 
                r'$\mathrm{log}_{10}\mathrm{dN}/\mathrm{d}y$'
                ]
    astro_rates = [x_rate_astro,y_rate_astro]
    prior_rates = [log10_dNdx_prior,log10_dNdy_prior]
    for n,data in enumerate([log10_dNdx,log10_dNdy]):
        low = np.quantile(data, 0.05, axis=0)
        median = np.quantile(data, 0.5, axis=0)
        high = np.quantile(data, 0.95, axis=0)

        low_prior = np.quantile(prior_rates[n], 0.05, axis=0)
        median_prior = np.quantile(prior_rates[n], 0.5, axis=0)
        high_prior = np.quantile(prior_rates[n], 0.95, axis=0)

        fig, ax = plt.subplots(1, 1, figsize=(6,6))

        ax.plot(grids[n], np.log10(astro_rates[n]), 
                color="black", ls="--",
                label=r"`Astro' pop",
                )

        ax.plot(grids[n], median, color="firebrick")
        ax.fill_between(grids[n], low, high, 
                        color="firebrick", alpha=0.5,
                        label='Reconstructed',
                        )

        ax.plot(grids[n], median_prior, color="forestgreen", alpha=0.5)
        ax.fill_between(grids[n], low_prior, high_prior,
                        color="forestgreen", alpha=0.25,
                        label='Prior',
                        )

        ax.set_xlabel(x_labels[n])
        ax.set_xlim(xmin=-10,xmax=10)
        ax.set_ylabel(y_labels[n])
        ax.legend(loc='upper right')
        ax.set_box_aspect(1)

        filename = '/'.join(outfile.split('/')[:-1])+'/marginals%i'%(n+1)+outfile.split('/')[-1]
        plt.savefig(filename, bbox_inches='tight', dpi=1200)




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
    parser.add_argument("--mu1", dest='mu1', help="Mean of first distribution", default=np.array([-5,0]))
    parser.add_argument("--cov1", dest='cov1', help="Covariance matrix of first distribution",
                        default=np.array([[3,1],[1,3]]),
                        )
    parser.add_argument("--mu2", dest='mu2', help="Mean of second distribution", default=np.array([5,0]))
    parser.add_argument("--cov2", dest='cov2', help="Covariance matrix of second distribution",
                        default=np.array([[2,-1],[-1,1]]),
                        )
    # Set detection probability
    parser.add_argument("--sigma", dest='sigma_det', help="Exponential parameter of the detection probability", type=float, default=3)
    # Events and samples
    parser.add_argument("--events", dest='Nevents', help="Number of events", type=int, default=1_000)
    parser.add_argument("--samples", dest='Nsamples', help="Number of samples per event", type=int, default=10_000)
    # Injections
    parser.add_argument("--injections", dest='Ninjections', help="Number of injections", type=int, default=1_000_000)
    # Initial Delaunay
    parser.add_argument("--start", dest='Nstart', help="Number of vertices in initial Delaunay", type=int, default=6)
    # Sampling
    parser.add_argument("--walkers", dest='nwalkers', help="Number of walkers", type=int, default=4)
    parser.add_argument("--temps", dest='ntemps', help="Number of temperatures", type=int, default=2)
    parser.add_argument("--steps", dest='nsteps', help="Number of iterations to run", type=int, default=1000)
    # Show the plots
    parser.add_argument("-p", dest='show_plots', action='store_true', help="Show plots")
    args = parser.parse_args()

    logging.basicConfig(
            filename=f"log",
            level=logging.DEBUG,
            )

    filename = 'samples.txt'
    outfile = os.path.join(work_dir,filename)
    if os.path.exists(outfile):
        print('Loading samples from:', outfile)
        samples = np.loadtxt(outfile)
        Nevents_det = int(len(samples)/args.Nsamples)
        print('Number of detected events:', Nevents_det)
    else:
        raise ValueError("File %s could not be found." %outfile)
    event_limits = np.arange(0,len(samples),args.Nsamples)
    event_barycenters = (
        np.add.reduceat(samples, event_limits, axis=0) /args.Nsamples
    )

    corners = np.array([[-10,-10],[-10,10],[10,-10],[10,10]])
    # Some properties of the delaunay sampling scheme
    branch_names = ["tri", "corners"]
    ndims = {"tri": 3, "corners": 4}
    nleaves_min = {"tri": 4, "corners": 1}
    nleaves_max = {"tri": 40, "corners": 1}

    # Simulate injections
    pdet = lambda events: p_det(events, sigma=args.sigma_det)
    detected_injections, injection_priors = make_injections(args.Ninjections, samples, pdet=pdet)

    # Set likelihood
    priors = set_uniform_priors(corners, ndims)
    event_logpriors = np.ones(samples.shape[0])
    log_like_fn = SquareLogLikelihood(
            corners=corners,
            events=samples,
            events_log_prior=event_logpriors,
            detected_injections=detected_injections,
            detected_injections_prior=injection_priors,
            num_events=len(samples)//args.Nsamples,
            num_samples=args.Nsamples,
            num_injections=args.Ninjections,
            minus_infinity=-1e300,
            )

    # Check that `astro' event rate is within priors
    filename = 'prior_triangulations.txt'
    if os.path.exists(filename):
        print('Loading prior triangulations from', filename)
        triangulations_prior = np.loadtxt(filename)
        triangulations_prior = triangulations_prior.reshape(triangulations_prior.shape[0], args.Nstart+corners.shape[0], ndims['tri'])
    else:
        raise ValueError('File %s could not be found.' %filename)

    print('Generating random initial delaunay configuration')
    le_log = -np.inf
    for _ in range(10_000):
        init_proposal = initial_delaunay_proposal(event_barycenters, corners, args.Nstart,
                                                  priors, ndims,
        )
        le_log = log_like_fn([init_proposal[key] for key in ["tri", "corners"]]) or -1e300
        if le_log > -1e300:
            break
    else:
        raise ValueError("Didn't work")

    # Plot initial delaunay
    outfile = os.path.join(plot_dir, 'InitialDelaunayProposal_events%i_samples%i.png' %(args.Nevents,args.Nsamples))
    plot_delaunay(event_barycenters, init_proposal, corners,
                  title=r'Initial Delaunay Proposal', outfile=outfile,
                  )

    # Get results from sampling
    filename = 'backend_events%i_samples%i' %(args.Nevents,args.Nsamples)
    backend_file = os.path.join(work_dir, filename)
    if os.path.exists(backend_file):
        print('Loading sampling results from %s' %backend_file)
        with open(backend_file, "rb") as f:
            backend = pickle.load(f)
        last_sample = backend.get_last_sample()
    else:
        raise ValueError('File %s could not be found' %backend_file)

    chain = backend.get_chain()
    inds = backend.get_inds()
    triangulations =  [
            chain["tri"][step, 0, walker][inds["tri"][step, 0, walker]]
            for step, walker in product(range(args.nsteps), range(backend.nwalkers))
            ]
    corner_weights = np.array([
        chain["corners"][step, 0, walker]
        for step, walker in product(range(args.nsteps), range(backend.nwalkers))
        ])
    triangulations = [
            np.vstack([triangulations[ind],np.c_[corners, corner_weights[ind].T]])
            for ind in range(len(triangulations))
            ]
    #selected_tris = np.random.choice(len(triangulations), size=args.nwalkers*args.nsteps, replace=False)
    selected_tris = np.random.choice(len(triangulations), size=500, replace=False)
        
    # Plot final delaunay of one walker
    ind = 0
    final_proposal = {"tri":last_sample.branches["tri"].coords[0, ind, last_sample.branches["tri"].inds[0, ind, :]]
                      } | {
                        "corners": last_sample.branches["corners"].coords[0, ind, 0]
                        }

    outfile = os.path.join(plot_dir, 'FinalDelaunayProposal_walker%i_events%i_samples%i.png' %(ind,args.Nevents,args.Nsamples))
    plot_delaunay(event_barycenters, final_proposal, corners, 
                  title=r'Final Delaunay Proposal - walker %i' %ind, outfile=outfile,
                  )

    # Plot some diagnostics of the sampling
    outfile = os.path.join(plot_dir, '_events%i_samples%i.png' %(args.Nevents,args.Nsamples))
    plot_diagnostics(backend, outfile)

    # Plot the estimated number of events
    plot_Nevents(triangulations, Nevents=args.Nevents, outfile=outfile)
    
    # Plot the reconstructed pdf
    plot_maps(triangulations, selected_tris, outfile=outfile)

    # Plot the marginal distributions 
    plot_marginals(triangulations, selected_tris, astro_pop=astro_pop, prior=triangulations_prior, Nevents=args.Nevents, outfile=outfile)

    if args.show_plots:
        plt.show()
