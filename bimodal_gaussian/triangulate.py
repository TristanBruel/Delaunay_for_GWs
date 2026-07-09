import numpy as np
import pickle
from itertools import product
import matplotlib.pyplot as plt
import matplotlib as mpl

from scipy import stats, special
from scipy.spatial import Delaunay
from local_utils import delaunaytor
from local_utils.moves import BinGaussRelMove
from generate_events import generate_pop, p_det

from eryn.ensemble import EnsembleSampler
from eryn.prior import uniform_dist, ProbDistContainer
from eryn.moves import DistributionGenerateRJ, StretchMove, GaussianMove
from eryn.state import State
from multiprocessing import Pool

import os
import argparse
from tqdm import trange, tqdm
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



class SquareLogLikelihood(delaunaytor.DelaunayLogLikelihood):
    def __init__(self, corners, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.corners = corners
    def __call__(self, population_parameters):
        (tri_parameters, corner_weights) = population_parameters
        corner_weights = corner_weights.squeeze()
        triangulation_parameters = np.vstack([tri_parameters, np.c_[self.corners, corner_weights.T]])
        return super().__call__(triangulation_parameters)


def make_injections(Ninjections, observed_events, pdet):
    """
    """
    print('Number of injections:', Ninjections)
    dX = 1.5 * (observed_events[:, 0].max() - observed_events[:, 0].min())
    dY = 1.5 * (observed_events[:, 1].max() - observed_events[:, 1].min())

    all_injection_events = stats.uniform(
            loc=[
                observed_events[:, 0].mean() - dX / 2.0,
                observed_events[:, 1].mean() - dY / 2.0,
                ],
                scale=[dX, dY],
            ).rvs(size=(Ninjections, 2))

    detected_injections = all_injection_events[np.random.random(Ninjections) < pdet(all_injection_events)]
    print('Number of detected injections:', len(detected_injections))
    injection_priors = np.ones(detected_injections.shape[0]) / (dX * dY)
    return detected_injections, injection_priors


def make_valid_delaunay(points, num_vertices, corners):
    """
    """
    
    if num_vertices > 3 + points.shape[0]:
        raise ValueError("That number of vertices breaks geometry")

    min_max_x = [np.min(points[:, 0]), np.max(points[:, 0])]
    min_max_y = [np.min(points[:, 1]), np.max(points[:, 1])]

    vertices = np.zeros((num_vertices + 4, 2))
    valid_vertices = 0
    vertices[:4] = corners

    c = 0
    while valid_vertices < num_vertices:
        vertices[valid_vertices + 4, 0] = np.random.uniform(*min_max_x)
        vertices[valid_vertices + 4, 1] = np.random.uniform(*min_max_y)
        
        this_tri = Delaunay(vertices[: 5 + valid_vertices])
        points_simplex = this_tri.find_simplex(points)
        event_simplex = points_simplex[: points.shape[0]]

        if (points_simplex != -1).all():
            valid_vertices += 1

        c += 1
        if c > 10_000:
            logger.debug("Arg, again!")
            vertices = np.zeros((num_vertices + 4, 2))
            vertices[:4] = corners
            valid_vertices = 0
            c = 0

    return vertices[4:]



def set_uniform_priors(corners, ndims):
    """
    """
    priors = {
            "tri": {
                0: uniform_dist(
                    corners[:, 0].min(), corners[:, 0].max()
                    ),
                1: uniform_dist(
                    corners[:, 1].min(), corners[:, 1].max()
                ),
                2: uniform_dist(-6, 9)
                },
            "corners": {
                d: uniform_dist(-6, 9) for d in range(ndims["corners"])
                }
            }
    return priors


def initial_delaunay_proposal(event_barycenters, corners, nstart,
                              priors, ndims,
                              ):
    """
    """

    test = make_valid_delaunay(event_barycenters, num_vertices=nstart, corners=corners)
    init_proposal = {"tri": np.c_[test, priors["tri"][2].rvs(nstart)]
                     } | {
                        branch: np.array(
                            [priors[branch][dim_indx].rvs() for dim_indx in range(ndims[branch])]).squeeze()
                        for branch in priors if branch != "tri"
                        }
    return init_proposal


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
    norm = mpl.colors.Normalize(vmin=-6,vmax=9)

    ax.scatter(
        vertices[:, 0], vertices[:, 1], 
        c=cmap(norm(weights)), zorder=10,
    )
    ax.triplot(
        vertices[:, 0], vertices[:, 1], d.simplices,
        color="fuchsia", ls="-",alpha=0.9,
    )

    ax.set_xlabel(r'x')
    #ax.set_xlim(xmin=-10, xmax=10)
    ax.set_ylabel(r'y')
    #ax.set_ylim(ymin=-10, ymax=10)
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


def define_moves(event_barycenters, nleaves_min, nleaves_max, priors):
    """
    Define moves for the sampling
    """
    moves = [
            BinGaussRelMove(
                sigma_vertices=0.1,
                weights_scale=event_barycenters.shape[0],
                ind_leaf=ind_leaf,
                branch_name="tri",
            )
        for ind_leaf in range(nleaves_max["tri"])
    ]

    moves += [
        StretchMove(
            gibbs_sampling_setup=["corners"],
            live_dangerously=True
        )
    ]

    prior_move = DistributionGenerateRJ(
        {key: ProbDistContainer(priors[key]) for key in priors},
        nleaves_min={key: val for key, val in nleaves_min.items()},
        nleaves_max={key: val for key, val in nleaves_max.items()},
    )
    rj_moves = [prior_move]

    return moves, rj_moves




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
    bins = np.linspace(-6,9,31)
    fig, ax = plt.subplots(1, 1, figsize=(6,6))
    for t in range(backend.ntemps):
        corner_weights = np.array([chains["corners"][step, t, walker]
                                   for step, walker in product(range(args.nsteps), range(backend.nwalkers))
                                   ]).ravel()
        hist, _ = np.histogram(corner_weights, bins=bins)
        ax.stairs(hist, bins, label='temp %i' %t)
    ax.axvline(x=-6, color='gray', linestyle='--')
    ax.axvline(x=9, color='gray', linestyle='--')
    ax.set_xlabel(r'Corner weights')
    ax.set_ylabel(r'')
    ax.legend(loc='best')
    ax.set_box_aspect(1)
    filename = '/'.join(outfile.split('/')[:-1])+'/CornerWeights'+outfile.split('/')[-1]
    plt.savefig(filename, bbox_inches='tight', dpi=1200)

    # Vertices weights
    inds = backend.get_inds()
    bins = np.linspace(-6,9,31)
    fig, ax = plt.subplots(1, 1, figsize=(6,6))
    for t in range(backend.ntemps):
        vertice_weights = np.concatenate([chains["tri"][step, t, walker][inds["tri"][step, t, walker]][:,-1]
                                   for step, walker in product(range(args.nsteps), range(backend.nwalkers))
                                   ])
        hist, _ = np.histogram(vertice_weights, bins=bins)
        ax.stairs(hist, bins, label='temp %i' %t)
    ax.axvline(x=-6, color='gray', linestyle='--')
    ax.axvline(x=9, color='gray', linestyle='--')
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
    norm = mpl.colors.Normalize(vmin=-6,vmax=9)

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
        c = ax.pcolormesh(xgrid, ygrid, data[:-1, :-1], vmin=-6, vmax=9)
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

        filename = '/'.join(outfile.split('/')[:-1])+'/lograte_q%.2f_'%q+outfile.split('/')[-1]
        plt.savefig(filename, bbox_inches='tight', dpi=1200)


def plot_marginals(triangulations, selected_tris, astro_pop, Nevents, outfile):
    """
    """
    n_triangulations = len(selected_tris)

    xgrid = np.linspace(-10,10,101)
    ygrid = np.linspace(-10,10,101)
    X, Y = np.meshgrid(xgrid, ygrid)
    grid = np.c_[X.ravel(), Y.ravel()]
    dx = xgrid[1] - xgrid[0]
    dy = ygrid[1] - ygrid[0]

    log10_dNdx = np.zeros((n_triangulations, xgrid.shape[0]))
    log10_dNdy = np.zeros((n_triangulations, ygrid.shape[0]))

    for ind, tri_ind in enumerate(tqdm(selected_tris)):
        this_delo = delaunaytor.CPUDelaunayInterpolator()
        this_delo.triangulate(triangulations[tri_ind])
        log_rate = this_delo.interpolate(grid).reshape(ygrid.shape[0], xgrid.shape[0])
        log10_dNdx[ind] = (special.logsumexp(log_rate, axis=0) + np.log(dy)) / np.log(10)
        log10_dNdy[ind] = (special.logsumexp(log_rate, axis=1) + np.log(dx)) / np.log(10)

    ## Compute `astro' marginals
    astro_samples = astro_pop.rvs(100_000)
    x_pdf_astro, _ = np.histogram(astro_samples[:, 0], bins=xgrid, density=True)
    y_pdf_astro, _ = np.histogram(astro_samples[:, 1], bins=ygrid, density=True)
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
    for n,data in enumerate([log10_dNdx,log10_dNdy]):
        low = np.quantile(data, 0.05, axis=0)
        median = np.quantile(data, 0.5, axis=0)
        high = np.quantile(data, 0.95, axis=0)

        fig, ax = plt.subplots(1, 1, figsize=(6,6))

        mid_bins = (grids[n][:-1]+grids[n][1:]) /2
        ax.plot(mid_bins, np.log10(astro_rates[n]), 
                color="black", ls="--",
                label=r"`Astro' pop",
                )

        ax.plot(grids[n], median, color="firebrick")
        ax.fill_between(grids[n], low, high, 
                        color="firebrick", alpha=0.5,
                        label='Reconstructed',
                        )

        ax.set_xlabel(x_labels[n])
        ax.set_xlim(xmin=-10,xmax=10)
        ax.set_ylabel(y_labels[n])
        ax.legend(loc='upper right')
        ax.set_box_aspect(1)

        filename = '/'.join(outfile.split('/')[:-1])+'/marginals%i_'%(n+1)+outfile.split('/')[-1]
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
    parser.add_argument("--injections", dest='Ninjections', help="Number of injections", type=int, default=100_000)
    # Initial Delaunay
    parser.add_argument("--start", dest='Nstart', help="Number of vertices in initial Delaunay", type=int, default=6)
    # Sampling
    parser.add_argument("--walkers", dest='nwalkers', help="Number of walkers", type=int, default=4)
    parser.add_argument("--temps", dest='ntemps', help="Number of temperatures", type=int, default=2)
    parser.add_argument("--burn", dest='nburn', help="Number of iterations to burn", type=int, default=500)
    parser.add_argument("--steps", dest='nsteps', help="Number of iterations to run", type=int, default=500)
    # Show the plots
    parser.add_argument("-p", dest='show_plots', action='store_true', help="Show plots")
    args = parser.parse_args()

    logging.basicConfig(
            filename=f"log",
            level=logging.DEBUG,
            )

    filename = 'events%i_samples%i.txt' %(args.Nevents,args.Nsamples)
    outfile = os.path.join(work_dir,filename)
    if os.path.exists(outfile):
        print('Loading samples from:', outfile)
        samples = np.loadtxt(outfile)
    else:
        raise ValueError("File %s could not be found." %outfile)
    event_limits = np.arange(0,len(samples),args.Nsamples)
    event_barycenters = (
        np.add.reduceat(samples, event_limits, axis=0) /args.Nsamples
    )
    # Simulate injections
    pdet = lambda events: p_det(events, sigma=args.sigma_det)
    detected_injections, injection_priors = make_injections(args.Ninjections, samples, pdet=pdet)

    corners = np.array([[-10,-10],[-10,10],[10,-10],[10,10]])
    # Some properties of the delaunay sampling scheme
    branch_names = ["tri", "corners"]
    ndims = {"tri": 3, "corners": 4}
    nleaves_min = {"tri": 4, "corners": 1}
    nleaves_max = {"tri": 40, "corners": 1}

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

    # Actually start sampling
    filename = 'backend_events%i_samples%i' %(args.Nevents,args.Nsamples)
    backend_file = os.path.join(work_dir, filename)
    if os.path.exists(backend_file):
        print('Loading sampling results from %s' %backend_file)
        with open(backend_file, "rb") as f:
            backend = pickle.load(f)
        last_sample = backend.get_last_sample()
    else:
        # Initialize
        coords = {}
        inds = {}
        for branch in branch_names:
            coords[branch] = np.zeros((args.ntemps, args.nwalkers, nleaves_max[branch], ndims[branch]))
            inds[branch] = np.zeros((args.ntemps, args.nwalkers, nleaves_max[branch]), dtype=bool)
            for i in range(ndims[branch]):
                coords[branch][..., i] = priors[branch][i].rvs(size=(args.ntemps, args.nwalkers, nleaves_max[branch]))
            if branch == "tri":
                inds["tri"][:, :, :args.Nstart] = True
            else:
                inds[branch][:, :, :] = True
            state = State(coords, inds=inds)

        for t, w in product(range(args.ntemps), range(args.nwalkers)):
            le_log = -np.inf
            for _ in trange(10_000):
                init_proposal = initial_delaunay_proposal(event_barycenters, corners, args.Nstart,
                                                          priors, ndims,
                )
                le_log = log_like_fn([init_proposal[key] for key in ["tri", "corners"]]) or -1e300
                if le_log > -1e300:
                    break
            else:
                raise ValueError("Didn't work")
            for branch in init_proposal:
                coords[branch][t, w, : (args.Nstart if branch == "tri" else nleaves_max[branch])] = init_proposal[branch]

        if args.show_plots:
            # Plot initial delaunay
            outfile = os.path.join(plot_dir, 'InitialDelaunayProposal_events%i_samples%i.png' %(args.Nevents,args.Nsamples))
            plot_delaunay(event_barycenters, init_proposal, corners,
                          title=r'Initial Delaunay Proposal', outfile=outfile,
                      )

        # Define moves for the sampling
        moves, rj_moves = define_moves(event_barycenters, nleaves_min, nleaves_max, priors)

        print('Starting the sampling...')
        nburn = args.nburn
        nsteps = args.nsteps

        with Pool(4) as pool:
            ensemble = EnsembleSampler(
                args.nwalkers,
                ndims,
                log_like_fn,
                priors,
                tempering_kwargs=dict(ntemps=args.ntemps),
                nbranches=len(branch_names),
                branch_names=branch_names,
                nleaves_max=nleaves_max,
                nleaves_min=nleaves_min,
                moves=moves,
                rj_moves=rj_moves,
                pool=pool,
            )

            last_sample = ensemble.run_mcmc(
                state, args.nsteps, burn=args.nburn, progress=True, thin_by=1
            )
            print('Done!')
            with open(backend_file, "wb") as f:
                pickle.dump(ensemble.backend, f)
            print('Saved to %s' %backend_file)
            with open(backend_file, "rb") as f:
                backend = pickle.load(f)

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
    selected_tris = np.random.choice(len(triangulations), size=args.nwalkers*args.nsteps, replace=False)
        
    if args.show_plots:
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
        
        # Plot the reconstructed pdf
        plot_maps(triangulations, selected_tris, outfile=outfile)

        # Plot the marginal distributions and compare with `astro' pop
        astro_pop = generate_pop(args.mu1,args.cov1,args.mu2,args.cov2)
        plot_marginals(triangulations, selected_tris, astro_pop=astro_pop, Nevents=args.Nevents, outfile=outfile)

        plt.show()
