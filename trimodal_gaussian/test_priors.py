import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from scipy import stats, special
from scipy.spatial import Delaunay
from local_utils import delaunaytor
from generate_events import generate_pop 
from triangulate import set_uniform_priors 

import os
import argparse
from tqdm import trange, tqdm



def make_delaunay(num_vertices, corners):
    """
    """
    min_max_x = [np.min(corners[:, 0]), np.max(corners[:, 0])]
    min_max_y = [np.min(corners[:, 1]), np.max(corners[:, 1])]
    min_max_z = [np.min(corners[:, 2]), np.max(corners[:, 2])]
    vertices = np.zeros((num_vertices + 8, 3))
    vertices[:8] = corners
    vertices[8:, 0] = np.random.uniform(*min_max_x, size=num_vertices)
    vertices[8:, 1] = np.random.uniform(*min_max_y, size=num_vertices)
    vertices[8:, 2] = np.random.uniform(*min_max_z, size=num_vertices)
    return vertices[8:]


def make_valid_delaunay(points, num_vertices, corners):
    """
    """
    num_corners = len(corners)
    if num_vertices > 3 + points.shape[0]:
        raise ValueError("That number of vertices breaks geometry")
    min_max_x = [np.min(points[:, 0]), np.max(points[:, 0])]
    min_max_y = [np.min(points[:, 1]), np.max(points[:, 1])]
    min_max_z = [np.min(points[:, 2]), np.max(points[:, 2])]
    vertices = np.zeros((num_vertices + num_corners, 3))
    valid_vertices = 0
    vertices[:num_corners] = corners
    c = 0
    while valid_vertices < num_vertices:
        vertices[valid_vertices + num_corners, 0] = np.random.uniform(*min_max_x)
        vertices[valid_vertices + num_corners, 1] = np.random.uniform(*min_max_y)
        vertices[valid_vertices + num_corners, 2] = np.random.uniform(*min_max_z)
        this_tri = Delaunay(vertices[: num_corners + 1 + valid_vertices])
        points_simplex = this_tri.find_simplex(points)
        event_simplex = points_simplex[: points.shape[0]]
        if (points_simplex != -1).all():
            valid_vertices += 1
        c += 1
        if c > 10_000:
            logger.debug("Arg, again!")
            vertices = np.zeros((num_vertices + num_corners, 3))
            vertices[:num_corners] = corners
            valid_vertices = 0
            c = 0
    return vertices[num_corners:]


def initial_delaunay_proposal(event_barycenters, corners, nstart,
                              priors, ndims,
                              ):
    """
    """

    test = make_valid_delaunay(event_barycenters, num_vertices=nstart, corners=corners)
    init_proposal = {"tri": np.c_[test, priors["tri"][3].rvs(nstart)]
                     } | {
                        branch: np.array(
                            [priors[branch][dim_indx].rvs() for dim_indx in range(ndims[branch])]).squeeze()
                        for branch in priors if branch != "tri"
                        }
    return init_proposal


def make_some_checks(astro_pop, prior, Nevents, plot_dir='./'):
    """
    """
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


    xgrid = np.linspace(-10,10,101)
    ygrid = np.linspace(-10,10,101)
    zgrid = np.linspace(-10,10,101)
    X, Y, Z = np.meshgrid(xgrid, ygrid, zgrid)
    grid = np.c_[X.ravel(), Y.ravel(), Z.ravel()]
    dx = xgrid[1] - xgrid[0]
    dy = ygrid[1] - ygrid[0]
    dz = zgrid[1] - zgrid[0]

    ## Prior rate ##
    d2N_prior = np.zeros((len(prior), ygrid.shape[0], xgrid.shape[0], zgrid.shape[0]))
    print('Computing prior rates over a grid...')
    for ind in trange(len(prior)):
        this_delo = delaunaytor.CPUDelaunayInterpolator()
        this_delo.triangulate(prior[ind])
        log_rate = this_delo.interpolate(grid).reshape(ygrid.shape[0], xgrid.shape[0], zgrid.shape[0])
        d2N_prior[ind] = np.exp(log_rate)
    estimated_num_events = np.sum(d2N_prior, axis=(1,2,3)) *dx *dy *dz
    
    ## Plot number of events from prior triangulations ##
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

    filename = os.path.join(plot_dir, 'prior_Nevents.png')
    plt.savefig(filename, bbox_inches='tight', dpi=1200)

    
    ## Plot marginal distributions ##
    log10_dNdx_prior = np.log10(np.sum(d2N_prior, axis=(1,3)) *dy*dz)
    log10_dNdy_prior = np.log10(np.sum(d2N_prior, axis=(2,3)) *dx*dz)
    log10_dNdz_prior = np.log10(np.sum(d2N_prior, axis=(1,2)) *dx*dz)

    ## Compute `astro' rate
    pdf = astro_pop.pdf(grid)
    pdf = pdf.reshape(X.shape)
    rate_astro = pdf * Nevents
    x_rate_astro = np.sum(rate_astro, axis=(0,2)) *dy*dz
    y_rate_astro = np.sum(rate_astro, axis=(1,2)) *dx*dz
    z_rate_astro = np.sum(rate_astro, axis=(0,1)) *dx*dy

    """
    low = np.quantile(d2N_prior, 0.05, axis=0)
    high = np.quantile(d2N_prior, 0.95, axis=0)
    above = low <= rate_astro
    below = rate_astro <= high
    if not above.all():
        print('WARNING: %.1f percent of the astro pdf is not above the 95 percent prior lower edge.' %(np.sum(pdf[~above])*dx*dy*dz*100))
        print('You might want to lower the range of weight distribution.')
        if not below.all():
            print('WARNING: %.1f percent of the astro pdf is not below the 95 percent prior upper edge.' %(np.sum(pdf[~below])*dx*dy*dz*100))
            print('You might want to increase the range of weight distribution.')
    else:
        print('Astro rate is well within the prior range.')
    """

    x_labels = [r'$x$', r'$y$', r'$z$']
    grids = [xgrid,ygrid,zgrid]
    y_labels = [r'$\mathrm{log}_{10}(\mathrm{dN}/\mathrm{d}x)$', 
                r'$\mathrm{log}_{10}(\mathrm{dN}/\mathrm{d}y)$',
                r'$\mathrm{log}_{10}(\mathrm{dN}/\mathrm{d}z)$',
                ]
    prior_rates = [log10_dNdx_prior,log10_dNdy_prior,log10_dNdz_prior]
    astro_rates = [x_rate_astro,y_rate_astro,z_rate_astro]
    for n in range(len(prior_rates)):
        low_prior = np.quantile(prior_rates[n], 0.05, axis=0)
        median_prior = np.quantile(prior_rates[n], 0.5, axis=0)
        high_prior = np.quantile(prior_rates[n], 0.95, axis=0)
        fig, ax = plt.subplots(1, 1, figsize=(6,6))
        ax.plot(grids[n], np.log10(astro_rates[n]), 
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

        filename = os.path.join(plot_dir, 'prior_dNd%s.png' %(['x','y','z'][n]))
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
    parser.add_argument("--samples", dest='Nsamples', help="Number of samples per event", type=int, default=10_000)
    # Prior range
    parser.add_argument("--wmin", dest='wmin', help="Lower range of the uniform distribution for the weights of vertices", type=int, default=-50)
    parser.add_argument("--wmax", dest='wmax', help="Upper range of the uniform distribution for the weights of vertices", type=int, default=10)
    # Show the plots
    parser.add_argument("-p", dest='show_plots', action='store_true', help="Show plots")
    args = parser.parse_args()


    corners = np.array([[-10,-10,-10,],[10,-10,-10],
                        [-10,10,-10],[10,10,-10],
                        [-10,-10,10],[10,-10,10],
                        [-10,10,10],[10,10,10]])
    # Some properties of the delaunay sampling scheme
    branch_names = ["tri", "corners"]
    ndims = {"tri": 4, "corners": 8}
    nleaves_min = {"tri": 8, "corners": 1}
    nleaves_max = {"tri": 100, "corners": 1}

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

    # Load triangulations from prior
    priors = set_uniform_priors(corners, ndims, weight_min=args.wmin, weight_max=args.wmax)
    #filename = 'prior_triangulations.npy'
    filename = 'prior_triangulations_valid.npy'
    if not os.path.exists(filename):
        print('Computing prior triangulations')
        triangulations_prior = np.zeros(100, dtype='object')
        for t in trange(len(triangulations_prior)):
            Nstart = np.random.randint(nleaves_min['tri'], nleaves_max['tri'])
            #test = make_delaunay(num_vertices=Nstart, corners=corners)
            test = make_valid_delaunay(points=event_barycenters, num_vertices=Nstart, corners=corners)
            init_proposal = {"tri": np.c_[test, priors["tri"][3].rvs(Nstart)]
                     } | {
                        branch: np.array(
                            [priors[branch][dim_indx].rvs() for dim_indx in range(ndims[branch])]).squeeze()
                        for branch in priors if branch != "tri"
                        }
            triangulations_prior[t] = np.vstack([init_proposal['tri'],np.c_[corners,init_proposal['corners'].T]])
        np.save(filename, triangulations_prior)
    else:
        print('Loading prior triangulations from', filename)
        triangulations_prior = np.load(filename, allow_pickle=True)

    astro_pop = generate_pop(args.mu1,args.cov1,args.mu2,args.cov2)
    make_some_checks(astro_pop=astro_pop, prior=triangulations_prior, Nevents=args.Nevents, plot_dir=plot_dir)
    if args.show_plots:
        plt.show()
