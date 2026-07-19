import numpy as np
import pickle
from itertools import product

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
                2: uniform_dist(-12, 8)
                },
            "corners": {
                d: uniform_dist(-12, 8) for d in range(ndims["corners"])
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



def check_prior_range(astro_pop, Nevents, prior):
    """
    """
    xgrid = np.linspace(-10,10,101)
    ygrid = np.linspace(-10,10,101)
    X, Y = np.meshgrid(xgrid, ygrid)
    grid = np.c_[X.ravel(), Y.ravel()]
    dx = xgrid[1] - xgrid[0]
    dy = ygrid[1] - ygrid[0]

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

    astro_rates = [x_rate_astro,y_rate_astro]
    prior_rates = [log10_dNdx_prior,log10_dNdy_prior]
    prior_is_ok = True
    for n,data in enumerate(prior_rates):
        low = np.quantile(data, 0.05, axis=0)
        high = np.quantile(data, 0.95, axis=0)
        if not (low <= np.log10((astro_rates[n]))).all():
            print('WARNING: Astro rate over dimension %i is not always above the prior range'%n)
            print('You might want to lower the range of weight distribution')
            prior_is_ok = False
        elif not (np.log10((astro_rates[n])) <= high).all():
            print('WARNING: Astro rate over dimension %i is not always below the prior range'%n)
            print('You might want to increase the range of weight distribution')
            prior_is_ok = False
        else:
            print('\tAstro rate is well within the prior range in dimension %i' %n)
    return prior_is_ok





##################################################################
### Run it!
###
##################################################################
if __name__ == "__main__":

    work_dir = './'

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
    parser.add_argument("--burn", dest='nburn', help="Number of iterations to burn", type=int, default=2000)
    parser.add_argument("--steps", dest='nsteps', help="Number of iterations to run", type=int, default=1000)
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
    filename = 'detected_injections.txt'
    if os.path.exists(filename):
        print('Loading injections from', filename)
        detected_injections = np.loadtxt(filename, usecols=[0,1])
        injection_priors = np.loadtxt(filename, usecols=2)
    else:
        print('Simulating injections')
        pdet = lambda events: p_det(events, sigma=args.sigma_det)
        detected_injections, injection_priors = make_injections(args.Ninjections, samples, pdet=pdet)
        np.savetxt(filename, np.vstack([detected_injections.T, injection_priors]).T, header='last column is injection prior')

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
    astro_pop = generate_pop(args.mu1,args.cov1,args.mu2,args.cov2)
    if not os.path.exists(filename):
        print('Computing prior triangulations')
        triangulations_prior = np.zeros((100, args.Nstart+corners.shape[0],ndims['tri']))
        for t in trange(len(triangulations_prior)):
            le_log = -np.inf
            for _ in range(10_000):
                init_proposal = initial_delaunay_proposal(event_barycenters, corners, args.Nstart,
                                                          priors, ndims,
                )
                le_log = log_like_fn([init_proposal[key] for key in ["tri", "corners"]]) or -1e300
                if le_log > -1e300:
                    triangulations_prior[t] = np.vstack([init_proposal['tri'],np.c_[corners,init_proposal['corners'].T]])
                    break
        tosave = triangulations_prior.reshape(triangulations_prior.shape[0],triangulations_prior.shape[1]*triangulations_prior.shape[2])
        np.savetxt(filename, tosave)
    else:
        print('Loading prior triangulations from', filename)
        triangulations_prior = np.loadtxt(filename)
        triangulations_prior = triangulations_prior.reshape(triangulations_prior.shape[0], args.Nstart+corners.shape[0], ndims['tri'])
    prior_is_ok = check_prior_range(astro_pop, args.Nevents, triangulations_prior)

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
