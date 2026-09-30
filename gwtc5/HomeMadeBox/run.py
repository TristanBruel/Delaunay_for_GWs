import os
import argparse
import sys
import pickle
from tqdm import trange
from itertools import product
from multiprocessing import Pool

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from astropy.cosmology import Planck15 
from astropy import units as u
import numpy as np
from scipy import special, stats
from scipy.spatial import Delaunay, ConvexHull

from eryn.ensemble import EnsembleSampler
from eryn.moves import DistributionGenerateRJ, StretchMove, GaussianMove
from eryn.prior import uniform_dist, log_uniform, ProbDistContainer
from eryn.state import State

from local_utils.moves import BinGaussRelMove
from likelihoods import SimpleDelaunay, M1ZQDelaunay




def make_valid_delaunay(event_points, num_vertices, corners):
    delaunay = Delaunay(corners)
    inside_corners = delaunay.find_simplex(event_points)!=-1
    points = event_points[inside_corners]
    dim = 3
    offset = 2**dim
    min_max = np.array([
        [np.min(points[:, i]), np.max(points[:, i])]
        for i in range(points.shape[1])
    ])
    vertices = np.zeros((num_vertices + offset, dim))
    valid_vertices = 0
    vertices[:offset] = corners
    c = 0
    while valid_vertices < num_vertices:
        vertices[valid_vertices + offset, :] = np.random.uniform(
            low=min_max[:, 0], high=min_max[:, 1]
        )
        #this_tri = Delaunay(vertices[: offset + 1 + valid_vertices])
        #points_simplex = this_tri.find_simplex(points)
        points_simplex = delaunay.find_simplex(vertices[: offset + 1 + valid_vertices])
        #event_simplex = points_simplex[: event_points.shape[0]]
        if (points_simplex != -1).all():
            valid_vertices += 1
    return vertices[offset:]


class PriorHomeMadeBox:
    def __init__(self, corners):
        self.corners = corners
        self.m1_min = np.min(corners[:,0])
        self.m1_max = np.max(corners[:,0])
        self.z_min = np.min(corners[:,1])
        self.z_max = np.max(corners[:,1])
        self.q_min = np.min(corners[:,2])
        self.q_max = np.max(corners[:,2])
        self.delaunay = Delaunay(corners)
        self.convex = ConvexHull(corners)
        self.volume = self.convex.volume

    def rvs(self, size=1):
        if not isinstance(size, int) and not isinstance(size, tuple):
            raise ValueError("size must be an integer or tuple of ints.")
        if isinstance(size, int):
            size = (size,)
        out = np.zeros((*size,3), dtype=np.float32)
        inside = np.zeros(size, dtype=np.bool)
        while np.sum(inside) < np.prod(np.shape(inside)):
            rand = np.random.rand(*size,3)
            m1s = self.m1_min + (self.m1_max-self.m1_min)*rand[...,0]
            zs = self.z_min + (self.z_max-self.z_min)*rand[...,1]
            qs = self.q_min + (self.q_max-self.q_min)*rand[...,2]
            samples = np.stack((m1s,zs,qs), axis=-1)
            inbox = (self.delaunay.find_simplex(samples)) != -1
            out[~inside&inbox] = samples[~inside&inbox]
            inside[~inside&inbox] = inbox[~inside&inbox]
        return out

    def pdf(self, x):
        outside = self.delaunay.find_simplex(x)==-1
        return np.where(outside, 0, 1/self.volume)

    def logpdf(self, x):
        return np.log(self.pdf(x))



def set_priors(corners, w_min, w_max):
    """
    Set priors
    """
    priors = {
        "tri": {
            0: PriorHomeMadeBox(corners=corners),
            3: uniform_dist(w_min, w_max),
        },
        "corner_w": {
            i: uniform_dist(w_min, w_max) for i in range(corners.shape[0])
        },
        "chi": {
            0: uniform_dist(0., 1.),
            1: uniform_dist(0.005, 1.),
        },
        "cos_tilt": {
            0: uniform_dist(0., 1.),
            1: uniform_dist(0.01, 4),
            2: uniform_dist(-0.1, 0.1),
        }
    }
    return priors



def define_moves(m1_min, m1_max, z_min, z_max, q_min, q_max,
                    nleaves_min, nleaves_max, priors):
    """
    Define moves for the sampling
    """
    moves = [
        (BinGaussRelMove(
            sigma_vertices=0.1 * np.array([m1_max - m1_min, z_max - z_min, q_max - q_min]),
            weights_scale=1.,
            ind_leaf=ind_leaf,
            branch_name="tri",
        ),
        0.8,)
        for ind_leaf in range(nleaves_max["tri"])
    ]
    moves += [
        (StretchMove(
            gibbs_sampling_setup=["corner_w", "chi", "cos_tilt"],
            live_dangerously=True,
        ),
        0.2,)
    ]
    moves += [(StretchMove(gibbs_sampling_setup=["chi"], live_dangerously=True), 0.2)]
    moves += [(StretchMove(gibbs_sampling_setup=["cos_tilt"],live_dangerously=True), 0.2)]
    prior_move = DistributionGenerateRJ(
        {key: ProbDistContainer(priors[key]) for key in priors},
        nleaves_min={key: val for key, val in nleaves_min.items()},
        nleaves_max={key: val for key, val in nleaves_max.items()},
    )
    rj_moves = [prior_move]
    return moves, rj_moves


##################################################################
### Run it!
###
##################################################################
if __name__ == "__main__":

    # Define command line options
    parser = argparse.ArgumentParser()
    # Save dir
    parser.add_argument("--label", dest='label', help="Name of the directory to save results", type=str, default='outdir0')
    # Initial Delaunay
    parser.add_argument("--start", dest='Nstart', help="Number of vertices in initial Delaunay", type=int, default=10)
    # Prior range
    parser.add_argument("--wmin", dest='w_min', help="Lower range of the uniform distribution for the weights of vertices", type=int, default=-10)
    parser.add_argument("--wmax", dest='w_max', help="Upper range of the uniform distribution for the weights of vertices", type=int, default=10)
    # Sampling
    parser.add_argument("--procs", dest='nprocs', help="Number of CPUs", type=int, default=8)
    parser.add_argument("--walkers", dest='nwalkers', help="Number of walkers", type=int, default=40)
    parser.add_argument("--temps", dest='ntemps', help="Number of temperatures", type=int, default=3)
    parser.add_argument("--burn", dest='nburn', help="Number of iterations to burn", type=int, default=0)
    parser.add_argument("--steps", dest='nsteps', help="Number of iterations to run", type=int, default=1_000)
    args = parser.parse_args()
    nprocs, nwalkers, ntemps, nburn, nsteps = \
        args.nprocs, args.nwalkers, args.ntemps, args.nburn, args.nsteps
    start_with_this_many = args.Nstart
    os.makedirs(f"{args.label}", exist_ok=True)

    # READ data and injections
    parameter_keys = ["m1", "z", "q", "chi1", "chi2", "cos_tilt_1", "cos_tilt_2"]
    data_file = np.load("../gwtc5_samples.npz")
    observed_events = np.vstack([data_file[key + "s"] for key in parameter_keys]).T
    event_logpriors = np.log(data_file["priors"])
    num_events = int(data_file["nevents"])
    num_samples = int(data_file["nsamples"])
    print(
        f"Running with {num_events} events × {num_samples} samples = {num_events*num_samples} total samples"
    )
    barycenters = observed_events.reshape(num_events, num_samples, 7).mean(axis=1)

    injections_file = np.load("../gwtc5_injections_full.npz")
    detected_injections = np.vstack(
        [injections_file[key + "s"] for key in parameter_keys]
    ).T
    injection_priors = injections_file["inj_priors"]
    num_injections = int(injections_file["ninjs"])
    print(
        f"and with {num_injections} injections, of which {np.shape(detected_injections)[0]} are detected."
    )

    corners = np.array([
        [2., 1e-6, 0.1],
        [2., 1e-6, 1.0],
        [100, 2.0, 1.0],
        [100, 2.0, 0.7],
        [150, 2.0, 0.7],
        [150, 1e-6, 0.1],
        [150, 1e-6, 1.0],
        [150, 2.0, 1.0],
    ])

    log_like_fn = M1ZQDelaunay(
        events=observed_events,
        events_log_prior=event_logpriors,
        num_events=num_events,
        num_samples=num_samples,
        detected_injections=detected_injections,
        detected_injections_prior=injection_priors,
        num_injections=num_injections,
        corners=corners,
    )

    branch_names = ["tri", "corner_w", "chi", "cos_tilt"]
    ndims = dict(zip(branch_names, [4, 8, 2, 3]))
    nleaves_min = dict(zip(branch_names, [4, 1, 1, 1]))
    nleaves_max = dict(zip(branch_names, [100, 1, 1, 1]))

    # Set priors
    priors = set_priors(corners, w_min=args.w_min, w_max=args.w_max)

    backend_file = f"{args.label}/backend"
    if not os.path.exists(backend_file):
        print("NO BACKEND; STARTING FROM SCRATCH!")
        # Initialize
        coords = {}
        inds = {}

        for branch in branch_names:
            coords[branch] = np.zeros(
                (ntemps, nwalkers, nleaves_max[branch], ndims[branch])
            )
            inds[branch] = np.zeros((ntemps, nwalkers, nleaves_max[branch]), dtype=bool)

            if branch=="tri":
                coords[branch][..., :-1] = priors[branch][0].rvs(
                    size=(ntemps, nwalkers, nleaves_max[branch])
                )
                inds["tri"][:, :, :start_with_this_many] = True
                coords[branch][..., -1] = priors[branch][ndims['tri']-1].rvs(
                    size=(ntemps, nwalkers, nleaves_max[branch])
                )

            else:
                for i in range(ndims[branch]):
                    coords[branch][..., i] = priors[branch][i].rvs(
                        size=(ntemps, nwalkers, nleaves_max[branch])
                    )
                inds[branch][:, :, :] = True

        from joblib import Parallel, delayed
        from tqdm import tqdm
        import time
        def process_item(t, w):
            result_coords = {
                branch: np.zeros((nleaves_max[branch], ndims[branch]))
                for branch in branch_names
            }

            start_time = time.time()
            for attempt in range(10_000):
                init_proposal = {
                    "tri": np.c_[
                        make_valid_delaunay(
                            barycenters[:, :(ndims['tri']-1)],
                            start_with_this_many,
                            log_like_fn.corners,
                        ),
                        priors["tri"][ndims['tri']-1].rvs(size=start_with_this_many),
                    ]
                } | {
                    branch: np.array(
                        [
                            priors[branch][dim_indx].rvs()
                            for dim_indx in range(ndims[branch])
                        ]
                    ).squeeze()
                    for branch in priors
                    if branch != "tri"
                }
                le_log = (
                    log_like_fn(
                        [
                            init_proposal[key]
                            for key in branch_names
                        ]
                    )
                    or -1e300
                )

                if le_log > -1e300:
                    break
            else:
                raise ValueError(f"Failed for t={t}, w={w}")

            elapsed = time.time() - start_time
            # Handle both array and scalar values correctly
            for branch in init_proposal:
                if np.isscalar(init_proposal[branch]):
                    # Handle scalar values
                    result_coords[branch][0, 0] = init_proposal[branch]
                else:
                    # Handle array values
                    if branch == "tri":
                        size = len(init_proposal[branch])
                        result_coords[branch][:size] = init_proposal[branch]
                    else:
                        # For other branches, reshape if needed
                        branch_data = np.atleast_1d(init_proposal[branch])
                        result_coords[branch][: len(branch_data)] = branch_data
            print(t, w, le_log)
            return t, w, result_coords, le_log, attempt + 1, elapsed

        def run_parallel(n_jobs=-1):
            current_time = time.strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{current_time}] Starting parallel computation with {n_jobs} workers")
            print(f"User: Tristan-Bruel")
            print(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(
                f"Running {ntemps} temperatures × {nwalkers} walkers = {ntemps*nwalkers} total tasks"
            )

            # Initialize coords and inds
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Initializing arrays...")
            coords = {}
            inds = {}

            for branch in branch_names:
                coords[branch] = np.zeros(
                    (ntemps, nwalkers, nleaves_max[branch], ndims[branch])
                )
                inds[branch] = np.zeros((ntemps, nwalkers, nleaves_max[branch]), dtype=bool)

                if branch=="tri":
                    coords[branch][..., :-1] = priors[branch][0].rvs(
                        size=(ntemps, nwalkers, nleaves_max[branch])
                    )
                    inds["tri"][:, :, :start_with_this_many] = True
                    coords[branch][..., -1] = priors[branch][ndims['tri']-1].rvs(
                        size=(ntemps, nwalkers, nleaves_max[branch])
                    )

                else:
                    for i in range(ndims[branch]):
                        coords[branch][..., i] = priors[branch][i].rvs(
                            size=(ntemps, nwalkers, nleaves_max[branch])
                        )
                    inds[branch][:, :, :] = True

            # Create task list
            task_list = list(product(range(ntemps), range(nwalkers)))
            total_tasks = len(task_list)

            # Run parallel computation with progress tracking
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Starting parallel processing...")
            start_time = time.time()

            # Use try-except to catch and report errors
            try:
                results = Parallel(n_jobs=n_jobs, verbose=10, timeout=None)(
                    delayed(process_item)(t, w)
                    for t, w in tqdm(task_list, desc="Processing walkers", unit="task")
                )

                total_time = time.time() - start_time

                # Statistics for reporting
                attempts = [r[4] for r in results]
                times = [r[5] for r in results]
                le_logs = [r[3] for r in results]

                print(
                    f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Completed {total_tasks} tasks in {total_time:.2f} seconds"
                )
                print(f"Average time per task: {sum(times)/len(times):.2f} seconds")
                print(f"Average attempts per task: {sum(attempts)/len(attempts):.1f}")
                print(
                    f"Min/Max/Avg log likelihood: {min(le_logs):.2f} / {max(le_logs):.2f} / {sum(le_logs)/len(le_logs):.2f}"
                )

                # Update coords with results
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Processing results...")
                for t, w, result_coords, le_log, attempts, _ in results:
                    for branch in result_coords:
                        if branch == "tri":
                            size = start_with_this_many
                            coords[branch][t, w, :size] = result_coords[branch][:size]
                        else:
                            # Handle other branches safely
                            branch_data = result_coords[branch]
                            if np.isscalar(branch_data):
                                coords[branch][t, w, 0] = branch_data
                            else:
                                coords[branch][t, w, : len(branch_data)] = branch_data

                print(
                    f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Done! Returning state object."
                )
                return State(coords, inds=inds)

            except Exception as e:
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ERROR: {str(e)}")
                print(f"Type of error: {type(e).__name__}")
                import traceback
                traceback.print_exc()
                raise

        state_0 = backend_file + "_state_0"
        if os.path.isfile(state_0):
            print("Reusing state 0")
            with open(state_0, "rb") as f:
                state = pickle.load(f)
        else:
            state = run_parallel(n_jobs=nprocs)  # State(coords, inds=inds)
            with open(state_0, "wb") as f:
                pickle.dump(state, f)

        backend = None

    else:
        from datetime import datetime
        import shutil
        print(f"Found {backend_file=}, restarting!")

        with open(backend_file, "rb") as f:
            last_backend = pickle.load(f)

        coords = {}
        inds = {}

        for branch in branch_names:
            coords[branch] = np.zeros(
                (ntemps, nwalkers, nleaves_max[branch], ndims[branch])
            )
            inds[branch] = np.zeros((ntemps, nwalkers, nleaves_max[branch]), dtype=bool)

        for t, w in product(range(ntemps), range(nwalkers)):
            for branch in branch_names:
                coords[branch][t, w] = last_backend.chain[branch][-1, t, w]
                inds[branch][t, w] = last_backend.inds[branch][-1, t, w]

            print(t, w)
            print(
                log_like_fn(
                    [coords[branch][t, w][inds[branch][t, w]] for branch in branch_names]
                )
            )

        state = State(coords, inds=inds)
        backend = None

        rename_backend = backend_file + f'-{datetime.today().strftime("%Y%m%d_%H%M%S")}'
        print(f"Copying existing backend {backend_file} to {rename_backend}")
        shutil.copyfile(backend_file, rename_backend)

    # Define moves
    moves, rj_moves = define_moves(
            m1_min, m1_max, z_min, z_max, q_min, q_max,
            nleaves_min, nleaves_max, priors,
            )
    print('Starting the sampling...')
    with Pool(nprocs) as pool:
        ensemble = EnsembleSampler(
            nwalkers,
            ndims,
            log_like_fn,
            priors,
            tempering_kwargs=dict(ntemps=ntemps),
            nbranches=len(branch_names),
            branch_names=branch_names,
            nleaves_max=nleaves_max,
            nleaves_min=nleaves_min,
            moves=moves,
            rj_moves=rj_moves,
            pool=pool,
        )
        last_sample = ensemble.run_mcmc(
                state, nsteps, burn=nburn, progress=True, thin_by=1
        )
        print('Done!')
        with open(backend_file, "wb") as f:
            pickle.dump(ensemble.backend, f)
        print("Saved to %s" %backend_file)
