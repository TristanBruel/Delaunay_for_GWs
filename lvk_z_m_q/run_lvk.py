from itertools import product
import logging
from multiprocessing import Pool

import os
import sys
import pickle

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from astropy.cosmology import Planck15 
from astropy import units as u
import numpy as np
import matplotlib.pyplot as plt
from scipy import special, stats
from scipy.spatial import Delaunay

from eryn.ensemble import EnsembleSampler
from eryn.moves import DistributionGenerateRJ, StretchMove, GaussianMove
from eryn.prior import uniform_dist, log_uniform, ProbDistContainer
from eryn.state import State

import corner
from tqdm import trange

from local_utils import delaunaytor
from local_utils.moves import BinGaussRelMove

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

no_plot = True
nproc = 8
nwalkers = 40
ntemps = 3

label = sys.argv[1]
nburn = int(sys.argv[2])
nsteps = int(sys.argv[3])

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

os.makedirs(f"{label}", exist_ok=True)
savefig = lambda fig, name: fig.savefig(f"{label}/{name}.pdf")

z_min = 0.
z_max = 2.3
m1_min = 2.0
m1_max = 100.0 + 1e-6
q_min = 0.
q_max = 1.

def chi_log_pdf(chi, mu_chi, var_chi):
    nu = mu_chi * (1 - mu_chi) / var_chi - 1.0

    alpha_chi = mu_chi * nu
    beta_chi = (1 - mu_chi) * nu

    zero_mask = np.logical_or(nu < 0, np.logical_or(alpha_chi < 0, beta_chi < 0))

    return np.where(zero_mask, -1e300, stats.beta(a=alpha_chi, b=beta_chi).logpdf(chi))


def tilts_log_pdf(tilt_1, tilt_2, zeta, sigma_t):
    truncated_gaussian = stats.truncnorm(
        loc=0, scale=sigma_t, a=-1 / sigma_t, b=1 / sigma_t
    )

    gauss_prod = truncated_gaussian.pdf(tilt_1) * truncated_gaussian.pdf(tilt_2)
    return np.where(
        gauss_prod <= 0.0, -1e300, np.log(0.25 * (1 - zeta) + zeta * gauss_prod)
    )


# READ data and injections
parameter_keys = ["m1", "z", "q", "chi1", "chi2", "tilt1", "tilt2"]
total_dimensions = len(parameter_keys)

data_file = np.load("../lvk_z_m/lvc_data_lvc_samples_full.npz")
observed_events = np.vstack([data_file[key + "s"] for key in parameter_keys]).T
event_logpriors = np.log(data_file["priors"])
num_events = data_file["nevents"][0]
num_samples = data_file["nsamples"][0]

injections_file = np.load("../lvk_z_m/selection_function_elements.npz")
detected_injections = np.vstack(
    [injections_file[key + "s"] for key in parameter_keys]
).T
injection_priors = injections_file["inj_priors"]
num_injections = injections_file["ninjs"]


class SimpleDelaunay(delaunaytor.DelaunayLogLikelihood):

    def rate(self, triangulation_params):
        self.delaunay_interpolator.triangulate(triangulation_params)

        events_simplex, events_b = self.delaunay_interpolator.simplex_and_barycenters(
            self.events
        )

        samples_inside = (events_simplex != -1).reshape(
            self.num_events, self.num_samples
        )

        log_dNdtheta_samples = self.delaunay_interpolator._interpolate(
            events_simplex, events_b
        )

        inj_simplex, inj_b = self.delaunay_interpolator.simplex_and_barycenters(
            self.detected_injections
        )
        log_Nxi = self.delaunay_interpolator._interpolate(inj_simplex, inj_b)

        maybe_infinity = np.exp(log_Nxi)
        if np.isinf(maybe_infinity).any():
            return self.minus_infinity

        inj_inside = inj_simplex != -1

        return (
            log_dNdtheta_samples,
            samples_inside,
            maybe_infinity,
            inj_inside,
        )

class M1ZQDelaunay:

    def __init__(
        self,
        events,
        events_log_prior,
        num_events,
        num_samples,
        detected_injections,
        detected_injections_prior,
        num_injections,
        corners,
        minus_infinity=-1e300,
    ):
        delaunay_indices = (0, 1, 2)
        if len(delaunay_indices) != 3:
            raise ValueError("Three is the number of dimensions I shall triangulate over")

        self.delaunay_rate = SimpleDelaunay(
            events=events[:, delaunay_indices],
            events_log_prior=event_logpriors,
            num_events=num_events,
            num_samples=num_samples,
            detected_injections=detected_injections[:, delaunay_indices],
            detected_injections_prior=detected_injections_prior,
            num_injections=num_injections,
            minus_infinity=None,
        )

        self.corners = corners

        self.events = events
        self.num_events = num_events
        self.num_samples = num_samples
        self.log_num_samples = np.log(num_samples)

        self.events_log_prior = events_log_prior.reshape(
            self.num_events, self.num_samples
        )

        self.detected_injections = detected_injections
        self.detected_injections_prior = detected_injections_prior
        self.num_injections = num_injections

        self.minus_infinity = minus_infinity

    def __call__(self, population_parameters):
        (inner_tri_parameters, corner_weights, mu_var_chi, zeta_sigma_t) = population_parameters

        mu_var_chi = mu_var_chi.squeeze()
        zeta_sigma_t = zeta_sigma_t.squeeze()

        triangulation_parameters = np.vstack([
            inner_tri_parameters,
            np.c_[self.corners, corner_weights.T]
        ])

        if (tri_result := self.delaunay_rate.rate(triangulation_parameters)) is None:
            return self.minus_infinity
        (
            log_dNdtheta_tri,
            samples_inside_tri,
            Nxi_tri,
            inj_inside_tri,
        ) = tri_result

        # parameter_keys = ["m1", "z", "q", "chi1", "chi2", "tilt1", "tilt2"]
        log_dNdtheta = (
            log_dNdtheta_tri
            + chi_log_pdf(self.events[:, 3], mu_var_chi[0], mu_var_chi[1])
            + chi_log_pdf(self.events[:, 4], mu_var_chi[0], mu_var_chi[1])
            + tilts_log_pdf(
                self.events[:, 5], self.events[:, 6], zeta_sigma_t[0], zeta_sigma_t[1]
                )
        ).reshape(self.num_events, self.num_samples)

        to_integrate = log_dNdtheta - self.events_log_prior
        log_bayes_factors = self.delaunay_rate.logsumexp(
            to_integrate,  b=samples_inside_tri, axis=-1
        )

        log_variance_likes = self.delaunay_rate.logsumexp(
            2 * to_integrate, b=samples_inside_tri, axis=-1
        )
        if np.isnan(log_variance_likes).any():
            logger.debug("Variances are bad")
            return self.minus_infinity

        log_effective_sample_sizes = 2 * log_bayes_factors - log_variance_likes
        if ((log_effective_sample_sizes < np.log(self.num_events))).any():
            logger.debug(f"Effective sample size is too low")
            return self.minus_infinity


        Nxi_presum = Nxi_tri * np.exp(
            (chi_log_pdf(self.detected_injections[:, 3], mu_var_chi[0], mu_var_chi[1]))
            + chi_log_pdf(self.detected_injections[:, 4], mu_var_chi[0], mu_var_chi[1])
            + tilts_log_pdf(
                self.detected_injections[:, 5],
                self.detected_injections[:, 6],
                zeta_sigma_t[0],
                zeta_sigma_t[1],
            )
        )

        Nxi = (
            integrand := (Nxi_presum * inj_inside_tri / self.detected_injections_prior)
        ).sum() / self.num_injections


        var = (
            (integrand / self.num_injections) ** 2
        ).sum() - Nxi**2 / self.num_injections

        n_eff = Nxi**2 / var
        if n_eff <= 4 * self.num_events:
            logger.debug("Not enough injection stuff")
            return self.minus_infinity

        #actual_log_num_samples = np.log(samples_inside_tri.sum(axis=1))
        actual_log_num_samples = self.log_num_samples

        result = (log_bayes_factors - actual_log_num_samples).sum() - Nxi
        return result.item()

def make_valid_delaunay(event_points, num_vertices, corners):
    
    points = event_points
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
        
        this_tri = Delaunay(vertices[: offset + 1 + valid_vertices])
        points_simplex = this_tri.find_simplex(points)
        event_simplex = points_simplex[: event_points.shape[0]]

        if (points_simplex != -1).all():
            valid_vertices += 1

    return vertices[offset:]


corners = np.array([
        [m1_min, z_min, q_min],
        [m1_min, z_min, q_max],
        [m1_min, z_max, q_min],
        [m1_min, z_max, q_max],
        [m1_max, z_min, q_min],
        [m1_max, z_min, q_max],
        [m1_max, z_max, q_min],
        [m1_max, z_max, q_max],
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

branch_names = ["tri", "corner_w", "chi", "tilt"]
ndims = dict(zip(branch_names, [4, 8, 2, 2]))
nleaves_min = dict(zip(branch_names, [4, 1, 1, 1]))
nleaves_max = dict(zip(branch_names, [40, 1, 1, 1]))

start_with_this_many = 6

priors = {
    "tri": {
        0: uniform_dist(m1_min, m1_max),
        1: uniform_dist(z_min, z_max),
        2: uniform_dist(q_min, q_max),
        3: uniform_dist(-5, 15),
    },
    "corner_w": {
        i: uniform_dist(-5, 15) for i in range(ndims["corner_w"])
    },
    "chi": {
        0: uniform_dist(0, 0.95),
        1: uniform_dist(0.005, 0.25)
    },
    "tilt": {
        0: uniform_dist(0, 1),
        1: uniform_dist(0.1, 4)
    }
}

coords = {}
inds = {}

for branch in branch_names:
    coords[branch] = np.zeros((ntemps, nwalkers, nleaves_max[branch], ndims[branch]))
    inds[branch] = np.zeros((ntemps, nwalkers, nleaves_max[branch]), dtype=bool)

    for i in range(ndims[branch]):
        coords[branch][..., i] = priors[branch][i].rvs(
            size=(ntemps, nwalkers, nleaves_max[branch])
        )

    if branch == "tri":
        inds["tri"][:, :, :start_with_this_many] = True
    else:
        inds[branch][:, :, :] = True


barycenters = observed_events.reshape(num_events, num_samples, 7).mean(axis=1)
for t, w in product(range(ntemps), range(nwalkers)):

    le_log = -np.inf
    for _ in trange(10_000):
        init_proposal = {
            "tri": np.c_[
                make_valid_delaunay(
                    barycenters[:, :3],
                    start_with_this_many,
                    log_like_fn.corners,
                ),
                priors["tri"][2].rvs(size=start_with_this_many),
                
            ]
        } | {
            branch: np.array(
                [priors[branch][dim_indx].rvs() for dim_indx in range(ndims[branch])]
            ).squeeze()
            for branch in priors
            if branch != "tri"
        }

        le_log = (
            log_like_fn(
                [
                    init_proposal[key]
                    for key in ["tri", "corner_w", "chi", "tilt"]
                ]
            )
            or -1e300
        )

        if le_log > -1e300:
            break
    else:
        raise ValueError("Didn't work")

    print(t, w, le_log)
    for branch in init_proposal:
        coords[branch][
            t, w, : (start_with_this_many if branch == "tri" else nleaves_max[branch])
        ] = init_proposal[branch]

    print(le_log)

from eryn.moves import MHMove

class BinGaussRelMove(MHMove):

    def __init__(
        self, sigma_vertices, weights_scale, ind_leaf, branch_name, *args, **kwargs
    ):

        self.sigma_vertices = sigma_vertices
        self.weights_scale = weights_scale
        self.branch_name = branch_name
        self.ind_leaf = ind_leaf

        super().__init__(*args, **kwargs)

    def get_proposal(self, branches_coords, random, branches_inds, **kwargs):

        coords = branches_coords[self.branch_name]
        inds = branches_inds[self.branch_name]

        ntemps, nwalkers, nleaves_max, ndim = coords.shape

        # replace is always true
        false_mask = ~inds[:, :, self.ind_leaf]
        random_indices = np.full((ntemps, nwalkers), -1, dtype=int)

        valid_mask = inds.any(axis=2)
        random_choice = inds * np.random.rand(ntemps, nwalkers, nleaves_max)
        random_selected = np.argmax(random_choice, axis=-1)
        random_indices[valid_mask] = random_selected[valid_mask]

        result_indices = np.where(false_mask, random_indices, self.ind_leaf)

        inds_leaf = np.take_along_axis(inds, result_indices[..., None], axis=2)[:, :, 0]
        coords_leaf = np.take_along_axis(
            coords, result_indices[..., None, None], axis=2
        )[:, :, 0]

        new_vertices = coords_leaf[:, :, :-1] + self.sigma_vertices * random.randn(
            ntemps, nwalkers, ndim - 1
        )

        sigma_weights = self.weights_scale
        new_weights = coords_leaf[:, :, -1] + sigma_weights * np.random.randn(
            ntemps, nwalkers
        )

        new_params = np.dstack([new_vertices, new_weights])

        proposal = {self.branch_name: np.copy(coords)}
        row_indices, col_indices = np.meshgrid(
            np.arange(ntemps), np.arange(nwalkers), indexing="ij"
        )
        proposal[self.branch_name][
            row_indices, col_indices, result_indices, :
        ] = new_params

        factors = 0.0

        return proposal, factors


moves = [
    BinGaussRelMove(
        sigma_vertices=0.1 * np.array([m1_max - m1_min, z_max - z_min, q_max - q_min]),
        weights_scale=1.,
        ind_leaf=ind_leaf,
        branch_name="tri",
    )
    for ind_leaf in range(nleaves_max["tri"])
]
moves += [
    StretchMove(
        gibbs_sampling_setup=["corner_w", "chi", "tilt"],
        live_dangerously=True,
    )
]
moves += [
    StretchMove(
        gibbs_sampling_setup=["tilt"],
        live_dangerously=True
    )
]

prior_move = DistributionGenerateRJ(
    {key: ProbDistContainer(priors[key]) for key in priors},
    nleaves_min={key: val for key, val in nleaves_min.items()},
    nleaves_max={key: val for key, val in nleaves_max.items()},
)
rj_moves = [prior_move]


state = State(coords, inds=inds)

with Pool(nproc) as pool:
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
    last_sample = ensemble.run_mcmc(state, nsteps, burn=nburn, progress=True, thin_by=1)

print(ensemble.backend.rj_accepted)

with open(f"{label}/backend", "wb") as f:
    pickle.dump(ensemble.backend, f)

print("Saved!")
# In[154]:


chain = ensemble.get_chain()
leaves = ensemble.get_nleaves()


for t in range(ntemps):
    fig, ax = plt.subplots()
    ax.set_rasterized(True)
    ax.set(
        xlabel="Walker index",
        ylabel="In-Model Acceptance fraction",
        yscale="linear",
        yticks=np.arange(0, 105, 5),
    )
    ax.grid()
    for move in ensemble.moves:
        ax.plot(100 * move.acceptance_fraction[t], ".-")

    for move in ensemble.rj_moves:
        ax.plot(100 * move.acceptance_fraction[t], "o--", markerfacecolor="white")

    savefig(fig, f"acceptance_temp_{t}")

fig, ax = plt.subplots()
ax.set_rasterized(True)
ax.grid()
for w in range(nwalkers):
    ax.plot(ensemble.get_log_like()[:, 0, w], label=w)
ax.legend()
savefig(fig, "likelihood")

fig, ax = plt.subplots()
ax.set_rasterized(True)
ax.hist(leaves["tri"][:, 0].ravel(), bins="auto")
savefig(fig, "leaves_distro")
