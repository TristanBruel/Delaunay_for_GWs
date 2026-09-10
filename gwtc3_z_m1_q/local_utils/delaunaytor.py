from functools import cached_property
import logging
import numpy as np

from scipy.spatial import Delaunay as CPUDelaunay
from scipy.special import logsumexp as CPUlogsumexp
from scipy.special import factorial

try:
    import cupy as cp
    from cupyx.scipy.spatial import Delaunay as GPUDelaunay
    from cupyx.scipy.special import logsumexp as GPUlogsumexp

    cupy_available = True
except:
    cupy_available = False

logger = logging.getLogger(__name__)


class CPUDelaunayInterpolator:
    tri = CPUDelaunay

    def triangulate(self, points_and_weights):
        """
        points_and_weights: (num_points, ndim_points + 1)
        """
        self.triangulation = self.tri(points_and_weights[:, :-1])
        self.weights = points_and_weights[:, -1]

    def interpolate(self, query_points):
        samples_simplex, samples_b = self.simplex_and_barycenters(query_points)
        return self._interpolate(samples_simplex, samples_b)

    def simplex_and_barycenters(self, query_points):
        """
        query_points: (num_points, ndim_points)
        """
        samples_simplex = self.triangulation.find_simplex(query_points)

        r3 = self.triangulation.transform[samples_simplex, -1]
        Tinv = self.triangulation.transform[samples_simplex, :-1]

        samples_b_except_last = np.einsum("ijk,ik->ij", Tinv, (query_points - r3))
        samples_b = np.c_[samples_b_except_last, 1 - samples_b_except_last.sum(axis=-1)]

        return samples_simplex, samples_b

    def _interpolate(self, samples_simplex, samples_b):
        samples_weights = self.weights[self.triangulation.simplices[samples_simplex]]
        return (samples_b * samples_weights).sum(axis=-1)

    def sample(self, num_points, weight_fun=None):
        volumes = self.volumes()

        accepted_so_far = 0
        out = np.zeros((num_points, self.triangulation.points.shape[1]))

        if weight_fun is None:
            weight_fun = lambda points: np.ones(points.shape[0])
        
        while accepted_so_far < num_points:  # don't really like a while loop...
            random_simplices = np.random.choice(
                np.arange(self.triangulation.simplices.shape[0]),
                size=num_points,
                p=volumes / volumes.sum(),
            )
            random_bcoords = np.random.dirichlet(
                alpha=np.ones(self.triangulation.points.shape[1] + 1), 
                size=num_points
            )
            
            cartesian_coordinates = (
                random_bcoords[..., None]
                * self.triangulation.points[self.triangulation.simplices][random_simplices]
            ).sum(axis=1)
            
            sampling_weights = weight_fun(cartesian_coordinates)
            
            log_to_accept = (
                np.log(sampling_weights) + self._interpolate(random_simplices, random_bcoords) - self.weights.max()
            )

            accepted = np.log(np.random.random(size=num_points)) < log_to_accept
            accepted_points = cartesian_coordinates[accepted]

            if accepted_points.shape[0] == 0:
                continue

            if accepted_so_far + accepted_points.shape[0] <= num_points:
                num_accepted = accepted_points.shape[0]
            else:
                num_accepted = num_points - accepted_so_far

            out[accepted_so_far : accepted_so_far + num_accepted] = accepted_points[
                :num_accepted
            ]
            accepted_so_far += num_accepted

        return out

    def volumes(self):
        vertex_coords = self.triangulation.points[self.triangulation.simplices]
        return np.abs(
            np.linalg.det(vertex_coords[:, 1:] - vertex_coords[:, [0]])
        ) / factorial(vertex_coords.shape[-1])

    def compute_events(self):

        simplices = self.triangulation.simplices
        weights = self.weights[simplices]
        n_simplices, n_vertices = weights.shape
        n = n_vertices - 1
        volumes = self.volumes()

        numer = np.exp(weights)
        wi = weights[:, :, None]
        wj = weights[:, None, :]
        diff = wi - wj

        mask = ~np.eye(n_vertices, dtype=bool)
        denom = np.prod(diff[:, mask].reshape(n_simplices, n_vertices, n_vertices-1), axis=-1)
        # Check for all-equal weights per simplex
        all_equal = np.all(np.abs(weights - weights[:, [0]]) < 1e-12, axis=1)
        bary_sum = np.sum(numer / denom, axis=1)
        result = bary_sum * volumes * factorial(n)
        # For all-equal weights, use the constant formula
        result[all_equal] = volumes[all_equal] * np.exp(weights[all_equal, 0])
        return result.sum()

if cupy_available:

    class GPUDelaunayInterpolator(CPUDelaunayInterpolator):
        tri = GPUDelaunay
        EPS_DELAUNAY = 100 * cp.finfo(cp.double).eps

        def simplex_and_barycenters(self, query_points):
            samples_simplex, samples_b = self.triangulation._find_simplex_coordinates(
                query_points, eps=self.EPS_DELAUNAY, find_coords=True
            )
            return samples_simplex, samples_b


class DelaunayLogLikelihood:

    def __init__(
        self,
        events,
        events_log_prior,
        num_events,
        num_samples,
        detected_injections,
        detected_injections_prior,
        num_injections,
        minus_infinity=-1e300,
    ):
        self.logsumexp = CPUlogsumexp
        self.delaunay_interpolator = CPUDelaunayInterpolator()

        # Events should have all samples concatenated along last axis.
        self.events = events
        self.num_events = num_events
        self.num_samples = num_samples
        self.events_log_prior = (
            events_log_prior.reshape(self.num_events, self.num_samples)
        )
        self.detected_injections = detected_injections
        self.detected_injections_prior = detected_injections_prior
        self.num_injections = num_injections

        self.minus_infinity = minus_infinity


    def __call__(self, pop_params_cpu):

        try:
            self.delaunay_interpolator.triangulate((pop_params_cpu))
        except:
            logger.info("Triangulation straight out failed")
            return self.minus_infinity

        events_simplex, events_b = self.delaunay_interpolator.simplex_and_barycenters(
            self.events
        )

        samples_inside = (events_simplex != -1).reshape(
            self.num_events, -1
        )
        actual_num_samples = samples_inside.sum(axis=-1)
        if (actual_num_samples < 1).any():
            logger.info("Some events ended up without samples")
            return self.minus_infinity

        log_dNdtheta_samples = self.delaunay_interpolator._interpolate(
            events_simplex, events_b
        ).reshape(self.num_events, -1)

        to_integrate = log_dNdtheta_samples - self.events_log_prior
        log_bayes_factors = self.logsumexp(to_integrate, b=samples_inside, axis=-1)

        if not self._valid_effective_sample_size(
            log_bayes_factors, to_integrate, samples_inside
        ):
            return self.minus_infinity

        inj_simplex, inj_b = self.delaunay_interpolator.simplex_and_barycenters(
            self.detected_injections
        )
        log_Nxi = self.delaunay_interpolator._interpolate(inj_simplex, inj_b)

        maybe_infinity = np.exp(log_Nxi)
        if np.isinf(maybe_infinity).any():
            return self.minus_infinity

        inj_inside = inj_simplex != -1
        Nxi = (
            maybe_infinity * inj_inside / self.detected_injections_prior
        ).sum() / self.num_injections

        return (
            (log_bayes_factors - np.log(actual_num_samples)).sum() - Nxi
        )

    def _valid_effective_sample_size(
        self, log_bayes_factors, to_integrate, samples_inside
    ):
        log_variance_likes = self.logsumexp(2 * to_integrate, b=samples_inside, axis=-1)
        log_effective_sample_sizes = 2 * log_bayes_factors - log_variance_likes

        if (
            log_effective_sample_sizes < np.log(self.num_events)
        ).any():
            print("Effective sample size is too low")
            return False
        if np.isnan(log_variance_likes).any():
            print("Variances are bad")
            return False

        return True
