import numpy as np
from scipy import special, stats
from scipy.interpolate import InterpolatedUnivariateSpline as spline
from local_utils import delaunaytor

from astropy.cosmology import Planck15

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)


z_max = 2.5
z_anchors = np.linspace(0, z_max, 1_000)
dVc_anchors = 4 * np.pi * Planck15.differential_comoving_volume(z_anchors).value
dVc_spline = spline(z_anchors, dVc_anchors)

def z_pdf_not_norm(z, kappa):
    return (1 + z) ** (kappa - 1) * dVc_spline(z)

def z_log_pdf(z, kappa, z_max=z_max):
    out_bounds = z > z_max
    z_grid = np.linspace(0, z_max, 1_000)
    p_of_z = z_pdf_not_norm(z, kappa) / np.trapezoid(
        y=z_pdf_not_norm(z_grid, kappa), x=z_grid
    )
    return np.where(out_bounds, -1e300, np.log(p_of_z))



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

        return (
            log_dNdtheta_samples,
            samples_inside,
        )


class MQDelaunay:

    def __init__(
        self,
        events,
        events_log_prior,
        num_events,
        num_samples,
        corners,
        minus_infinity=-1e300,
    ):
        delaunay_indices = (0,2)
        if len(delaunay_indices) != 2:
            raise ValueError("Two is the number of dimensions I shall triangulate over")

        self.delaunay_rate = SimpleDelaunay(
            events=events[:, delaunay_indices],
            events_log_prior=events_log_prior,
            num_events=num_events,
            num_samples=num_samples,
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

        self.minus_infinity = minus_infinity

    def __call__(self, population_parameters):
        (inner_tri_parameters, corner_weights, kappa) = (
            population_parameters
        )

        kappa = kappa.squeeze()

        triangulation_parameters = np.vstack([
            inner_tri_parameters,
            np.c_[self.corners, corner_weights.T]
        ])

        if (tri_result := self.delaunay_rate.rate(triangulation_parameters)) is None:
            return self.minus_infinity
        (
            log_dNdtheta_tri,
            samples_inside_tri,
        ) = tri_result

        # parameter_keys = ["m1", "z", "q"]
        log_dNdtheta = (
            log_dNdtheta_tri
            + z_log_pdf(
                self.events[:,1],
                kappa,
                )
        ).reshape(self.num_events, self.num_samples)
        to_integrate = log_dNdtheta - self.events_log_prior
        if (to_integrate>709).any():
            return self.minus_infinity

        NL_j_to_sum = samples_inside_tri * np.exp(to_integrate)
        NL_j = NL_j_to_sum.sum(axis=-1) / self.num_samples
        var_NL_j = (
            (NL_j_to_sum**2).sum(axis=-1) / (self.num_samples - 1) - NL_j**2
        ) / self.num_samples
        var_log_NL = np.inf if (NL_j < 1e-20).any() else (var_NL_j / NL_j**2).sum()

        ## Threshold on variance in log-likelihood estimator (GWTC-4+) ##
        if var_log_NL > 1:
            logger.debug(f"Variance in log-likelihood estimator exceeds 1")
            return self.minus_infinity

        return (np.log(NL_j).sum()).item()
