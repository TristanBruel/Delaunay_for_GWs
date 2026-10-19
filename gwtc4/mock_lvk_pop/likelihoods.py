import numpy as np
from scipy import special, stats
from local_utils import delaunaytor


import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)




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

        inj_inside = inj_simplex != -1

        if (log_Nxi>709).any():
            return self.minus_infinity

        return (
            log_dNdtheta_samples,
            samples_inside,
            log_Nxi,
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
            events_log_prior=events_log_prior,
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
        (inner_tri_parameters, corner_weights) = (
            population_parameters
        )

        triangulation_parameters = np.vstack([
            inner_tri_parameters,
            np.c_[self.corners, corner_weights.T]
        ])

        if (tri_result := self.delaunay_rate.rate(triangulation_parameters)) is None:
            return self.minus_infinity
        (
            log_dNdtheta_tri,
            samples_inside_tri,
            log_Nxi_tri,
            inj_inside_tri,
        ) = tri_result

        # parameter_keys = ["m1", "z", "q"]
        log_dNdtheta = (
            log_dNdtheta_tri
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

        Nxi_presum = np.exp(
            log_Nxi_tri
        )

        Nxi_to_sum = Nxi_presum * inj_inside_tri / self.detected_injections_prior
        Nxi = Nxi_to_sum.sum() / self.num_injections
        var_Nxi = (
            ((Nxi_to_sum**2).sum() / (self.num_injections - 1) - Nxi**2)
            / self.num_injections
            )
        #var_Nxi *= (
        #        self.num_events
        #        / self.delaunay_rate.delaunay_interpolator.compute_events()
        #        )** 2

        ## Threshold on effective sample size (GWTC-3) ##
        if Nxi**2 /var_Nxi <= 4*self.num_events:
            logger.debug("Not enough injection stuff")
            return self.minus_infinity

        ## Threshold on variance in log-likelihood estimator (GWTC-4+) ##
        #if (var_log_NL + var_Nxi) > 1:
        #    logger.debug(f"Variance in log-likelihood estimator exceeds 1")
        #    return self.minus_infinity

        # Correction for the likelihood (Heinzel & Vitale 2025)
        #NL_j = NL_j * np.exp(-var_Nxi/2)

        return (np.log(NL_j).sum() - Nxi).item()
