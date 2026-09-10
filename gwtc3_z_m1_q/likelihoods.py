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

def chi_log_pdf(chi, mu_chi, var_chi):
    nu = mu_chi * (1 - mu_chi) / var_chi - 1.0
    alpha_chi = mu_chi * nu
    beta_chi = (1 - mu_chi) * nu
    zero_mask = np.logical_or(nu < 0, np.logical_or(alpha_chi < 0, beta_chi < 0))
    return np.where(zero_mask, -1e300, stats.beta(a=alpha_chi, b=beta_chi).logpdf(chi))


def tilts_log_pdf(tilt_1, tilt_2, zeta, sigma_t, mu_t=0):
    truncated_gaussian = stats.truncnorm(
        loc=mu_t, scale=sigma_t, a=(-1-mu_t) /sigma_t, b=(1-mu_t) /sigma_t
    )
    gauss_prod = truncated_gaussian.pdf(tilt_1) * truncated_gaussian.pdf(tilt_2)
    return np.where(
        gauss_prod <= 0.0, -1e300, np.log(0.25 * (1 - zeta) + zeta * gauss_prod)
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
                self.events[:, 5], self.events[:, 6], 
                zeta_sigma_t[0], zeta_sigma_t[1], 
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
                self.detected_injections[:, 5], self.detected_injections[:, 6],
                zeta_sigma_t[0], zeta_sigma_t[1],
            )
        )

        Nxi_to_sum = Nxi_presum * inj_inside_tri / self.detected_injections_prior
        Nxi = Nxi_to_sum.sum() / self.num_injections
        var_Nxi = (
            ((Nxi_to_sum**2).sum() / (self.num_injections - 1) - Nxi**2)
            / self.num_injections
        )

        if Nxi**2 /var_Nxi <= 4*self.num_events:
            logger.debug("Not enough injection stuff")
            return self.minus_infinity
        
        result = (log_bayes_factors - self.log_num_samples).sum() - Nxi
        return result.item()

