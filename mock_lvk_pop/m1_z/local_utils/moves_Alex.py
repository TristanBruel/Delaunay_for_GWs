import numpy as np

from eryn.moves import MHMove


class BinGaussRelMove(MHMove):

    def __init__(
        self, sigma_vertices, weights_scale, branch_name, *args, **kwargs
    ):

        self.sigma_vertices = sigma_vertices
        self.weights_scale = weights_scale
        self.branch_name = branch_name
        #self.ind_leaf = ind_leaf

        super().__init__(*args, **kwargs)

    def get_proposal(self, branches_coords, random, branches_inds, **kwargs):

        coords = branches_coords[self.branch_name]
        inds = branches_inds[self.branch_name]

        ntemps, nwalkers, nleaves_max, ndim = coords.shape

        # replace is always true
        # false_mask = ~inds[:, :, self.ind_leaf]
        # random_indices = np.full((ntemps, nwalkers), -1, dtype=int)

        # valid_mask = inds.any(axis=2)
        random_choice = inds * np.random.rand(ntemps, nwalkers, nleaves_max)
        random_selected = np.argmax(random_choice, axis=2)
        # random_indices[valid_mask] = random_selected[valid_mask]
        random_indices = random_selected

        # result_indices = np.where(false_mask, random_indices, self.ind_leaf)
        result_indices = random_indices

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
