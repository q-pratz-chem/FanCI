r"""
FanCI AP1roG module.

"""

from itertools import product

import numpy as np

import pyci

from .fanci import FanCI
from .apig import APIG


__all___ = [
    'AP1roG',
    ]


class AP1roG(APIG):
    r"""
    AP1roG FanCI class.

    """
    def __init__(self, ham, nocc, nproj=None, wfn=None, **kwargs):
        r"""
        Initialize the FanCI problem.

        Parameters
        ----------
        ham : pyci.hamiltonian
            Hamiltonian.
        nocc : int
            Number of occupied indices.
        nproj : int, optional
            Number of determinants in P space.
        wfn : pyci.doci_wfn, optional
            If specified, this wfn defines the P space.

        """
        # Compute number of parameters (c_kl + energy)
        nparam = nocc * (ham.nbasis - nocc) + 1

        # Handle default nproj
        nproj = nparam if nproj is None else nproj

        # Handle default wfn (P space == single pair excitations)
        if wfn is None:
            wfn = pyci.doci_wfn(ham.nbasis, nocc)
            wfn.add_excited_dets(1)
        elif wfn.nocc_up != nocc or wfn.nocc_dn != nocc:
            raise ValueError('wfn.nocc_{up,dn} does not match nocc parameter')

        # Assign reference occupations for compute_overlap{,_deriv}
        self.ref_occs = np.arange(nocc, dtype=pyci.c_int)

        # Initialize base class
        FanCI.__init__(self, ham, wfn, nproj, nparam, **kwargs)

    def compute_overlap(self, x, occs_array):
        r"""
        Compute the overlap vector.

        """
        x_mat = x.reshape(self.nocc_up, self.nvir_up)
        y = np.empty(occs_array.shape[0], dtype=x.dtype)
        for i, occs in enumerate(occs_array):
            holes = np.setdiff1d(self.ref_occs, occs, assume_unique=True)
            if holes.size:
                particles = np.setdiff1d(occs, self.ref_occs, assume_unique=True)
                particles -= self.nocc_up
                y[i] = self.permanent(x_mat[holes][:, particles])
            else:
                y[i] = 1
        return y

    def compute_overlap_deriv(self, x, occs_array):
        r"""
        Compute the overlap derivative matrix.

        """
        x_mat = x.reshape(self.nocc_up, self.nvir_up)
        y = np.empty((occs_array.shape[0], self.nactive - self.mask[-1]), dtype=x.dtype)
        for y_row, occs in zip(y, occs_array):
            holes = np.setdiff1d(self.ref_occs, occs, assume_unique=True)
            if holes.size:
                particles = np.setdiff1d(occs, self.ref_occs, assume_unique=True)
                particles -= self.nocc_up
                x_mat_slice = x_mat[holes][:, particles]
                i = 0
                j = 0
                for k, l in product(range(self.nocc_up), range(self.nvir_up)):
                    if self.mask[i]:
                        y_row[j] = self.permanent_deriv(x_mat_slice, k, l)
                        j += 1
                    i += 1
            else:
                y_row[:] = 0
        return y
