r"""
FanCI APIG module.

"""

from itertools import product

import numpy as np

import pyci

from .fanci import FanCI


__all___ = [
    'APIG',
    ]


class APIG(FanCI):
    r"""
    APIG FanCI class.

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
        nparam = ham.nbasis * nocc + 1

        # Handle default nproj
        nproj = nparam if nproj is None else nproj

        # Handle default wfn
        if wfn is None:
            wfn = pyci.doci_wfn(ham.nbasis, nocc)
        elif wfn.nocc_up != nocc or wfn.nocc_dn != nocc:
            raise ValueError('wfn.nocc_{up,dn} does not match nocc parameter')

        # Initialize base class
        FanCI.__init__(self, ham, wfn, nproj, nparam, **kwargs)

    def compute_overlap(self, x, occs_array):
        r"""
        Compute the overlap vector.

        """
        x_mat = x.reshape(self.nbasis, self.nocc_up)
        y = np.empty(occs_array.shape[0], dtype=x.dtype)
        for i, occs in enumerate(occs_array):
            y[i] = self.permanent(x_mat[occs])
        return y

    def compute_overlap_deriv(self, x, occs_array):
        r"""
        Compute the overlap derivative matrix.

        """
        x_mat = x.reshape(self.nbasis, self.nocc_up)
        y = np.empty((occs_array.shape[0], self.nactive - self.mask[-1]), dtype=x.dtype)
        for y_row, occs in zip(y, occs_array):
            i = 0
            j = 0
            for k, l in product(range(self.nbasis), range(self.nocc_up)):
                if self.mask[i]:
                    y_row[j] = self.permanent_deriv(x_mat[occs], k, l)
                    j += 1
                i += 1
        return y

    @staticmethod
    def permanent(matrix):
        r"""
        Compute the permanent of a square matrix using Glynn's algorithm.

        Gray code generation from Knuth, D. E. (2005). The Art of Computer Programming,
        Volume 4, Fascicle 2: Generating All Tuples and Permutations.

        Glynn's algorithm from Glynn, D. G. (2010). The permanent of a square matrix.
        European Journal of Combinatorics, 31(7), 1887-1891.

        """
        # initialize gray code
        n = matrix.shape[0]
        pos = 0
        sign = 1
        bound = n - 1
        delta = np.ones(n, dtype=int)
        graycode = np.arange(n, dtype=int)
        # iterate over every delta
        perm = np.prod(np.sum(matrix, axis=0))
        while pos < bound:
            # update delta and add term to permanent
            sign *= -1
            delta[bound - pos] *= -1
            perm += sign * np.prod(delta.dot(matrix))
            # update gray code and position
            graycode[0] = 0
            graycode[pos] = graycode[pos + 1]
            graycode[pos + 1] = pos + 1
            pos = graycode[0]
        # divide by constant external term and return
        return perm / (2 ** bound)

    @staticmethod
    def permanent_deriv(matrix, i, j):
        r"""
        Compute the derivative of the permanent of a square matrix.

        """
        # TODO
        raise NotImplementedError
