r"""
FanCI APIG class module.

"""

import numpy as np

from pyci import doci

from objective.base import BaseFanCI, BaseOverlap


__all___ = [
    'APIGFanci',
    ]


class APIGFanCI(BaseFanCI):
    r"""
    APIG FanCI class.

    """
    def __init__(self, ham, nocc, norm_det=-1, norm_param=-1, ndet_pspace=None):
        r"""
        Initialize the APIG FanCI problem.

        """
        # handle default arguments
        if ndet_pspace is None:
            ndet_pspace = ham.nbasis * nocc

        # check system dimensions
        if ham.nbasis < nocc:
            raise ValueError('ham.nbasis must be >= nocc')
        if ndet_pspace < ham.nbasis * nocc:
            raise ValueError('ndet_pspace must be >= ham.nbasis * nocc')

        # Initialize base class
        super().__init__(ham, nocc, norm_det=-1, norm_param=-1, ndet_pspace=None)

    def init_system(self, ham, nocc, ndet_pspace):
        r"""
        Initialize the system dimensions and CI wavefunction.

        """
        # assign hamiltonian
        self.ham = ham

        # build doci wavefunction
        self.wfn = doci.wfn(ham.nbasis, nocc)
        self.wfn.add_all_dets()
        if len(self.wfn) < ndet_pspace:
            raise ValueError('Cannot generate `len_pspace` determinants')

        # assign system dimensions
        self.nbasis = ham.nbasis
        self.nocc_up = nocc
        self.nocc_dn = nocc
        self.ndet_pspace = ndet_pspace
        self.ndet_sspace = len(self.wfn)

    def init_overlap(self, ham, nocc, ndet_pspace):
        r"""
        Initialize the FanCI overlap operator.

        """
        # return overlap operator
        return APIGOverlap(ham.nbasis, nocc)


class APIGOverlap(BaseOverlap):
    r"""
    APIG overlap operator.

    """

    def __init__(self, nbasis, nocc):
        r"""
        Initialize APIG overlap operator.

        """
        self.nbasis = nbasis
        self.nocc = nocc

    def overlap(self, x, occs_array):
        r"""
        Compute the overlap vector.

        """
        x_mat = x.reshape(self.nbasis, self.nocc)
        y = np.empty(occs_array.shape[0], dtype=x.dtype)
        for i, occs in enumerate(occs_array):
            y[i] = self.permanent(x_mat[occs])
        return y

    def overlap_deriv(self, x, occs_array):
        r"""
        Compute the overlap derivative matrix.

        """
        x_mat = x.reshape(self.nbasis, self.nocc)
        y = np.empty((occs_array.shape[0], self.nbasis * self.nocc), dtype=x.dtype)
        y_mat_array = y.reshape(occs_array.shape[0], self.nbasis, self.nocc)
        for y_mat, occs in zip(y_mat_array, occs_array):
            for i in range(self.nbasis):
                for j in range(self.nocc):
                    y_mat[i, j] = self.permanent_deriv(x_mat[occs], i, j)
        return y

    @staticmethod
    def permanent(matrix):
        r"""
        Compute the permanent of a square matrix using Glynn's algorithm.

        Gray code generation from Knuth, D. E. (2005). The Art of Computer
        Programming, Volume 4, Fascicle 2: Generating All Tuples and Permutations.

        Glynn's algorithm from Glynn, D. G. (2010). The permanent of a square
        matrix. European Journal of Combinatorics, 31(7), 1887-1891.

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
        Compute the derivative of the permanent of a square matrix with respect to element (i, j).

        """
        # TODO
        pass
