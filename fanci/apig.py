r"""
FanCI APIG module.

"""

from typing import Any

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
    def __init__(self, ham: pyci.hamiltonian, nocc: int, nproj: int = None,
            wfn: pyci.doci_wfn = None, **kwargs: Any) -> None:
        r"""
        Initialize the FanCI problem.

        Parameters
        ----------
        ham : pyci.hamiltonian
            PyCI Hamiltonian.
        nocc : int
            Number of occupied orbitals.
        nproj : int, optional
            Number of determinants in projection ("P") space.
        wfn : pyci.doci_wfn, optional
            If specified, this PyCI wave function defines the projection ("P") space.
        kwargs : Any, optional
            Additional keyword arguments for base FanCI class.

        """
        # Compute number of parameters (c_kl + energy)
        nparam = ham.nbasis * nocc + 1

        # Handle default nproj
        nproj = nparam if nproj is None else nproj

        # Handle default wfn
        if wfn is None:
            wfn = pyci.doci_wfn(ham.nbasis, nocc)
        elif not isinstance(wfn, pyci.doci_wfn):
            raise TypeError('wfn must be a `pyci.doci_wfn`')
        elif wfn.nocc_up != nocc or wfn.nocc_dn != nocc:
            raise ValueError('wfn.nocc_{up,dn} does not match nocc parameter')

        # Initialize base class
        FanCI.__init__(self, ham, wfn, nproj, nparam, **kwargs)

    def compute_overlap(self, x: np.ndarray, occ_array: np.ndarray) -> np.ndarray:
        r"""
        Compute the FanCI overlap vector.

        Parameters
        ----------
        x : np.ndarray
            Parameter array, [p_0, p_1, ..., p_n].
        occ_array : np.ndarray
            Array of determinant occupations for which to compute overlap.

        Returns
        -------
        ovlp : np.ndarray
            Overlap array.

        """
        # Reshape parameter array to APIG matrix
        x_mat = x.reshape(self._wfn.nbasis, self._wfn.nocc_up)

        # Compute overlaps of occupation vectors
        y = np.empty(occs_array.shape[0], dtype=pyci.c_double)
        for i, occs in enumerate(occs_array):
            y[i] = permanent(x_mat[occs])
        return y

    def compute_overlap_deriv(self, x: np.ndarray, occ_array: np.ndarray) -> np.ndarray:
        r"""
        Compute the FanCI overlap derivative matrix.

        Parameters
        ----------
        x : np.ndarray
            Parameter array, [p_0, p_1, ..., p_n].
        occ_array : np.ndarray
            Array of determinant occupations for which to compute overlap derivative.

        Returns
        -------
        ovlp : np.ndarray
            Overlap derivative array.

        """
        # Reshape parameter array to APIG matrix
        x_mat = x.reshape(self._wfn.nbasis, self._wfn.nocc_up)

        # Shape of y is (no. determinants, no. active parameters excluding energy)
        y = np.zeros((occs_array.shape[0], self._nactive - self._mask[-1]), dtype=pyci.c_double)

        # Iterate over occupation vectors
        for y_row, occs in zip(y, occs_array):
            # Iterate over all parameters (i), active parameters (j)
            i = 0
            j = 0
            for k in range(self._wfn.nbasis):
                for l in range(self._wfn.nocc_up):
                    # Check if element is active
                    if self._mask[i]:
                        k_pos = occs.searchsorted(k)
                        # Check if row is present
                        if k_pos != occs.size and occs[k_pos] == k:
                            k_slice = np.delete(occs, k_pos, axis=0)
                            # Check if any rows remain after deleting k_pos
                            if k_slice.size:
                                # Check if column is present
                                l_pos = occs.searchsorted(l)
                                if l_pos != occs.size and occs[l_pos] == l:
                                    # Compute permanent of (k, l) minor
                                    y_row[j] = permanent(np.delete(x_mat[k_slice], l, axis=1))
                        # Go to next active parameter
                        j += 1
                    # Go to next parameter
                    i += 1

        # Return overlap derivative matrix
        return y


def permanent(matrix : np.ndarray) -> float:
    r"""
    Compute the permanent of a square matrix using Glynn's algorithm.

    Gray code generation from Knuth, D. E. (2005). The Art of Computer Programming,
    Volume 4, Fascicle 2: Generating All Tuples and Permutations.

    Glynn's algorithm from Glynn, D. G. (2010). The permanent of a square matrix.
    European Journal of Combinatorics, 31(7), 1887-1891.

    Parameters
    ----------
    matrix : np.ndarray
        Square matrix.

    Returns
    -------
    result : matrix.dtype
        Permanent of the matrix.

    """
    # Initialize gray code
    n = matrix.shape[0]
    pos = 0
    sign = 1
    bound = n - 1
    delta = np.ones(n, dtype=np.int)
    graycode = np.arange(n, dtype=np.intp)

    # Iterate over every delta
    result = np.prod(np.sum(matrix, axis=0))
    while pos < bound:
        # Update delta and add term to permanent
        sign *= -1
        delta[bound - pos] *= -1
        result += sign * np.prod(delta.dot(matrix))
        # Update gray code and position
        graycode[0] = 0
        graycode[pos] = graycode[pos + 1]
        graycode[pos + 1] = pos + 1
        pos = graycode[0]

    # Divide by constant factor
    return result / (2 ** bound)
