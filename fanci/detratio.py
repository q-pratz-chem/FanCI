r"""
FanCI Determinant ratio module.

Adapted from:
https://github.com/QuantumElephant/fanpy/blob/master/wfns/wfn/quasiparticle/det_ratio.py

"""

from typing import Any, Sequence, Union

import numpy as np

import pyci

from .fanci import FanCI


__all___ = [
        'DetRatio',
        ]


class DetRatio(FanCI):
    r"""
    Determinant ratio FanCI class.

    """

    def __init__(self, ham: pyci.hamiltonian, nocc: int, numerator : int, denominator: int,
            nproj: int = None, wfn: pyci.doci_wfn = None, **kwargs: Any) -> None:
        r"""
        Initialize the FanCI problem.

        Parameters
        ----------
        ham : pyci.hamiltonian
            PyCI Hamiltonian.
        nocc : int
            Number of occupied orbitals.
        numerator : int
            Number of matrices in the numerator.
        denominator : int
            Number of matrices in the denominator.
        nproj : int, optional
            Number of determinants in projection ("P") space.
        wfn : pyci.doci_wfn, optional
            If specified, this PyCI wave function defines the projection ("P") space.
        kwargs : Any, optional
            Additional keyword arguments for base FanCI class.

        """
        # Parse arguments
        # ---------------

        # Check number of matrices
        nmatrices = numerator + denominator
        if nmatrices % 2:
            raise ValueError('Number of matrices cannot be odd')

        # Compute number of parameters (c_{i;kl} + energy)
        nparam = nmatrices * ham.nbasis * nocc + 1

        # Handle default nproj
        nproj = nparam if nproj is None else nproj

        # Handle default wfn
        if wfn is None:
            wfn = pyci.doci_wfn(ham.nbasis, nocc)
        elif not isinstance(wfn, pyci.doci_wfn):
            raise TypeError("wfn must be a `pyci.doci_wfn`")
        elif wfn.nocc_up != nocc or wfn.nocc_dn != nocc:
            raise ValueError("wfn.nocc_{up,dn} does not match nocc parameter")

        # Initialize base class
        # ---------------------

        FanCI.__init__(self, ham, wfn, nproj, nparam, **kwargs)

        # Pre-compute data needed to compute the overlap (and its derivative) of the
        # "P" space and "S" space determinants
        # --------------------------------------------------------------------------

        # Get results of 'searchsorted(i)' from i=0 to i=nbasis for each det. in "S" space
        arange = np.arange(self._wfn.nbasis, dtype=pyci.c_int)
        sspace_data = [occs.searchsorted(arange) for occs in self._sspace]
        pspace_data = sspace_data[:self._nproj]

        # Save sub-class -specific attributes
        # -----------------------------------

        self._nmatrices = nmatrices
        self._numerator = numerator
        self._denominator = denominator
        self._matrix_mask = self._mask[:-1].reshape(nmatrices, ham.nbasis, nocc)
        self._sspace_data = sspace_data
        self._pspace_data = pspace_data

    def freeze_matrix(self, *matrices: Sequence[int]) -> None:
        r"""
        Set a matrix to be frozen during optimization.

        Parameters
        ----------
        matrices : Sequence[int]
            Indices of matrices to freeze.

        """
        for matrix in matrices:
            self._matrix_mask[matrix] = False
        # Update nactive
        self._nactive = self._mask.sum()

    def unfreeze_matrix(self, *matrices: Sequence[int]) -> None:
        r"""
        Set a matrix to be active during optimization.

        Parameters
        ----------
        matrices : Sequence[int]
            Indices of matrices to unfreeze.

        """
        for matrix in matrices:
            self._matrix_mask[matrix] = True
        # Update nactive
        self._nactive = self._mask.sum()

    def compute_overlap(self, x: np.ndarray, occs_array: Union[np.ndarray, str]) -> np.ndarray:
        r"""
        Compute the FanCI overlap vector.

        Parameters
        ----------
        x : np.ndarray
            Parameter array, [p_0, p_1, ..., p_n].
        occs_array : (np.ndarray | 'P' | 'S')
            Array of determinant occupations for which to compute overlap. A string "P" or "S" can
            be passed instead that indicates whether ``occs_array`` corresponds to the "P" space
            or "S" space, so that a more efficient, specialized computation can be done for these.

        Returns
        -------
        ovlp : np.ndarray
            Overlap array.

        """
        if isinstance(occs_array, np.ndarray):
            pass
        elif occs_array == 'P':
            occs_array = self._pspace
        elif occs_array == 'S':
            occs_array = self._sspace
        else:
            raise ValueError('invalid `occs_array` argument')

        # Reshape parameter array to numerator and denominator matrices
        x_mats = x.reshape(self._nmatrices, self._wfn.nbasis, self._wfn.nocc_up)
        n_mats = x_mats[:self._numerator]
        d_mats = x_mats[self._numerator:]

        # Compute overlaps of occupation vectors
        y = np.empty(occs_array.shape[0], dtype=pyci.c_double)
        for i, occs in enumerate(occs_array):
            y[i] = np.prod(np.linalg.det(n_mat[occs]) for n_mat in n_mats) \
                 / np.prod(np.linalg.det(d_mat[occs]) for d_mat in d_mats)
        return y

    def compute_overlap_deriv(self, x: np.ndarray, occs_array: Union[np.ndarray, str]) \
            -> np.ndarray:
        r"""
        Compute the FanCI overlap derivative matrix.

        Parameters
        ----------
        x : np.ndarray
            Parameter array, [p_0, p_1, ..., p_n].
        occs_array : (np.ndarray | 'P' | 'S')
            Array of determinant occupations for which to compute overlap. A string "P" or "S" can
            be passed instead that indicates whether ``occs_array`` corresponds to the "P" space
            or "S" space, so that a more efficient, specialized computation can be done for these.

        Returns
        -------
        ovlp : np.ndarray
            Overlap derivative array.

        """
        # Check if we can use our pre-computed {p,s}space_data
        if isinstance(occs_array, np.ndarray):
            # Get results of 'searchsorted(i)' from i=0 to i=nbasis for each det. in occs_array
            arange = np.arange(self._wfn.nbasis, dtype=pyci.c_int)
            pos_list = [occs.searchsorted(arange) for occs in occs_array]
        elif occs_array == 'P':
            occs_array = self._pspace
            pos_list = self._pspace_data
        elif occs_array == 'S':
            occs_array = self._sspace
            pos_list = self._sspace_data
        else:
            raise ValueError('invalid `occs_array` argument')

        # Reshape parameter array to numerator and denominator matrices
        x_mats = x.reshape(self._nmatrices, self._wfn.nbasis, self._wfn.nocc_up)
        n_mats = x_mats[:self._numerator]
        d_mats = x_mats[self._numerator:]

        # Shape of y is (no. determinants, no. active parameters excluding energy)
        y = np.zeros((occs_array.shape[0], self._nactive - self._mask[-1]), dtype=pyci.c_double)

        # Iterate over occupation vectors
        for y_row, occs, positions in zip(y, occs_array, pos_list):

            # Compute determinants of numerator and denominator matrices
            n_dets = [np.linalg.det(n_mat[occs]) for n_mat in n_mats]
            d_dets = [np.linalg.det(d_mat[occs]) for d_mat in d_mats]
            n_det_prod = np.prod(n_dets)
            d_det_prod = np.prod(d_dets)

            # Get results of 'searchsorted' of each {k,l} in occs
            k_positions = positions
            l_positions = positions[:self._wfn.nocc_up]

            # Iterate over all parameters (i), active parameters (j)
            i = -1
            j = -1

            # Iterate over numerator matrices
            for n_mat, n_det in zip(n_mats, n_dets):
                # Iterate over row indices
                k_sign = -1
                for k, k_pos in enumerate(k_positions):
                    k_sign *= -1
                    # Check if row is present
                    if k_pos != occs.size and occs[k_pos] == k:
                        k_slice = np.delete(occs, k_pos, axis=0)
                    else:
                        k_slice = np.empty(0)
                    # Iterate over column indices
                    l_sign = -1
                    for l, l_pos in enumerate(l_positions):
                        l_sign *= -1
                        # Check if parameter is active, advance (i, j)
                        i += 1
                        if not self._mask[i]:
                            continue
                        j += 1
                        # Check if any rows remain after deleting k_pos
                        # Check if column is present
                        if k_slice.size and l_pos != occs.size and occs[l_pos] == l:
                            # Compute derivative of overlap function
                            minor = np.delete(n_mat[k_slice], l, axis=1)
                            y_row[j] = k_sign * l_sign * n_det_prod * np.linalg.det(minor) \
                                     / (n_det * d_det_prod)

            # Iterate over denominator matrices
            for d_mat, d_det in zip(d_mats, d_dets):
                # Iterate over row indices
                k_sign = -1
                for k, k_pos in enumerate(k_positions):
                    k_sign *= -1
                    # Check if row is present
                    if k_pos != occs.size and occs[k_pos] == k:
                        k_slice = np.delete(occs, k_pos, axis=0)
                    else:
                        k_slice = np.empty(0)
                    # Iterate over column indices
                    l_sign = -1
                    for l, l_pos in enumerate(l_positions):
                        l_sign *= -1
                        # Check if parameter is active, advance (i, j)
                        i += 1
                        if not self._mask[i]:
                            continue
                        j += 1
                        # Check if any rows remain after deleting k_pos
                        # Check if column is present
                        if k_slice.size and l_pos != occs.size and occs[l_pos] == l:
                            # Compute derivative of overlap function
                            minor = np.delete(d_mat[k_slice], l, axis=1)
                            y_row[j] = -k_sign * l_sign * n_det_prod \
                                     / (np.linalg.det(minor) * d_det * d_det_prod)

        # Return overlap derivative matrix
        return y
