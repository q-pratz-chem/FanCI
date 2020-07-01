r"""
FanCI AP1roG module.

"""

from itertools import permutations

from typing import Any

import numpy as np

import pyci

from .fanci import FanCI


__all___ = [
    'AP1roG',
    ]


class AP1roG(FanCI):
    r"""
    AP1roG FanCI class.

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
        nparam = nocc * (ham.nbasis - nocc) + 1

        # Handle default nproj
        nproj = nparam if nproj is None else nproj

        # Handle default wfn (P space == single pair excitations)
        if wfn is None:
            wfn = pyci.doci_wfn(ham.nbasis, nocc)
            wfn.add_excited_dets(1)
        elif not isinstance(wfn, pyci.doci_wfn):
            raise TypeError('wfn must be a `pyci.doci_wfn`')
        elif wfn.nocc_up != nocc or wfn.nocc_dn != nocc:
            raise ValueError('wfn.nocc_{up,dn} does not match nocc parameter')

        # Initialize base class
        FanCI.__init__(self, ham, wfn, nproj, nparam, **kwargs)

        # Assign reference occupations for compute_overlap{,_deriv}
        self._ref_occs = np.arange(nocc, dtype=pyci.c_int)

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
        # Reshape parameter array to AP1roG matrix
        x_mat = x.reshape(self._wfn.nocc_up, self._wfn.nvir_up)

        # Compute overlaps of occupation vectors
        y = np.empty(occs_array.shape[0], dtype=pyci.c_double)
        for i, occs in enumerate(occs_array):
            # Find hole indices
            holes = np.setdiff1d(self._ref_occs, occs, assume_unique=True)
            if holes.size:
                # Excited determinant; find particle indices
                particles = np.setdiff1d(occs, self._ref_occs, assume_unique=True)
                particles -= self._wfn.nocc_up
                y[i] = permanent(x_mat[holes][:, particles])
            else:
                # Reference determinant; <\psi|\Psi> == 1
                y[i] = 1
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
        # Reshape parameter array to AP1roG matrix
        x_mat = x.reshape(self._wfn.nocc_up, self._wfn.nvir_up)

        # Shape of y is (no. determinants, no. active parameters excluding energy)
        y = np.zeros((occs_array.shape[0], self._nactive - self._mask[-1]), dtype=pyci.c_double)

        # Iterate over occupation vectors
        for y_row, occs in zip(y, occs_array):
            # Find hole indices
            holes = np.setdiff1d(self._ref_occs, occs, assume_unique=True)
            # Check for reference determinant; d(<\psi|\Psi>)/dp == 0
            if not holes.size:
                continue
            # Find particle indices
            particles = np.setdiff1d(occs, self._ref_occs, assume_unique=True)
            particles -= self._wfn.nocc_up
            # Iterate over all parameters (i), active parameters (j)
            i = 0
            j = 0
            for k in range(self._wfn.nocc_up):
                k_pos = holes.searchsorted(k)
                for l in range(self._wfn.nvir_up):
                    # Check if element is active
                    if self._mask[i]:
                        # Check if row is present
                        if k_pos != holes.size and holes[k_pos] == k:
                            k_slice = np.delete(holes, k_pos, axis=0)
                            # Check if any rows remain after deleting k_pos
                            if k_slice.size:
                                # Check if column is present
                                l_pos = particles.searchsorted(l)
                                if l_pos != particles.size and particles[l_pos] == l:
                                    # Compute permanent of (k, l) minor
                                    l_slice = np.delete(particles, l_pos, axis=0)
                                    y_row[j] = permanent(x_mat[k_slice][:, l_slice])
                        # Go to next active parameter
                        j += 1
                    # Go to next parameter
                    i += 1

        # Return overlap derivative matrix
        return y


def permanent(matrix : np.ndarray) -> float:
    r"""
    Compute the permanent of a square matrix.

    Parameters
    ----------
    matrix : np.ndarray
        Square matrix.

    Returns
    -------
    result : matrix.dtype
        Permanent of the matrix.

    """
    rows = np.arange(matrix.shape[0])
    return sum(np.prod(matrix[rows, cols]) for cols in permutations(rows))
