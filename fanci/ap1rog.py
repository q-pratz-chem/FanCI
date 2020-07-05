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
        # Parse arguments
        # ---------------

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
        # ---------------------

        FanCI.__init__(self, ham, wfn, nproj, nparam, **kwargs)

        # Pre-compute data needed to compute the overlap (and its derivative) of the
        # "P" space and "S" space determinants
        # --------------------------------------------------------------------------

        # Assign reference occupations
        ref_occs = np.arange(nocc, dtype=pyci.c_int)

        # Use set differences to get hole/particle indices
        holes_list = [
                np.setdiff1d(ref_occs, occs, assume_unique=1)
                for occs in self._sspace
                ]
        parts_list = [
                np.setdiff1d(occs, ref_occs, assume_unique=1) - self._wfn.nvir_up
                for occs in self._sspace
                ]

        # Get results of 'searchsorted(i)' from i=0 to i=nbasis for each holes/particles
        arange = np.arange(self._wfn.nbasis, dtype=pyci.c_int)
        holes_pos_list = [holes.searchsorted(arange) for holes in holes_list]
        parts_pos_list = [parts.searchsorted(arange) for parts in parts_list]

        # Save sub-class -specific attributes
        # -----------------------------------

        self._ref_occs = ref_occs
        self._sspace_exc_data = holes_list, parts_list
        self._pspace_exc_data = holes_list[:nproj], parts_list[:nproj]
        self._sspace_pos_data = holes_pos_list, parts_pos_list
        self._pspace_pos_data = holes_pos_list[:nproj], parts_pos_list[:nproj]

    def compute_overlap(self, x: np.ndarray, occs_array: np.ndarray, mode: str = None) \
            -> np.ndarray:
        r"""
        Compute the FanCI overlap vector.

        Parameters
        ----------
        x : np.ndarray
            Parameter array, [p_0, p_1, ..., p_n].
        occs_array : np.ndarray
            Array of determinant occupations for which to compute overlap.
        mode : ('P' | 'S'), optional
            Optional flag that indicates whether ``occs_array`` corresponds to the "P" space
            or "S" space, so that a more efficient, specialized computation can be done for these.

        Returns
        -------
        ovlp : np.ndarray
            Overlap array.

        """
        # Check if we can use our precomputed {p,s}space_exc_data
        if mode == 'P':
            holes_list, parts_list = self._pspace_exc_data
        elif mode == 'S':
            holes_list, parts_list = self._sspace_exc_data
        else:
            # Use set differences to get hole/particle indices
            holes_list = [
                    np.setdiff1d(self._ref_occs, occs, assume_unique=1)
                    for occs in occs_array
                    ]
            parts_list = [
                    np.setdiff1d(occs, self._ref_occs, assume_unique=1) - self._wfn.nocc_up
                    for occs in occs_array
                    ]

        # Reshape parameter array to AP1roG matrix
        x_mat = x.reshape(self._wfn.nocc_up, self._wfn.nvir_up)

        # Compute overlaps of occupation vectors
        y = np.empty(occs_array.shape[0], dtype=pyci.c_double)
        for i, (occs, holes, parts) in enumerate(zip(occs_array, holes_list, parts_list)):
            # Overlap is equal to one for the reference determinant
            y[i] = permanent(x_mat[holes][:, parts]) if holes.size else 1
        return y

    def compute_overlap_deriv(self, x: np.ndarray, occs_array: np.ndarray, mode: str = None) \
            -> np.ndarray:
        r"""
        Compute the FanCI overlap derivative matrix.

        Parameters
        ----------
        x : np.ndarray
            Parameter array, [p_0, p_1, ..., p_n].
        occs_array : np.ndarray
            Array of determinant occupations for which to compute overlap derivative.
        mode : ('P' | 'S'), optional
            Optional flag that indicates whether ``occs_array`` corresponds to the "P" space
            or "S" space, so that a more efficient, specialized computation can be done for these.

        Returns
        -------
        ovlp : np.ndarray
            Overlap derivative array.

        """
        # Check if we can use our precomputed {p,s}space_{exc,pos}_data
        if mode == 'P':
            holes_list, parts_list = self._pspace_exc_data
            holes_pos_list, parts_pos_list = self._pspace_pos_data
        elif mode == 'S':
            holes_list, parts_list = self._sspace_exc_data
            holes_pos_list, parts_pos_list = self._sspace_pos_data
        else:
            # Use set differences to get hole/particle indices
            holes_list = [
                    np.setdiff1d(self._ref_occs, occs, assume_unique=1)
                    for occs in occs_array
                    ]
            parts_list = [
                    np.setdiff1d(occs, self._ref_occs, assume_unique=1) - self._wfn.nocc_up
                    for occs in occs_array
                    ]
            # Get results of 'searchsorted(i)' from i=0 to i=nbasis for each holes/particles
            arange = np.arange(self._wfn.nbasis, dtype=pyci.c_int)
            holes_pos_list = [holes.searchsorted(arange) for holes in holes_list]
            parts_pos_list = [parts.searchsorted(arange) for parts in parts_list]

        # Reshape parameter array to AP1roG matrix
        x_mat = x.reshape(self._wfn.nocc_up, self._wfn.nvir_up)

        # Shape of y is (no. determinants, no. active parameters excluding energy)
        y = np.zeros((occs_array.shape[0], self._nactive - self._mask[-1]), dtype=pyci.c_double)

        # Iterate over occupation vectors
        iterator = zip(y, occs, holes_list, parts_list, holes_pos_list, parts_pos_list)
        for y_row, occs, holes, parts, holes_pos, parts_pos in iterator:

            # Check for reference determinant; d(<\psi|\Psi>)/dp == 0
            if not holes.size:
                continue

            # Get results of 'searchsorted' of each {k,l} in holes,particles
            k_positions = holes_pos[:self._wfn.nocc_up]
            l_positions = parts_pos[:self._wfn.nvir_up]

            # Iterate over all parameters (i), active parameters (j)
            i = -1
            j = -1

            # Iterate over row indices
            for k, k_pos in enumerate(k_positions):
                # Check if row is present
                if k_pos != holes.size and holes[k_pos] == k:
                    k_slice = np.delete(holes, k_pos, axis=0)
                else:
                    k_slice = np.empty(0)
                # Iterate over column indices
                for l, l_pos in enumerate(l_positions):
                    # Check if element is active, advance (i, j)
                    i += 1
                    if not self._mask[i]:
                        continue
                    j += 1
                    # Check if any rows remain after deleting k_pos
                    # Check if column is present
                    if k_slice.size and l_pos != parts.size and parts[l_pos] == l:
                        # Compute permanent of (k, l) minor
                        l_slice = np.delete(parts, l_pos, axis=0)
                        y_row[j] = permanent(x_mat[k_slice][:, l_slice])

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
