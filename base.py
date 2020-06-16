"""Refactor objective function"""

import numpy as np

import abc

from pyci import doci


class FanCIBase(abc.ABC):
    r"""
    """

    def __init__(self, ham, nocc, len_pspace):
        r"""
        """
        # Save ham, system dimensions as attributes
        self.ham = ham
        # For now, we are only concerned with DOCI, but we keep the base class general:
        self.nbasis = ham.nbasis
        self.nocc_up = nocc
        self.nocc_dn = nocc
        # Initialize pyci wavefunction
        self.wfn = self.init_wavefunction()
        # Define fanCI spaces "P" and "S"
        self.len_pspace = len_pspace
        self.len_sspace = wfn.ndet
        # Get array of determinant bitstrings
        self.det_array = wfn.to_det_array()
        # Get array of occupied indices for each determinant
        self.occs_array = wfn.to_occs_array()
        # Define array of dets/occs for fanCI space "P"
        self.pspace_dets = self.det_array[: self.len_pspace]
        self.pspace_occs = self.occs_array[: self.len_pspace]
        # Define array of dets/occs for fanCI space "S"
        self.sspace_dets = self.det_array[: self.len_sspace]
        self.sspace_occs = self.occs_array[: self.len_sspace]
        # Initialize compute_overlap function
        self.compute_overlap = self.init_overlap()
        # Initialize pyci matrix operator with dimensions (pspace, sspace)
        # Note: sparse_op works with both fullci and doci wfns, so the below is fine:
        self.ci_matrix_op = doci.sparse_op(self.ham, self.wfn, self.len_pspace)
        # Although we should move `sparse_op` out of the doci submodule...

    @abc.abstractmethod
    def init_wavefunction(self):
        r"""
        """
        # Create your CI wfn here and return it
        #
        raise NotImplementedError

    @abc.abstractmethod
    def init_overlap(self):
        r"""
        """
        # Create your callable() here and return it
        # def overlap(x):
        #     r"""
        #     """
        #     pass
        # return overlap
        raise NotImplementedError

    def compute_objective(self, x):
        r"""
        """
        # Compute overlaps of determinants in sspace
        ovlp_vals = self.compute_overlap(x[:-1], self.sspace_occs)
        # Compute:
        #     \sum_{n} {<m|H|n> c_n - E \delta_{m,n} c_n}
        # Note: x[-1] == Energy
        f_vals = self.ci_matrix_op.dot(ovlp_vals)
        f_vals -= x[-1] * ovlp_vals[: self.len_pspace]
        return f_vals

    def compute_jacobian(self, x):
        r"""
        """
        # This method doesn't change, keep here in the base class
        #
        # d(<m|H|\Psi>)/dp_k = \sum_n {<m|H|n> dc_n/dp_k}
        # Taken from FanPy
        size_wfnprms = len(x[:-1])
        # Jacobian equation
        # d(<m|H|\Psi>)/dp_k - E d(<m|\Psi>)/dp_k - (dE/dp_k) <m|\Psi>
        jac = np.empty((self.len_pspace, len(x)))
        for idx in range(size_wfnprms):
            # compute_overlap function should also return
            # the overlap derivatives
            dovlp_vals = self.compute_overlap(x[:-1], self.sspace_occs, deriv=idx)
            jac[:, idx] = self.ci_matrix_op.dot(dovlp_vals)
            # E <m|\Psi>/dp_k = E \sum_n {\delta_{m,n} dc_n/dp_k}
            jac[:, idx] -= (
                x[-1] * dovlp_vals[: self.len_pspace]
            )  # alter: self.compute_overlap(self.pspace_occs, deriv=idx)
        # (E/dp_k) <m|\Psi> = (E / dp_k) \sum_n {\delta_{m,n} c_n}
        # Compute overlaps of determinants in pspace
        ovlp_vals = self.compute_overlap(self.pspace_occs)
        jac[:, -1] = -ovlp_vals.T
        return jac
