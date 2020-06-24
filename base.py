r"""
FanCI base class module.

"""

import abc

import numpy as np

from scipy.optimize import root

from pyci import sparse_op


__all__ = [
    "BaseFanCI",
    "BaseOverlap",
]


class BaseFanCI(abc.ABC):
    r"""
    FanCI problem base class.

    """

    def __init__(self, norm_det=-1, norm_param=-1, *args, **kwargs):
        r"""
        Initialize the FanCI problem.

        """
        if norm_det != -1 and norm_param != -1:
            raise ValueError("Only one normalization constraint can be set at a time.")
        self.norm_param = norm_param
        self.norm_det = norm_det
        # Initialize system attributes
        self.nbasis = None
        self.nocc_up = None
        self.nocc_dn = None
        self.ndet_pspace = None
        self.ndet_sspace = None
        self.ham = None
        self.wfn = None

        # Initialize system
        self.init_system(*args, **kwargs)

        # Initialize arrays of determinants
        self.det_array = self.wfn.to_det_array()
        self.pspace_dets = self.det_array[: self.ndet_pspace]
        self.sspace_dets = self.det_array[: self.ndet_sspace]
        # Initialize arrays of occupations
        self.occs_array = self.wfn.to_occs_array()
        self.pspace_occs = self.occs_array[: self.ndet_pspace]
        self.sspace_occs = self.occs_array[: self.ndet_sspace]

        # Initialize FanCI overlap and overlap derivative operator
        self.fanci_op = self.init_overlap(*args, **kwargs)

        # Compute PyCI CI matrix operator with dimensions (pspace, sspace)
        self.ci_op = sparse_op(self.ham, self.wfn, self.ndet_pspace)

    @abc.abstractmethod
    def init_system(self, *args, **kwargs):
        r"""
        Initialize the system dimensions and CI wavefunction.

        """
        # Create your CI wfn here and return it
        #
        # Example:
        #
        #     self.ham = ...
        #     self.wfn = ...
        #     self.nbasis = ...
        #     self.nocc_up = ...
        #     self.nocc_dn = ...
        #     self.ndet_pspace = ...
        #     self.ndet_sspace = ...
        #
        raise NotImplementedError("this method must be overwritten in a sub-class")

    @abc.abstractmethod
    def init_overlap(self, *args, **kwargs):
        r"""
        Initialize the FanCI overlap operator.

        """
        # Create your object here and return it
        #
        #     obj.overlap:
        #
        #       f : x[k], occs_array[m, :] -> y[m]
        #
        #     obj.overlap_deriv:
        #
        #       j : x[k], occs_array[m, :] -> y[m, k]
        #
        # Example:
        #
        #     return ExampleOverlap(*args, **kwargs)
        #
        raise NotImplementedError("this method must be overwritten in a sub-class")

    def compute_norm_det(self, wfn_params, norm_det):
        ref_sd = self.occs_array[norm_det]
        ref_sd_ovlp = self.fanci_op.overlap(wfn_params, [ref_sd])
        # <\Psi|r><r|\Psi>
        return ref_sd_ovlp ** 2

    def compute_norm_det_deriv(self, wfn_params, norm_det):
        ref_sd = self.occs_array[norm_det]
        ref_sd_ovlp = self.fanci_op.overlap(wfn_params, [ref_sd])
        ref_sd_dovlp = self.fanci_op.overlap_deriv(wfn_params, [ref_sd])
        # d(<\Psi|r><r|\Psi>)/d(p_k) = d(<\Psi|r>)/d(p_k) * <r|\Psi>
        #                            + <r|\Psi> * d(<\Psi|r>)/d(p_k)
        return 2 * ref_sd_dovlp * ref_sd_ovlp[0]

    def compute_norm_param(self, norm_param):
        pass

    def compute_norm_param_deriv(self):
        pass

    def compute_objective(self, x):
        r"""
        Compute the FanCI objective function.

        """
        # Compute overlaps of determinants in sspace:
        #
        #   c_m
        #
        ovlp = self.fanci_op.overlap(x[:-1], self.sspace_occs)

        # Compute objective function:
        #
        #   f_n = <n|H|\Psi> - E <n|\Psi>
        #
        #       = <m|H|n> c_m - E \delta_{mn} c_m
        #
        energy = x[-1]
        f = self.ci_op.dot(ovlp)
        f -= energy * ovlp[: self.ndet_pspace]
        # Add intermediate normalization constraint
        #   f_n+1 = <\Psi|r><r|\Psi> - 1
        if (self.norm_det != -1) or (self.norm_param != -1):
            if self.norm_det != -1:
                norm = self.compute_norm_det(x[:-1], self.norm_det)
                np.hstack(f, norm - 1)
            else:
                raise NotImplementedError
        return f

    def compute_jacobian(self, x):
        r"""
        Compute the FanCI Jacobian function.

        """
        # Allocate Jacobian matrix transpose:
        #
        #   {J^T}_{kn}
        #
        jac_t = np.empty((x.size, self.ndet_pspace), dtype=x.dtype)

        # Assign Energy = x[-1]
        #
        energy = x[-1]

        # Compute overlaps in pspace:
        #
        #   c_n
        #
        ovlp = self.fanci_op.overlap(x[:-1], self.pspace_occs)

        # Compute overlap derivatives in sspace:
        #
        #   d(c_m)/d(p_k)
        #
        d_ovlp = self.fanci_op.overlap_deriv(x[:-1], self.sspace_occs)

        # Compute Jacobian:
        #
        #   J_{nk} = d(<n|H|\Psi>)/d(p_k) - E d(<n|\Psi>)/d(p_k) - dE/d(p_k) <n|\Psi>
        #
        # Iterate over columns of Jacobian (excluding final column) and rows of d_ovlp
        #
        for jac_col, d_ovlp_row in zip(jac_t[:-1], d_ovlp):
            #
            # Compute each column of the Jacobian:
            #
            #   d(<n|H|\Psi>)/d(p_k) = <m|H|n> d(c_m)/d(p_k)
            #
            #   E d(<n|\Psi>)/d(p_k) = E \delta_{nk} d(c_n)/d(p_k)
            #
            self.ci_op.dot(d_ovlp_row, out=jac_col)
            jac_col -= energy * d_ovlp_row[: self.ndet_pspace]

        # Compute final column of the Jacobian:
        #
        #   dE/d(p_k) <n|\Psi> = dE/d(p_k) \delta_{nk} c_n
        #
        jac_t[-1, :] = ovlp
        jac_t[-1, :] *= -1

        # Add intermediate normalization constraint derivative
        #   d(f_n+1)/d(p_k)
        if (self.norm_det != -1) or (self.norm_param != -1):
            if self.norm_det != -1:
                d_norm = self.compute_norm_det_deriv(x[:-1], self.norm_det)
                np.vstack(jac_t, d_norm)
            else:
                raise NotImplementedError

        # Return the Jacobian proper, i.e., from the allocated Jacobian transpose matrix:
        #
        #   J_{nk} = {(J^T)^T}_{nk}
        #
        return jac_t.transpose()

    # def constraints(self, x0):
    #     if self.normalization_det =! -1:
    #         self.ref_sd = [get_ref_sd_from_wfn]
    #         ovlp_rdet = self.fanci_op.overlap(x0[:-1], self.ref_sd)
    #         return (ovlp_rdet**2 -1)

    def solve_fanci(self, x0, *args, **kwargs):
        r"""
        Solve the FanCI problem.

        """
        return root(
            self.compute_objective, x0, jac=self.compute_jacobian, *args, **kwargs
        )


class BaseOverlap(abc.ABC):
    r"""
    FanCI overlap base class.

    """

    @abc.abstractmethod
    def overlap(self, *args, **kwargs):
        r"""
        Compute the overlap vector.

        """
        #
        # f : x[k], occs_array[m, :] -> y[m]
        #
        raise NotImplementedError("this method must be overwritten in a sub-class")

    @abc.abstractmethod
    def overlap_deriv(self, *args, **kwargs):
        r"""
        Compute the overlap derivative matrix.

        """
        #
        # j : x[k], occs_array[m, :] -> y[m, k]
        #
        raise NotImplementedError("this method must be overwritten in a sub-class")

