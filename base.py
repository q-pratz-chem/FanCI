r"""
FanCI base class module.

"""

import abc

import numpy as np

from scipy.optimize import root

from pyci import sparse_op


__all__ = [
    'BaseFanCI',
    'BaseOverlap',
    ]


class BaseFanCI(abc.ABC):
    r"""
    FanCI problem base class.

    """

    def __init__(self, *args, **kwargs):
        r"""
        Initialize the FanCI problem.

        """
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
        self.pspace_dets = self.det_array[:self.ndet_pspace]
        self.sspace_dets = self.det_array[:self.ndet_sspace]
        # Initialize arrays of occupations
        self.occs_array = self.wfn.to_occs_array()
        self.pspace_occs = self.occs_array[:self.ndet_pspace]
        self.sspace_occs = self.occs_array[:self.ndet_sspace]

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
        raise NotImplementedError('this method must be overwritten in a sub-class')

    @abc.abstractmethod
    def init_overlap(self, *args, **kwargs):
        r"""
        Initialize the FanCI overlap operator.

        """
        # Create your object here and return it
        #
        #     obj.overlap:
        #
        #       f : x[m], occs_array[n, :] -> y[n]
        #
        #     obj.overlap_deriv:
        #
        #       j : x[m], occs_array[n, :] -> y[m, n]
        #
        # Example:
        #
        #     return ExampleOverlap(*args, **kwargs)
        #
        raise NotImplementedError('this method must be overwritten in a sub-class')

    def compute_objective(self, x):
        r"""
        Compute the FanCI objective function.

        """
        # Compute overlaps ``c_n`` of determinants in sspace
        #
        ovlp_vals = self.fanci_op.overlap(x[:-1], self.sspace_occs)

        # Compute:
        #
        #     \sum_{n} {<m|H|n> c_n - E \delta_{m,n} c_n}
        #
        energy = x[-1]
        f_vals = self.ci_op.dot(ovlp_vals)
        f_vals -= energy * ovlp_vals[:self.ndet_pspace]
        return f_vals

    def compute_jacobian(self, x):
        r"""
        Compute the FanCI Jacobian function.

        """
        # Allocate jacobian matrix
        #
        jac_vals = np.empty((self.ndet_pspace, x.size), dtype=x.dtype)

        # Assign Energy = x[-1]
        #
        energy = x[-1]

        # Compute overlaps in pspace:
        #
        #     c_n
        #
        ovlp_vals = self.fanci_op.overlap(x[:-1], self.pspace_occs)

        # Compute overlap derivatives in sspace:
        #
        #     d(c_m)/d(p_k)
        #
        d_ovlp_vals = self.fanci_op.overlap_deriv(x[:-1], self.sspace_occs)

        # Iterate over rows of jac_vals (excluding final column) and d_ovlp_vals
        #
        for jac_row, d_ovlp_row in zip(jac_vals[:, :-1], d_ovlp_vals):
            #
            # Compute rows of jac:
            #
            #     d(<m|H|\Psi>)/dp_k - E d(<m|\Psi>)/dp_k - (dE/dp_k) <m|\Psi>
            #
            #     E <m|\Psi>/dp_k = E \sum_n {\delta_{m,n} (dc_n/dp_k)}
            #
            self.ci_op.dot(d_ovlp_row, out=jac_row)
            jac_row -= energy * d_ovlp_row[:self.ndet_pspace]

        # Compute final column of jac_vals:
        #
        #     (dE/dp_k) <m|\Psi> = (dE/dp_k) \sum_n {\delta_{m,n} c_n}
        #
        jac_vals[:, -1] = ovlp_vals
        jac_vals[:, -1] *= -1
        return jac_vals

    def solve_fanci(self, x0, *args, **kwargs):
        r"""
        Solve the FanCI problem.

        """
        return root(self.compute_objective, x0, jac=self.compute_jacobian, *args, **kwargs)


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
        # f : x[m], occs_array[n, :] -> y[n]
        #
        raise NotImplementedError('this method must be overwritten in a sub-class')

    @abc.abstractmethod
    def overlap_deriv(self, *args, **kwargs):
        r"""
        Compute the overlap derivative matrix.

        """
        #
        # j : x[m], occs_array[n, :] -> y[m, n]
        #
        raise NotImplementedError('this method must be overwritten in a sub-class')
