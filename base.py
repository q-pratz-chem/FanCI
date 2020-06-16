r"""
Refactor objective function

"""

import numpy as np

import abc


class FanCIBase(abc.ABC):
    r"""
    """

    def __init__(self, ham, *args, **kwargs):
        r"""
        Initialize the FanCI problem.

        """
        # Initialize system attributes
        self.nbasis = None
        self.nocc_up = None
        self.nocc_dn = None
        self.len_pspace = None
        self.len_sspace = None
        self.wfn = None

        # Save ham attribute
        self.ham = ham

        # Initialize system
        self.init_system(*args, **kwargs)

        # Initialize compute_overlap function
        self.compute_overlap = self.init_overlap(*args, **kwargs)

        # Initialize compute_overlap_deriv function
        self.compute_overlap_deriv = self.init_overlap_deriv(*args, **kwargs)

        # Initialize pyci matrix operator with dimensions (pspace, sspace)
        self.matrix_op = sparse_op(self.ham, self.wfn, self.len_pspace)

        # Initialize arrays of determinants
        self.det_array = self.wfn.to_det_array()
        self.pspace_dets = self.det_array[:self.len_pspace]
        self.sspace_dets = self.det_array[:self.len_sspace]

        # Initialize arrays of occupations
        self.occs_array = self.wfn.to_occs_array()
        self.pspace_occs = self.occs_array[:self.len_pspace]
        self.sspace_occs = self.occs_array[:self.len_sspace]

    @abc.abstractmethod
    def init_system(self, *args, **kwargs):
        r"""
        Initialize the system dimensions and CI wavefunction.

        """
        # Create your CI wfn here and return it
        #
        # Example:
        #     self.nbasis = ...
        #     self.nocc_up = ...
        #     self.nocc_dn = ...
        #     self.len_pspace = ...
        #     self.len_sspace = ...
        #     self.wfn = ...
        #
        raise NotImplementedError('this method must be overwritten in a sub-class')

    @abc.abstractmethod
    def init_overlap(self, *args, **kwargs):
        r"""
        Initialize the compute_overlap() function.

        """
        # Create your callable() here and return it
        #
        #     f : x[m], occs_array[n, :] -> y[n]
        #
        # Example:
        #     def f(x, occs_array):
        #         pass
        #     return f
        #
        raise NotImplementedError('this method must be overwritten in a sub-class')

    @abc.abstractmethod
    def init_overlap_deriv(self, *args, **kwargs):
        r"""
        Initialize the compute_overlap_deriv() function.

        """
        # Create your callable() here and return it
        #
        #     f : x[m], occs_array[n, :] -> y[m, n]
        #
        # Example:
        #     def f(x, occs_array):
        #         pass
        #     return f
        #
        raise NotImplementedError('this method must be overwritten in a sub-class')

    def compute_objective(self, x):
        r"""
        Compute the FanCI objective function.

        """
        # Compute overlaps ``c_n`` of determinants in sspace
        ovlp_vals = self.compute_overlap(x[:-1], self.sspace_occs)

        # Compute:
        #     \sum_{n} {<m|H|n> c_n - E \delta_{m,n} c_n}
        #
        # Note: x[-1] == Energy
        #
        f_vals = self.matrix_op.dot(ovlp_vals)
        f_vals -= x[-1] * ovlp_vals[:self.len_pspace]
        return f_vals

    def compute_jacobian(self, x):
        r"""
        Compute the FanCI Jacobian function.

        """
        # Compute overlaps in pspace:
        #     c_n
        ovlp_vals = self.compute_overlap(x[:-1], self.pspace_occs)

        # Compute overlap derivatives in sspace:
        #     d(c_m)/d(p_k)
        d_ovlp_vals = self.compute_overlap_deriv(x[:-1], self.sspace_occs)

        # Allocate jacobian matrix
        jac_vals = np.empty((self.len_pspace, x.size), dtype=x.dtype)

        # Compute rows of jac:
        #     d(<m|H|\Psi>)/dp_k - E d(<m|\Psi>)/dp_k - (dE/dp_k) <m|\Psi>
        #
        # Note:
        #     x[-1] == Energy
        #
        # And we use:
        #     E <m|\Psi>/dp_k = E \sum_n {\delta_{m,n} dc_n/dp_k}
        #
        # TODO: Not sure about the math here
        # Iterate over rows of jac_vals and d_ovlp_vals
        for jac_row, d_ovlp_row in zip(jac_vals[:, :-1], d_ovlp_vals):
            jac_row[:] = self.matrix_op.dot(d_ovlp_row)
            jac_row[:] -= x[-1] * d_ovlp_row[:self.len_pspace]

        # TODO: Not sure about the math here
        # Compute final column of jac_vals:
        #     (E/dp_k) <m|\Psi> = (E / dp_k) \sum_n {\delta_{m,n} c_n}
        jac_vals[:, -1] = -x[-1] * ovlp_vals
        return jac_vals

    def solve_fanci(self):
        r"""
        Solve the FanCI problem.

        """
        raise NotImplementedError('implement this please')
