r"""
FanCI base class module.

"""

import abc

import numpy as np

import pyci

from scipy.optimize import root, least_squares


__all__ = [
    'BaseFanCI',
    ]


class BaseFanCI(abc.ABC):
    r"""
    FanCI problem base class.

    Attributes
    ----------
    ham : pyci.hamiltonian
        PyCI Hamiltonian.
    wfn : (pyci.doci_wfn | pyci.fullci_wfn | pyci.genci_wfn)
        PyCI wave function.
    ci_op : pyci.sparse_op
        PyCI sparse CI operator.
    pspace : np.ndarray
        Array of determinant occupations in projection ("P") space.
    sspace : np.ndarray
        Array of determinant occupations in auxiliiary ("S") space.
    nequation : int
        Number of nonlinear equations.
    ndet : int
        Number of determinants in projection ("P") space.
    nparam : int
        Number of parameters for this FanCI problem.
    nparam_active : int
        Number of active parameters for this FanCI problem.
    nbasis : int
        Number of orbital basis functions.
    nocc_up : int
        Number of spin-up occupied orbitals.
    nocc_dn : int
        Number of spin-down occupied orbitals.
    nvir_up : int
        Number of spin-up virtual orbitals.
    nvir_dn : int
        Number of spin-down virtual orbitals.

    Methods
    -------
    optimize : np.ndarray, *args, **kwargs -> scipy.optimize.OptimizeResult
        Optimize the wave function parameters.
    compute_overlap : np.ndarray, np.ndarray -> np.ndarray
        Compute the overlap vector.
    compute_overlap_deriv : np.ndarray, np.ndarray -> np.ndarray
        Compute the overlap derivative matrix.
    compute_objective : np.ndarray -> np.ndarray
        Compute the objective function for the optimizer.
    compute_jacobian : np.ndarray -> np.ndarray
        Compute the Jacobian function for the optimizer.

    """

    def __init__(self, ham, wfn, ndet, nparam, norm_det=None, norm_param=None, constraints=None,
        mask=None, pspace_hf=True):
        r"""
        Initialize the FanCI problem.

        Parameters
        ----------
        ham : pyci.hamiltonian
            PyCI Hamiltonian.
        wfn : (pyci.doci_wfn | pyci.fullci_wfn | pyci.genci_wfn)
            PyCI wave function.
        ndet : int
            Number of determinants in projection ("P") space.
        nparam : int
            Number of parameters for this FanCI problem.
        norm_det : list(int, float)
            Indices of determinant whose overlaps to constrain, and the value to which to constrain.
        norm_param : list(int, float)
            Indices of parameters whose values to constrain, and the value to which to constrain.
        constraints : list(function, function), optional
            List of (f, dfdx) functions corresponding to additional constraints.
        mask : list(int | bool), optional
            List of parameters to freeze. If the list contains ints, then each element corresponds
            to a frozen parameter. If the list contains bools, then each element indicates whether
            that parameter is active (True) or frozen (False).
        pspace_hf : bool, optional
            Whether to add Hartree-Fock determinant to projection ("P") space.

        """
        # Parse arguments
        # ---------------

        # Generate constraints list
        norm_det = list() if norm_det is None else norm_det
        norm_param = list() if norm_param is None else norm_param
        constraints = list() if constraints is None else constraints.copy()
        for index, value in norm_det:
            constraints.append(self._norm_det_constraint(index, value))
        for index, value in norm_param:
            constraints.append(self._norm_param_constraint(index, value))

        # Compute number of nonlinear equations
        nequation = ndet + len(constraints)

        # Generate mask (True for active parameter, False for frozen parameter)
        if mask is None:
            mask = np.ones(nparam, dtype=np.bool)
        else:
            mask = np.array(mask)
            # Boolean mask
            if mask.dtype == np.bool:
                if mask.size != nparam:
                    raise ValueError('boolean mask must have length nparam')
            # Convert integer mask to boolean
            else:
                ints = mask
                mask = np.ones(nparam, dtype=np.bool)
                mask[ints] = 0

        # Compute number of active parameters
        nparam_active = np.count_nonzero(mask)

        # Generate system
        # ---------------

        # Check system dimensions
        if ham.nbasis != wfn.nbasis:
            raise ValueError('ham and wfn nbasis do not match')
        elif nequation < nparam_active:
            raise ValueError('number of equations < number of active parameters')

        # Handle wfn parameter; decide values for generating determinants from its type
        wfn = wfn.copy()
        if isinstance(wfn, pyci.doci_wfn):
            e_max = min(wfn.nocc_up, wfn.nvir_up)
            noccs = (wfn.nocc_up,)
            connections = (1,)
        elif isinstance(wfn, pyci.fullci_wfn):
            e_max = min(wfn.nocc, wfn.nvir)
            noccs = wfn.nocc_up, wfn.nocc_dn
            connections = (1, 2)
        elif isinstance(wfn, pyci.genci_wfn):
            e_max = min(wfn.nocc, wfn.nvir)
            noccs = (wfn.nocc_up,)
            connections = (1, 2)
        else:
            raise TypeError('invalid wfn type')

        # Fill wfn with P space determinants in excitation order until len(wfn) >= ndet
        for nexc in range(int(not pspace_hf), e_max + 1):
            if len(wfn) >= ndet:
                break
            wfn.add_excited_dets(nexc)
        if len(wfn) < ndet:
            raise ValueError('cannot generate enough determinants')

        # Compute P space determinants
        pspace_dets = wfn.to_det_array(ndet)

        # Truncate wave function if we generated > ndet determinants
        if len(wfn) > ndet:
            wfn = wfn.from_det_array(wfn.nbasis, *noccs, pspace_dets)

        # Fill wfn with S space determinants
        for det in pspace_dets:
            wfn.add_excited_dets(*connections, det=det)

        # Compute arrays of occupations
        sspace = wfn.to_occs_array()
        pspace = sspace[:ndet]

        # Compute CI matrix operator with ndet rows and len(wfn) columns
        ci_op = pyci.sparse_op(ham, wfn, ndet)

        # Assign public attributes to instance
        self.ham = ham
        self.wfn = wfn
        self.ci_op = ci_op
        self.pspace = pspace
        self.sspace = sspace
        self.nequation = nequation
        self.ndet = ndet
        self.nparam = nparam
        self.nparam_active = nparam_active
        self.nbasis = wfn.nbasis
        self.nocc_up = wfn.nocc_up
        self.nocc_dn = wfn.nocc_dn
        self.nvir_up = wfn.nvir_up
        self.nvir_dn = wfn.nvir_dn

        # Assign private attributes to instance
        self._constraints = constraints
        self._mask = mask

    def optimize(self, x0, mode='lstsq', use_jac=False, **kwargs):
        r"""
        Optimize the wave function parameters.

        Parameters
        ----------
        x0 : np.ndarray
            Initial guess for wave function parameters.
        mode : ('lstsq' | 'root'), default='lstsq'
            Solver mode.
        use_jac : bool, default=False
            Whether to use the Jacobian function or a finite-difference approximation.
        kwargs : dict, optional
            Additional keyword arguments to pass to optimizer.

        Returns
        -------
        result : scipy.optimize.OptimizeResult
            Result of optimization.

        """
        # Check x0 vector length
        if x0.size != self.nparam:
            raise ValueError('length of x0 does not match nparam')

        # Prepare objective, Jacobian, x0
        if self.nparam_active < self.nparam:
            # Generate masked versions with frozen parameters
            f, j, x0 = self._apply_mask(x0)
        else:
            # Use bare functions and copy of x0
            f = self.compute_objective
            j = self.compute_jacobian
            x0 = np.copy(x0)

        # Set up initial arguments to optimizer
        opt_args = (f,)
        opt_kwargs = kwargs.copy()
        if use_jac:
            opt_kwargs['jac'] = j

        # Parse mode parameter; choose optimizer and fix arguments
        if mode == 'lstsq':
            optimizer = least_squares
        elif mode == 'root':
            if self.nequation > self.nparam_active:
                raise ValueError('\'root\' mode does not work with over-determined system')
            optimizer = root
        elif mode == 'cma':
            raise NotImplementedError
        else:
            raise ValueError('invalid mode parameter')

        # Run optimizer
        return optimizer(*opt_args, **opt_kwargs)

    @abc.abstractmethod
    def compute_overlap(self, x, occs_array):
        r"""
        Compute the FanCI overlap vector.

            f : x[k], occs_array[m, :] -> y[m]

        """
        raise NotImplementedError('this method must be overwritten in a sub-class')

    @abc.abstractmethod
    def compute_overlap_deriv(self, x, occs_array):
        r"""
        Compute the FanCI overlap derivative matrix.

            j : x[k], occs_array[m, :] -> y[m, k]

        """
        raise NotImplementedError('this method must be overwritten in a sub-class')

    def compute_objective(self, x):
        r"""
        Compute the FanCI objective function.

            f : x[k] -> y[n]

        """
        # Allocate objective vector
        f = np.empty(self.nequation, dtype=x.dtype)
        f_dets = f[:self.ndet]
        f_cons = f[self.ndet:]

        # Assign Energy = x[-1]
        energy = x[-1]

        # Compute overlaps of determinants in sspace:
        #
        #   c_m
        #
        ovlp = self.compute_overlap(x[:-1], self.sspace)

        # Compute objective function:
        #
        #   f_n = <n|H|\Psi> - E <n|\Psi>
        #
        #       = <m|H|n> c_m - E \delta_{mn} c_m
        #
        # Note: we update ovlp in-place here
        self.ci_op(ovlp, out=f_dets)
        ovlp_dets = ovlp[:self.ndet]
        ovlp_dets *= energy
        f_dets -= ovlp_dets

        # Compute constraint functions
        for i, constraint in enumerate(self._constraints):
            f_cons[i] = constraint[0](x)

        # Return objective
        return f

    def compute_jacobian(self, x):
        r"""
        Compute the FanCI Jacobian function.

            j : x[k] -> y[n, k]

        """
        # Allocate Jacobian matrix (in transpose memory order)
        jac = np.empty((self.nequation, x.size), order='F', dtype=x.dtype)
        jac_dets = jac[:self.ndet]
        jac_cons = jac[self.ndet:]

        # Assign Energy = x[-1]
        energy = x[-1]

        # Compute overlaps in pspace:
        #
        #   c_n
        #
        ovlp = self.compute_overlap(x[:-1], self.pspace)

        # Compute overlap derivatives in sspace:
        #
        #   d(c_m)/d(p_k)
        #
        d_ovlp = self.compute_overlap_deriv(x[:-1], self.sspace)

        # Compute Jacobian:
        #
        #   J_{nk} = d(<n|H|\Psi>)/d(p_k) - E d(<n|\Psi>)/d(p_k) - dE/d(p_k) <n|\Psi>
        #
        # Iterate over columns of Jacobian (except final column) and rows of d_ovlp
        for jac_col, d_ovlp_row in zip(jac_dets.transpose()[:-1], d_ovlp):

            # Compute each column of the Jacobian:
            #
            #   d(<n|H|\Psi>)/d(p_k) = <m|H|n> d(c_m)/d(p_k)
            #
            #   E d(<n|\Psi>)/d(p_k) = E \delta_{nk} d(c_n)/d(p_k)
            #
            # Note: we update d_ovlp in-place here
            self.ci_op(d_ovlp_row, out=jac_col)
            d_ovlp_dets = d_ovlp_row[:self.ndet]
            d_ovlp_dets *= energy
            jac_col -= d_ovlp_dets

        # Compute final column of the Jacobian:
        #
        #   dE/d(p_k) <n|\Psi> = dE/d(p_k) \delta_{nk} c_n
        #
        jac_dets[:, -1] = ovlp
        jac_dets[:, -1] *= -1

        # Compute Jacobian of constraint functions
        for i, constraint in enumerate(self._constraints):
            jac_cons[i] = constraint[1](x)

        # Return Jacobian
        return jac

    def _norm_det_constraint(self, i, val):
        r"""
        Generate determinant norm constraint functions.

        Parameters
        ----------
        i : int
            Index of determinant whose norm to constrain.
        val : float
            Value to which to constrain norm.

        Returns
        -------
        f: function
            Constraint function.
        j : function
            Constraint derivative function.

        """
        def f(x):
            r""""
            <\psi_i|\Psi> - v_i

            """
            return self.compute_overlap(x[:-1], self.sspace[np.newaxis, i])[0] - val

        def j(x):
            r""""
            d(<\psi_i|\Psi>)/d(p_k)

            """
            return self.compute_overlap_deriv(x[:-1], self.sspace[np.newaxis, i])[0]

        return f, j

    def _norm_param_constraint(self, i, val):
        r"""
        Generate parameter norm constraint functions.

        Parameters
        ----------
        i : int
            Index of parameter whose value to constrain.
        val : float
            Value to which to constrain parameter.

        Returns
        -------
        f: function
            Constraint function.
        j : function
            Constraint derivative function.

        """
        def f(x):
            r""""
            p_i - v_i

            """
            return x[i] - val

        def j(x):
            r"""
            \delta_{ki}

            """
            y = np.zeros_like(x)
            y[i] = 1
            return y

        return f, j

    def _apply_mask(self, x0):
        r"""
        Generate masked objective, jacobian, and x0 for optimization with frozen parameters.

        Parameters
        ----------
        x0 : np.ndarray
            Initial guess for all parameters.

        Returns
        -------
        f : function
            Masked objective function.
        j : function
            Masked Jacobian function.
        x : np.ndarray
            Masked initial guess array.

        """
        y0 = np.copy(x0)

        def f(x):
            r"Masked objective function."
            y = np.copy(y0)
            y[self.mask] = x
            return self.compute_objective(y)

        def j(x):
            r"Masked Jacobian function."
            y = np.copy(y0)
            y[self.mask] = x
            return self.compute_jacobian(y)[:, self.mask]

        return f, j, np.copy(x0[self.mask])
