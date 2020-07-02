r"""
FanCI base class module.

"""

from abc import ABCMeta, abstractmethod

from collections import OrderedDict

from typing import Any, Callable, Dict, Sequence, Tuple

import numpy as np

from scipy.optimize import OptimizeResult, least_squares, root

import pyci


__all__ = [
    'FanCI',
    ]


Constraint = Tuple[Callable, Callable]
r"""
Constraint function and gradient (f, dfdx) type hint.

"""


class FanCI(metaclass=ABCMeta):
    r"""
    FanCI problem class.

    """

    @property
    def nequation(self) -> int:
        r"""
        Number of nonlinear equations.

        """
        return self._nequation

    @property
    def nproj(self) -> int:
        r"""
        Number of determinants in projection ("P") space.

        """
        return self._nproj

    @property
    def nparam(self) -> int:
        r"""
        Number of parameters for this FanCI problem.

        """
        return self._nparam

    @property
    def nactive(self) -> int:
        r"""
        Number of active parameters for this FanCI problem.

        """
        return self._nactive

    @property
    def constraints(self) -> Tuple[str]:
        r"""
        List of constraints.

        """
        return tuple(self._constraints.keys())

    @property
    def mask(self) -> np.ndarray:
        r"""
        Frozen parameter mask.

        """
        return self._mask_view

    @property
    def ham(self) -> pyci.hamiltonian:
        r"""
        PyCI Hamiltonian.

        """
        return self._ham

    @property
    def wfn(self) -> pyci.wavefunction:
        r"""
        PyCI wave function.

        """
        return self._wfn

    @property
    def ci_op(self) -> pyci.sparse_op:
        r"""
        PyCI sparse CI matrix operator.

        """
        return self._ci_op

    @property
    def pspace(self) -> np.ndarray:
        r"""
        Array of determinant occupations in projection ("P") space.

        """
        return self._pspace

    @property
    def sspace(self) -> np.ndarray:
        r"""
        Array of determinant occupations in auxiliiary ("S") space.

        """
        return self._sspace

    @property
    def nbasis(self) -> int:
        r"""
        Number of molecular orbital basis functions.

        """
        return self._wfn.nbasis

    @property
    def nocc_up(self) -> int:
        r"""
        Number of spin-up occupied orbitals.

        """
        return self._wfn.nocc_up

    @property
    def nocc_dn(self) -> int:
        r"""
        Number of spin-down occupied orbitals.

        """
        return self._wfn.nocc_dn

    @property
    def nvir_up(self) -> int:
        r"""
        Number of spin-up virtual orbitals.

        """
        return self._wfn.nvir_up

    @property
    def nvir_dn(self) -> int:
        r"""
        Number of spin-down virtual orbitals.

        """
        return self._wfn.nvir_dn

    def __init__(self, ham: pyci.hamiltonian, wfn: pyci.wavefunction, nproj: int, nparam: int,
            norm_param: Sequence[Tuple[int, float]] = None,
            norm_det: Sequence[Tuple[int, float]] = None,
            constraints: Dict[str, Constraint] = None,
            mask: Sequence[int] = None,
            ) -> None:
        r"""
        Initialize the FanCI problem.

        Parameters
        ----------
        ham : pyci.hamiltonian
            PyCI Hamiltonian.
        wfn : pyci.wavefunction
            PyCI wave function.
        nproj : int
            Number of determinants in projection ("P") space.
        nparam : int
            Number of parameters for this FanCI problem.
        norm_param : Sequence[Tuple[int, float]], optional
            Indices of parameters whose values to constrain, and the value to which to constrain
            them.
        norm_det : Sequence[Tuple[int, float]], optional
            Indices of determinant whose overlaps to constrain, and the value to which to constrain
            them.
        constraints : Dict[str, Constraint], optional
            Pairs of functions (f, dfdx) corresponding to additional constraints.
        mask : Sequence[int] or Sequence[bool], optional
            List of parameters to freeze. If the list contains ints, then each element corresponds
            to a frozen parameter. If the list contains bools, then each element indicates whether
            that parameter is active (True) or frozen (False).

        """
        # Parse arguments
        # ---------------

        # Generate constraints dict
        if constraints is None:
            constraints = OrderedDict()
        elif isinstance(constraints, dict):
            constraints = OrderedDict(constraints)
        else:
            raise TypeError('`constraints` must be dictionary `{name: (f, dfdx)}`')

        # Add norm_det and norm_param constraints
        norm_param = list() if norm_param is None else norm_param
        norm_det = list() if norm_det is None else norm_det
        for index, value in norm_param:
            name = f'p_{{{index}}} - v_{{{index}}}'
            constraints[name] = self.make_param_constraint(index, value)
        for index, value in norm_det:
            name = f'<\\psi_{{{index}}}|\\Psi> - v_{{{index}}}'
            constraints[name] = self.make_det_constraint(index, value)

        # Number of nonlinear equations
        nequation = nproj + len(constraints)

        # Generate mask (True for active parameter, False for frozen parameter)
        if mask is None:
            mask = np.ones(nparam, dtype=np.bool)
        else:
            mask = np.array(mask)
            # Boolean mask
            if mask.dtype == np.bool:
                if mask.size != nparam:
                    raise ValueError('boolean mask must have length `nparam`')
            # Convert integer mask to boolean
            else:
                ints = mask
                mask = np.ones(nparam, dtype=np.bool)
                mask[ints] = 0

        # Number of active parameters
        nactive = mask.sum()

        # Check if system is underdetermined
        if nequation < nactive:
            raise ValueError('system is underdetermined')

        # Generate determinant spaces
        # ---------------------------

        # Handle wfn parameter; decide values for generating determinants from its type
        wfn = wfn.copy()
        if isinstance(wfn, pyci.doci_wfn):
            e_max = min(wfn.nocc_up, wfn.nvir_up)
            connections = (1,)
        elif isinstance(wfn, (pyci.fullci_wfn, pyci.genci_wfn)):
            e_max = min(wfn.nocc, wfn.nvir)
            connections = (1, 2)
        else:
            raise TypeError('`wfn` must be a `pyci.{doci,fullci,genci}_wfn`')

        # Fill wfn with P space determinants in excitation order until len(wfn) >= nproj;
        # only add Hartree-Fock det. (zero order excitation) if wfn is empty
        for nexc in range(bool(len(wfn)), e_max + 1):
            if len(wfn) >= nproj:
                break
            wfn.add_excited_dets(nexc)
        if len(wfn) < nproj:
            raise ValueError('unable to generate `nproj` determinants')

        # Truncate wave function if we generated > nproj determinants
        if len(wfn) > nproj:
            wfn = wfn.truncated(nproj)

        # Fill wfn with S space determinants
        for det in wfn.to_det_array(nproj):
            wfn.add_excited_dets(*connections, det=det)

        # Compute arrays of occupations (flattened; spin-up, then spin-down if applicable)
        sspace = wfn.to_occs_array()
        sspace = sspace.reshape(sspace.shape[0], -1)
        pspace = sspace[:nproj]

        # Compute CI matrix operator with nproj rows and len(wfn) columns
        ci_op = pyci.sparse_op(ham, wfn, nproj)

        # Assign attributes to instance
        self._nequation = nequation
        self._nproj = nproj
        self._nparam = nparam
        self._nactive = nactive
        self._constraints = constraints
        self._mask = mask
        self._mask_view = mask[...]
        self._ham = ham
        self._wfn = wfn
        self._ci_op = ci_op
        self._pspace = pspace
        self._sspace = sspace

        # Set arrays to read-only
        self._pspace.setflags(write=0)
        self._sspace.setflags(write=0)
        self._mask_view.setflags(write=0)

    def optimize(self, x0: np.ndarray, mode: str = 'lstsq', use_jac: bool = False, **kwargs: Any) \
            -> OptimizeResult:
        r"""
        Optimize the wave function parameters.

        Parameters
        ----------
        x0 : np.ndarray
            Initial guess for wave function parameters.
        mode : ('lstsq' | 'root' | 'cma'), default='lstsq'
            Solver mode.
        use_jac : bool, default=False
            Whether to use the Jacobian function or a finite-difference approximation.
        kwargs : Any, optional
            Additional keyword arguments to pass to optimizer.

        Returns
        -------
        result : scipy.optimize.OptimizeResult
            Result of optimization.

        """
        # Check x0 vector length
        if x0.size != self.nparam:
            raise ValueError('length of `x0` does not match `param`')
        # Check if system is underdetermined
        elif self.nequation < self.nactive:
            raise ValueError('system is underdetermined')

        # Convert x0 to proper dtype array
        x0 = np.array(x0, dtype=pyci.c_double)

        # Prepare objective, Jacobian, x0
        if self.nactive < self.nparam:
            # Generate objective, Jacobian, x0 with frozen parameters
            x_ref = np.copy(x0)
            f = self.mask_function(self.compute_objective, x_ref)
            j = self.mask_function(self.compute_jacobian, x_ref)
            x0 = np.copy(x0[self.mask])
        else:
            # Use bare functions
            f = self.compute_objective
            j = self.compute_jacobian

        # Set up initial arguments to optimizer
        opt_args = (f,)
        opt_kwargs = kwargs.copy()
        if use_jac:
            opt_kwargs['jac'] = j

        # Parse mode parameter; choose optimizer and fix arguments
        if mode == 'lstsq':
            optimizer = least_squares
        elif mode == 'root':
            if self.nequation != self.nactive:
                raise ValueError('\'root\' does not work with over-determined system')
            optimizer = root
        elif mode == 'cma':
            raise NotImplementedError
        else:
            raise ValueError('invalid mode parameter')

        # Run optimizer
        return optimizer(*opt_args, **opt_kwargs)

    def add_constraint(self, name: str, f: Callable, dfdx: Callable = None) -> None:
        r"""
        Add a constraint to the system.

        ``dfdx`` must be specified to compute the Jacobian of the system.

        Parameters
        ----------
        name : str
            Label for constraint.
        f : Callable
            Constraint function.
        dfdx : Callable, optional
            Gradient of constraint function.

        """
        self._constraints[name] = f, dfdx
        # Update nequation
        self._nequation = self._nproj + len(self._constraints)

    def remove_constraint(self, name: str) -> None:
        r"""
        Remove a constraint from the system.

        Parameters
        ----------
        name : str
            Label for constraint.

        """
        del self._constraints[name]
        # Update nequation
        self._nequation = self._nproj + len(self._constraints)

    def freeze_parameter(self, *params: Sequence[int]) -> None:
        r"""
        Set a FanCI parameter to be frozen during optimization.

        Parameters
        ----------
        params : Sequence[int]
            Indices of parameters to freeze.

        """
        for param in params:
            self._mask[param] = False
        # Update nactive
        self._nactive = self._mask.sum()

    def unfreeze_parameter(self, *params: Sequence[int]) -> None:
        r"""
        Set a FanCI parameter to be active during optimization.

        Parameters
        ----------
        params : Sequence[int]
            Indices of parameters to unfreeze.

        """
        for param in params:
            self._mask[param] = True
        # Update nactive
        self._nactive = self._mask.sum()

    def compute_objective(self, x: np.ndarray) -> np.ndarray:
        r"""
        Compute the FanCI objective function.

            f : x[k] -> y[n]

        Parameters
        ----------
        x : np.ndarray
            Parameter array, [p_0, p_1, ..., p_n, E].

        Returns
        -------
        obj : np.ndarray
            Objective vector.

        """
        # Allocate objective vector
        f = np.empty(self._nequation, dtype=pyci.c_double)
        f_proj = f[:self._nproj]
        f_cons = f[self._nproj:]

        # Assign Energy = x[-1]
        energy = x[-1]

        # Compute overlaps of determinants in sspace:
        #
        #   c_m
        #
        ovlp = self.compute_overlap(x[:-1], self._sspace)

        # Compute objective function:
        #
        #   f_n = <n|H|\Psi> - E <n|\Psi>
        #
        #       = <m|H|n> c_m - E \delta_{mn} c_m
        #
        # Note: we update ovlp in-place here
        self._ci_op(ovlp, out=f_proj)
        ovlp_proj = ovlp[:self._nproj]
        ovlp_proj *= energy
        f_proj -= ovlp_proj

        # Compute constraint functions
        for i, constraint in enumerate(self._constraints.values()):
            f_cons[i] = constraint[0](x)

        # Return objective
        return f

    def compute_jacobian(self, x: np.ndarray) -> np.ndarray:
        r"""
        Compute the Jacobian of the FanCI objective function.

            j : x[k] -> y[n, k]

        Parameters
        ----------
        x : np.ndarray
            Parameter array, [p_0, p_1, ..., p_n, E].

        Returns
        -------
        jac : np.ndarray
            Jacobian matrix.

        """
        # Allocate Jacobian matrix (in transpose memory order)
        jac = np.empty((self._nequation, self._nactive), order='F', dtype=pyci.c_double)
        jac_proj = jac[:self._nproj]
        jac_cons = jac[self._nproj:]

        # Assign Energy = x[-1]
        energy = x[-1]

        # Compute Jacobian:
        #
        #   J_{nk} = d(<n|H|\Psi>)/d(p_k) - E d(<n|\Psi>)/d(p_k) - dE/d(p_k) <n|\Psi>
        #
        # Compute overlap derivatives in sspace:
        #
        #   d(c_m)/d(p_k)
        #
        d_ovlp = self.compute_overlap_deriv(x[:-1], self._sspace)

        # Check is energy parameter is active:
        if self._mask[-1]:
            #
            # Compute final Jacobian column if mask[-1] == True
            #
            #   dE/d(p_k) <n|\Psi> = dE/d(p_k) \delta_{nk} c_n
            #
            ovlp = self.compute_overlap(x[:-1], self._pspace)
            ovlp *= -1
            jac_proj[:, -1] = ovlp
            #
            # Remove final column from jac_proj
            #
            jac_proj = jac_proj[:, :-1]

        # Iterate over columns of Jacobian and rows of d_ovlp
        for jac_col, d_ovlp_row in zip(jac_proj.transpose(), d_ovlp):
            #
            # Compute each column of the Jacobian:
            #
            #   d(<n|H|\Psi>)/d(p_k) = <m|H|n> d(c_m)/d(p_k)
            #
            #   E d(<n|\Psi>)/d(p_k) = E \delta_{nk} d(c_n)/d(p_k)
            #
            # Note: we update d_ovlp in-place here
            self._ci_op(d_ovlp_row, out=jac_col)
            d_ovlp_proj = d_ovlp_row[:self._nproj]
            d_ovlp_proj *= energy
            jac_col -= d_ovlp_proj

        # Compute Jacobian of constraint functions
        for i, constraint in enumerate(self._constraints.values()):
            jac_cons[i] = constraint[1](x)

        # Return Jacobian
        return jac

    def make_param_constraint(self, i: int, val: float) -> Constraint:
        r"""
        Generate parameter constraint functions.

        Parameters
        ----------
        i : int
            Index of parameter whose value to constrain.
        val : float
            Value to which to constrain parameter.

        Returns
        -------
        f: Callable
            Constraint function.
        dfdx : Callable
            Gradient of constraint function.

        """
        def f(x: np.ndarray) -> float:
            r""""
            Constraint function p_{i} - v_{i}.

            """
            return x[i] - val if self._mask[i] else 0

        def dfdx(x: np.ndarray) -> np.ndarray:
            r"""
            Constraint gradient \delta_{ki}.

            """
            y = np.zeros(self.nactive, dtype=pyci.c_double)
            if self._mask[i]:
                y[self._mask[:i].sum()] = 1
            return y

        return f, dfdx

    def make_det_constraint(self, i: int, val: float) -> Constraint:
        r"""
        Generate determinant overlap constraint functions.

        Parameters
        ----------
        i : int
            Index of determinant whose overlap to constrain.
        val : float
            Value to which to constrain overlap.

        Returns
        -------
        f: Callable
            Constraint function.
        dfdx : Callable
            Gradient of constraint function.

        """
        def f(x: np.ndarray) -> float:
            r""""
            Constraint function <\psi_{i}|\Psi> - v_{i}.

            """
            return self.compute_overlap(x[:-1], self._sspace[np.newaxis, i])[0] - val

        def dfdx(x: np.ndarray) -> np.ndarray:
            r""""
            Constraint gradient d(<\psi_{i}|\Psi>)/d(p_{k}).

            """
            return self.compute_overlap_deriv(x[:-1], self._sspace[np.newaxis, i])[0]

        return f, dfdx

    def mask_function(self, f: Callable, x_ref: np.ndarray) -> Callable:
        r"""
        Generate masked function for optimization with frozen parameters.

        Parameters
        ----------
        f : Callable
            Initial function.
        x_ref : np.ndarray
            Full parameter vector including frozen terms.

        Returns
        -------
        f : Callable
            Masked function.

        """
        if self.nactive == self.nparam:
            return f

        def masked_f(x: np.ndarray) -> Any:
            r"""
            Masked function.

            """
            y = np.copy(x_ref)
            y[self._mask] = x
            return f(y)

        return masked_f

    @abstractmethod
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
        raise NotImplementedError('this method must be overwritten in a sub-class')

    @abstractmethod
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
        raise NotImplementedError('this method must be overwritten in a sub-class')
