"""Refactor objective function"""

import numpy as np

from pyci import doci

from wfns.backend.slater import create, excite
from wfns.ham.senzero import SeniorityZeroHamiltonian
from wfns.wfn.geminal.base import BaseGeminal
from wfns.wfn.ci.base import CIWavefunction
from wfns.wfn.geminal.apig import APIG
from wfns.wfn.geminal.ap1rog import AP1roG


def occs2det(occs):
    """
    occs: numpy (N,M) array where N = wfn.size, M = nspins
    The wfn is stored as alpha and beta occupations.
    """
    dets = []
    for occ in occs:
        sd = create(0, *occ)
        dets.append(sd)
    return dets


class ObjFunction:
    """
    Objective function

    """

    def __init__(self, wfn, ham, len_pspace, dets):
        self.wfn = wfn
        self.ham = ham
        # Fix: Add check for len property of dets
        self.pspace = dets[:len_pspace]
        self.sspace = dets[len_pspace:]
        self.ham_elements = self.eval_hamiltonian_terms()

    def eval_hamiltonian_terms(self):
        ham = self.ham
        # Construct the Hamiltonian matrix for the
        # Slater determinants in the reference and
        # projection space
        # Based on FanPy brute ci solver
        ham_elements = np.zeros((len(self.pspace), len(self.sspace)))
        for i, sd1 in enumerate(self.pspace):  # projection space
            for j, sd2 in enumerate(self.sspace):  # wavefunction space
                ham_elements[i, j] += sum(ham.integrate_sd_sd(sd1, sd2))
        return ham_elements

    def __call__(self, x):
        # Fix needed: number of equation in objective
        # has to be higher or equal than number
        # of unknowns.
        # O[numberunkn + 1]
        # Fix: Parameter asignation nees to be specific to the
        # FanCI wfn type.
        if isinstance(self.wfn, CIWavefunction):
            params = x[:-1]
            self.wfn.assign_params = params
        elif isinstance(self.wfn, BaseGeminal):
            params = x[:-1].reshape(self.wfn.ngem, self.wfn.norbpair)
            self.wfn.assign_params = params

        coeffs = np.empty_like(self.sspace, dtype=float)
        for idx, sd in enumerate(self.sspace):
            coeffs[idx] = self.wfn.get_overlap(sd)
        ovlps = np.empty_like(self.pspace, dtype=float)
        for idx, sd in enumerate(self.pspace):
            ovlps[idx] = self.wfn.get_overlap(sd)
        energy = x[-1]
        # \sum_{n} {<m|H|n> c_n - E \delta_{m,n} c_n}
        f_vals = np.einsum("ij,j->i", self.ham_elements, coeffs)
        f_vals -= energy * ovlps
        return f_vals


class JacFun:
    """
    Objective function Jacobian

    """

    def __init__(self, objective):
        if isinstance(objective, ObjFunction):
            self.fun = objective
        else:
            raise TypeError("objective must be an ObjFunction instance")

    def __call__(self, x):
        # Fix needed: J[numbereqs., numberunkn.]
        ham_elements = self.fun.ham_elements
        wfn = self.fun.wfn
        sspace = self.fun.sspace
        pspace = self.fun.pspace
        energy = x[-1]
        size_wfnprms = len(x[:-1])
        
        # Jacobian equation
        # d(<m|H|\Psi>)/dp_k - E d(<m|\Psi>)/dp_k - (dE/dp_k) <m|\Psi>
        jac = np.empty((len(pspace), len(x)))
        # d(<m|H|\Psi>)/dp_k = \sum_n {<m|H|n> dc_n/dp_k}
        dcoeffs = [
            [wfn.get_overlap(sd, deriv=idx) for idx in range(size_wfnprms)]
            for sd in sspace
        ]
        hamdc = np.einsum("ij,jl->il", ham_elements, dcoeffs)
        jac[: len(pspace), :-1] = hamdc
        # E <m|\Psi>/dp_k = E \sum_n {\delta_{m,n} dc_n/dp_k}
        # Taken from FanPy
        dovlps = [
            [wfn.get_overlap(sd, deriv=idx) for idx in range(size_wfnprms)]
            for sd in pspace
        ]
        dovlps = np.array(dovlps)
        jac[: len(pspace), :-1] -= energy * dovlps
        # (E/dp_k) <m|\Psi> = (E / dp_k) \sum_n {\delta_{m,n} c_n}
        denergy = -np.array([wfn.get_overlap(sd) for sd in pspace])
        jac[:, -1] = denergy.T
        return jac


if __name__ == "__main__":
    # System specifications
    nelec = 2
    nspin = 6
    nbasis = nspin // 2
    nalpha = nelec // 2
    # Define Hamiltonian
    one_int = np.eye(3, dtype=float)
    two_int = np.eye(9, dtype=float).reshape(nbasis, nbasis, nbasis, nbasis)
    nuc_nuc = 0.0
    ham = SeniorityZeroHamiltonian(one_int, two_int, energy_nuc_nuc=nuc_nuc)

    # Define a CI space |Phi>. This is the wfn of a given state being analyzed.
    # Generate the |Phi> wfn with PyCI.
    # Make initial projection space (pspace)
    # Ex.: Add 0b001001 and 0b010010 Slater dets
    pyciwfn = doci.wfn(nbasis, nalpha)
    pyciwfn.add_hartreefock_det()
    pyciwfn.add_det_from_occs(np.array([1]))
    len_pspace = len(pyciwfn)
    # Add excitations connected to pspace
    # (pspace: 0b001001 and 0b010010 Slater dets)
    for sd in pyciwfn:
        pyciwfn.add_excited_dets(1, det=sd)
    # Transform from PyCI wfn dets format to FanCI dets format
    # (Store the |Phi> wfn as a list of Slater dets occupations
    # and map from occupations to bitstrings?.)
    occs = []
    for a in pyciwfn:
        occa = pyciwfn.occs_from_det(a)
        occb = [i + pyciwfn.nbasis for i in occa]
        occa = np.append(occa, occb, axis=0)
        occs.append(occa)
    dets = occs2det(occs)

    #######################
    # Export Slater dets space  
    # (sset, PyCI wfn) to FanCI.
    # CI wavefunction case
    #######################
    ## sdets = dets[len_pspace:]
    ## fanciwfn = CIWavefunction(nelec, ham.nspin, params=None, memory="1gb", sd_vec=sdets)


    ###########################
    # Export Slater dets space 
    # (sset, PyCI wfn) to FanCI???
    # Geminal wavefunction case
    ###########################
    nelec = 2 * pyciwfn.nocc
    nspin = 2 * pyciwfn.nbasis
    ## refsd = 0b001001
    ## fanciwfn = AP1roG(nelec, nspin, ref_sd=refsd, orbpairs=None)
    fanciwfn = APIG(nelec, nspin)
    ## print(fanciwfn.params)


    # Try objective and jacobian functions.
    # Generate some initial parameters, where
    # the last parameter would correspond to an energy value.
    ## randomguess = np.random.rand(len(fanciwfn.params) + 1)
    randomguess = np.random.rand((fanciwfn.ngem * fanciwfn.norbpair) + 1)
    randomfun = ObjFunction(fanciwfn, ham, len_pspace, dets)
    val = randomfun(randomguess)
    somejacobian = JacFun(randomfun)
    jval = somejacobian(randomguess)
    print(val)
    print(jval)
