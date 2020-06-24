""" Test APIGFanCI"""
import numpy as np
import pytest

from pyci import doci

from wfns.backend import math_tools
from objective.apig import APIGOverlap, APIGFanCI


class ReplaceAPIGFanCI(APIGFanCI):
    def __init__(self):
        """Empty init"""


class ReplaceAPIGOverlap(APIGOverlap):
    def __init__(self):
        """Empty init"""


def test_apig_init():
    """
    """
    nbasis=5
    one_mo = np.arange(25, dtype=float).reshape(nbasis, nbasis)
    two_mo = np.arange(25*25, dtype=float).reshape((nbasis,)*4)
    ham = doci.ham.from_mo_arrays(0.0, one_mo, two_mo)
    nocc = 2
    apig = APIGFanCI(ham, nocc)
    assert apig.ndet_pspace == ham.nbasis * nocc

    with pytest.raises(ValueError):
        APIGFanCI(ham, nocc, ndet_pspace=9)
    nocc = 6
    with pytest.raises(ValueError):
        APIGFanCI(ham, nocc)


def test_apig_init_system():
    """
    """
    nbasis=3
    one_mo = np.arange(9, dtype=float).reshape(nbasis, nbasis)
    two_mo = np.arange(9*9, dtype=float).reshape((nbasis,)*4)
    ham = doci.ham.from_mo_arrays(0.0, one_mo, two_mo)
    nocc = 2
    with pytest.raises(ValueError):
        APIGFanCI(ham, nocc, ndet_pspace=6)
    
    nbasis=6
    one_mo = np.arange(36, dtype=float).reshape(nbasis, nbasis)
    two_mo = np.arange(36*36, dtype=float).reshape((nbasis,)*4)
    ham = doci.ham.from_mo_arrays(0.0, one_mo, two_mo)
    nocc = 2
    apig = APIGFanCI(ham, nocc, ndet_pspace=12)
    assert apig.ham == ham
    assert apig.nbasis == nbasis
    assert apig.nocc_up == nocc
    assert apig.nocc_dn == nocc
    assert apig.ndet_pspace == 12
    assert apig.ndet_sspace == 15 


def test_apig_init_overlap():
    pass


def test_apig_overlap():
    """
    """
    nbasis = 3
    nocc = 2
    occ = np.array([0, 2])
    occs_array = np.array([occ])
    parameters = np.arange(1,7, dtype=float).reshape(nbasis, nocc)

    test = APIGOverlap(nbasis, nocc)
    ovlps1 = test.overlap(parameters, occs_array)

    fanci_pmnt = math_tools.permanent_combinatoric
    ovlps2 = [fanci_pmnt(parameters[occ]) for occ in occs_array]

    assert np.allclose(ovlps1, ovlps2)


def test_apig_overlap_deriv():
    pass


def test_apig_permanent():
    """ 
    """
    matrix = np.array(
            [
                [1, 1, 1, 1],
                [2, 1, 0, 0],
                [3, 0, 1, 0],
                [4, 0, 0, 1],
            ]
        )
    answer = 10
    test = ReplaceAPIGOverlap
    pmnt = test.permanent(matrix)
    assert pmnt == answer


def test_apig_permanent_deriv():
    pass

