""" Test APIGFanCI"""
import numpy as np

from pyci import doci

from wfns.backend import math_tools
from objective.apig import APIGOverlap, APIGFanCI


class ReplaceAPIGFanCI(APIGFanCI):
    def __init__(self):
        """Empty init"""


class ReplaceAPIGOverlap(APIGOverlap):
    def __init__(self):
        """Empty init"""


def test_apig_init_system():
    """
    Broken
    """
    nbasis = 3
    nocc = 2
    one_mo = np.arange(9, dtype=float).reshape(nbasis, nbasis)
    two_mo = np.arange(81, dtype=float).reshape((nbasis,)*4)
    ham = doci.ham.from_mo_arrays(0.0, one_mo, two_mo)

    apig = APIGFanCI(ham, nocc)
    assert apig.ndet_pspace == (nocc * nbasis)


def test_apig_init_overlap():
    pass


def test_apig_overlap():
    """
    Broken
    """
    # Variables set-up
    nbasis = 3
    nocc = 2
    occs_array = np.array([0, 2])
    parameters = np.arange(1,7, dtype=float).reshape(nbasis, nocc)

    test = APIGOverlap(nbasis, nocc)
    ovlp1 = test.overlap(parameters, occs_array)
    print(ovlp1, '\n')

    fanci_pmnt = math_tools.permanent_combinatoric
    ovlp2 = fanci_pmnt(parameters[occs_array])
    print(ovlp2)


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


# test_apig_overlap()
test_apig_init_system()