""" Test DetRatio"""

import os

import numpy as np

import pytest

import pyci

from fanci import DetRatio
from fanci.test import find_datafile, assert_deriv


@pytest.fixture
def dummy_system():
    nbasis = 10
    nocc = 2
    one_mo = np.arange(nbasis ** 2, dtype=float).reshape(nbasis, nbasis)
    two_mo = np.arange(nbasis ** 4, dtype=float).reshape((nbasis,) * 4)
    ham = pyci.restricted_ham(0.0, one_mo, two_mo)
    params = np.arange(2 * ham.nbasis * nocc + 1, dtype=pyci.c_double) + 1
    return (ham, nocc, params)


def init_errors():
    """
    """
    # Define dummy hamiltonian
    nbasis = 10
    nocc = 2
    one_mo = np.arange(nbasis ** 2, dtype=pyci.c_double).reshape(nbasis, nbasis)
    two_mo = np.arange(nbasis ** 4, dtype=pyci.c_double).reshape((nbasis,) * 4)
    ham = pyci.restricted_ham(0.0, one_mo, two_mo)

    for p in [
        (ValueError, [ham, nocc, 2, 1], {}),
    ]:
        yield p


@pytest.mark.parametrize("expecting, args, kwargs", init_errors())
def test_detratio_init_errors(expecting, args, kwargs):
    """
    """
    with pytest.raises(expecting):
        DetRatio(*args, **kwargs)


def test_detratio_init_defaults(dummy_system):
    """
    """
    ham, nocc, params = dummy_system
    test = DetRatio(ham, nocc, 1, 1)

    assert test.nparam == 2 * ham.nbasis * nocc + 1
    assert test.nproj == 2 * ham.nbasis * nocc + 1
    assert test.nactive == 2 * ham.nbasis * nocc + 1
    assert test.nequation == 2 * ham.nbasis * nocc + 1
    assert np.all(test.mask)

    assert isinstance(test.wfn, pyci.doci_wfn)
    assert test.nbasis == ham.nbasis
    assert test.nocc_up == nocc
    assert test.nocc_dn == nocc
    assert test.nvir_up == ham.nbasis - nocc
    assert test.nvir_dn == ham.nbasis - nocc
    assert test.pspace.shape[0] == 41


def test_detratio_freeze_matrix(dummy_system):
    ham, nocc, params = dummy_system
    numerator = 1
    denominator = 1
    detratio = DetRatio(ham, nocc, numerator, denominator)
    detratio.freeze_matrix(0)

    expected = np.ones_like(params, dtype=np.bool)
    expected[: (ham.nbasis * nocc)] = False
    assert np.allclose(detratio.mask, expected)


def test_apig_compute_overlap(dummy_system):
    ham, nocc, params = dummy_system
    numerator = 1
    denominator = 1
    matrices = params[:-1].reshape(2, ham.nbasis, nocc)
    matrix1 = matrices[0]
    matrix2 = matrices[1]
    detratio = DetRatio(ham, nocc, numerator, denominator)

    f = lambda x, y: x / y
    for occ in detratio.sspace:
        expected = f(np.linalg.det(matrix1[occ, :]), np.linalg.det(matrix2[occ, :]))
        answer = detratio.compute_overlap(params[:-1], np.array([occ]))
        assert np.allclose(answer, expected)


@pytest.mark.xfail
def test_apig_compute_overlap_deriv(dummy_system):
    ham, nocc, params = dummy_system
    numerator = 1
    denominator = 1
    detratio = DetRatio(ham, nocc, numerator, denominator)

    f = lambda x: detratio.compute_overlap(x, detratio.sspace)
    j = lambda x: detratio.compute_overlap_deriv(x, detratio.sspace)
    # # origin = np.random.rand(len(params[:-1]))
    origin = params[:-1]
    assert_deriv(f, j, origin)


def test_apig_compute_objective(dummy_system):
    ham, nocc, params = dummy_system
    numerator = 1
    denominator = 1
    detratio = DetRatio(ham, nocc, numerator, denominator)

    nproj = 2 * ham.nbasis * nocc + 1
    objective = detratio.compute_objective(params)
    op = pyci.sparse_op(detratio.ham, detratio.wfn, nproj)
    ovlp = detratio.compute_overlap(params[:-1], detratio.sspace)
    answer = op(ovlp) - params[-1] * ovlp[:nproj]
    assert np.allclose(objective, answer)


@pytest.mark.xfail
def test_apig_compute_jacobian(dummy_system):
    ham, nocc, params = dummy_system
    numerator = 1
    denominator = 1
    detratio = DetRatio(ham, nocc, numerator, denominator)

    f = lambda x: detratio.compute_objective
    j = lambda x: detratio.compute_jacobian
    # origin = np.random.rand(len(params))
    origin = params[:-1]
    assert_deriv(f, j, origin)


def systems_ground():
    options_list = [
        (2, "be_ccpvdz", 0.0, -14.57233, [(0, 1.0)], (4 * 14 + 1), -14.600556994),
    ]
    for p in options_list:
        yield p


@pytest.mark.xfail
@pytest.mark.parametrize(
    "nocc, system, nucnuc, e_hf, normdet, nproj, expected", systems_ground()
)
def test_apig_systems_ground(nocc, system, nucnuc, e_hf, normdet, nproj, expected):
    """Test cases adapted from FanCI's test_wfn_geminal_apig.

    """
    ham = pyci.restricted_ham(find_datafile("{0:s}.fcidump".format(system)))
    numerator = 1
    denominator = 1
    nmatrices = numerator + denominator
    apig = DetRatio(ham, nocc, numerator, denominator, nproj=nproj, norm_det=normdet)

    params_guess = np.zeros((nmatrices * ham.nbasis * nocc + 1), dtype=pyci.c_double)
    params_guess[: (ham.nbasis * nocc)] = np.eye(ham.nbasis, nocc).transpose().flatten()
    params_guess[(ham.nbasis * nocc) : -1] = (
        np.eye(ham.nbasis, nocc).transpose().flatten()
    )
    params_guess[-1] = e_hf
    # FIXME: least_squares.py:814: ValueError
    # ValueError: Residuals are not finite in the initial point.
    results = apig.optimize(params_guess)
    apig_energy = results.x[-1] + nucnuc
    assert np.allclose(apig_energy, expected)
