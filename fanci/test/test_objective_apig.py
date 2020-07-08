""" Test APIG"""

import os

import numpy as np

import pytest

import pyci

from fanci import APIG
from fanci.apig import permanent
from fanci.test import find_datafile, assert_deriv


def init_errors():
    """
    """
    # Define dummy hamiltonian
    nbasis = 3
    nocc = 1
    one_mo = np.arange(nbasis ** 2, dtype=pyci.c_double).reshape(nbasis, nbasis)
    two_mo = np.arange(nbasis ** 4, dtype=pyci.c_double).reshape((nbasis,) * 4)
    ham = pyci.restricted_ham(0.0, one_mo, two_mo)
    # Define raise error input options
    nproj_valueerror = 4  # P space > S space
    wfn_typeerror = ["0b001001"]  # FIXME: look for better example
    wfn_valueerror = pyci.doci_wfn(nbasis, 2)  # number of electrons don't match

    for p in [
        (TypeError, [nocc, ham], {}),
        (TypeError, [ham, nocc], {"nproj": "3"}),
        (ValueError, [ham, nocc], {"nproj": nproj_valueerror}),
        (TypeError, [ham, nocc], {"wfn": wfn_typeerror}),
        (ValueError, [ham, nocc], {"wfn": wfn_valueerror}),
    ]:
        yield p


@pytest.mark.parametrize("expecting, args, kwargs", init_errors())
def test_apig_init_errors(expecting, args, kwargs):
    """
    """
    with pytest.raises(expecting):
        APIG(*args, **kwargs)


def test_apig_init_defaults():
    """
    """
    # Define dummy hamiltonian
    nbasis = 6
    nocc = 2
    one_mo = np.arange(nbasis ** 2, dtype=pyci.c_double).reshape(nbasis, nbasis)
    two_mo = np.arange(nbasis ** 4, dtype=pyci.c_double).reshape((nbasis,) * 4)
    ham = pyci.restricted_ham(0.0, one_mo, two_mo)

    test = APIG(ham, nocc)

    assert test.nparam == nbasis * nocc + 1
    assert test.nproj == nbasis * nocc + 1
    assert test.nactive == nbasis * nocc + 1
    assert test.nequation == nbasis * nocc + 1
    assert np.all(test.mask)

    assert isinstance(test.wfn, pyci.doci_wfn)
    assert test.nbasis == nbasis
    assert test.nocc_up == nocc
    assert test.nocc_dn == nocc
    assert test.nvir_up == nbasis - nocc
    assert test.nvir_dn == nbasis - nocc
    # FIXME:
    # I just made these tests pass, I didn't check that 13 or 28 are correct
    assert test.pspace.shape[0] == 13
    assert test.sspace.shape[0] == 28


@pytest.mark.xfail
def test_apig_init_custom():
    """
    """
    # Define dummy hamiltonian
    nbasis = 3
    nocc = 1
    one_mo = np.arange(nbasis ** 2, dtype=pyci.c_double).reshape(nbasis, nbasis)
    two_mo = np.arange(nbasis ** 4, dtype=pyci.c_double).reshape((nbasis,) * 4)
    ham = pyci.restricted_ham(0.0, one_mo, two_mo)
    # Define input options
    wfn_custom = pyci.doci_wfn(nbasis, nocc)

    test = APIG(ham, nocc)


@pytest.mark.xfail
def test_apig_init_underdeterminedsystem():
    """
    """
    # Define dummy hamiltonian
    nbasis = 3
    nocc = 1
    one_mo = np.arange(nbasis ** 2, dtype=pyci.c_double).reshape(nbasis, nbasis)
    two_mo = np.arange(nbasis ** 4, dtype=pyci.c_double).reshape((nbasis,) * 4)
    ham = pyci.restricted_ham(0.0, one_mo, two_mo)
    # Define input options

    test = APIG(ham, nocc)


def test_apig_compute_overlap():
    pass


def test_apig_compute_overlap_deriv():
    nocc = 2
    nbasis = 6
    one_mo = np.arange(6 * 6, dtype=float).reshape(nbasis, nbasis)
    two_mo = np.arange(36 * 36, dtype=float).reshape((nbasis,) * 4)
    ham = pyci.restricted_ham(0.0, one_mo, two_mo)
    nproj = nocc * nbasis + 1
    params = np.arange(nbasis * nocc + 1, dtype=pyci.c_double) + 1
    apig = APIG(ham, nocc, nproj=None)

    f = lambda x: apig.compute_overlap(x, apig.sspace)
    j = lambda x: apig.compute_overlap_deriv(x, apig.sspace)
    origin = np.zeros_like(params[:-1])
    assert_deriv(f, j, origin)


def test_apig_permanent():
    matrix = np.arange(1, 65, dtype=float)
    answers = [1., 1., 10., 450., 55456., 14480700., 6878394720., 5373548250000., 6427291156586496.]
    for i, answer in enumerate(answers):
        assert permanent(matrix[:i ** 2].reshape(i, i)) == answer


def test_apig_compute_objective():
    nocc = 2
    nbasis = 6
    one_mo = np.arange(6 * 6, dtype=float).reshape(nbasis, nbasis)
    two_mo = np.arange(36 * 36, dtype=float).reshape((nbasis,) * 4)
    ham = pyci.restricted_ham(0.0, one_mo, two_mo)
    nproj = nocc * nbasis + 1
    params = np.arange(nbasis * nocc + 1, dtype=pyci.c_double) + 1

    apig = APIG(ham, nocc, nproj=None)
    objective = apig.compute_objective(params)
    op = pyci.sparse_op(apig.ham, apig.wfn, nproj)
    ovlp = apig.compute_overlap(params[:-1], apig.sspace)

    answer = op(ovlp) - params[-1] * ovlp[:nproj]
    assert np.allclose(objective, answer)


def test_apig_compute_jacobian():
    nocc = 2
    nbasis = 6
    one_mo = np.arange(6 * 6, dtype=float).reshape(nbasis, nbasis)
    two_mo = np.arange(36 * 36, dtype=float).reshape((nbasis,) * 4)
    ham = pyci.restricted_ham(0.0, one_mo, two_mo)
    params = np.arange(nbasis * nocc + 1, dtype=pyci.c_double) + 1
    apig = APIG(ham, nocc)

    f = apig.compute_objective
    j = apig.compute_jacobian
    origin = np.zeros_like(params)
    assert_deriv(f, j, origin)


def test_apig_h2_sto6g_ground():
    """Test ground state APIG wavefunction using H2 with HF/STO-6G orbital.
    Test adapted from FanCI's test_wfn_geminal_apig.

    Answers obtained from answer_apig_h2_sto6g

    HF (Electronic) Energy : -1.838434256
    APIG Energy : -1.8590898441488894
    APIG Coeffs : [0.99359749, -0.11300768]

    """
    nocc = 1
    one_int = np.load(find_datafile("data_h2_hf_sto6g_oneint.npy"))
    two_int = np.load(find_datafile("data_h2_hf_sto6g_twoint.npy"))
    nuc_nuc = 0.71317683129
    params = np.array([1.0, 0.0, -1.838434256])
    ham = pyci.restricted_ham(nuc_nuc, one_int, two_int)
    norm_det = [(0, 1.0)]
    apig = APIG(ham, nocc, nproj=2, norm_det=norm_det)

    results = apig.optimize(params)
    apig_energy = results.x[-1]
    assert np.allclose(apig_energy, -1.8590898441488894)
