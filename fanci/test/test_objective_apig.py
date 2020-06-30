""" Test APIG"""

import os

import numpy as np

import pytest

from scipy.optimize import least_squares

import pyci

from fanci import APIG
from fanci.test import find_datafile


def test_apigfanci_init():
    """
    """
    nbasis = 10
    one_mo = np.arange(nbasis ** 2, dtype=pyci.c_double).reshape(nbasis, nbasis)
    two_mo = np.arange(nbasis ** 4, dtype=pyci.c_double).reshape((nbasis,)*4)
    ham = pyci.restricted_ham(0.0, one_mo, two_mo)

    nocc = 2
    apig = APIG(ham, nocc)
    assert apig.nproj == ham.nbasis * nocc + 1
    with pytest.raises(ValueError):
        APIG(ham, nocc, nproj=10000000)

    nocc = 11
    with pytest.raises(ValueError):
        APIG(ham, nocc)


def test_apigfanci_init_system():
    """
    """
    nbasis=3
    one_mo = np.arange(9, dtype=float).reshape(nbasis, nbasis)
    two_mo = np.arange(9*9, dtype=float).reshape((nbasis,)*4)
    ham = pyci.restricted_ham(0.0, one_mo, two_mo)
    nocc = 2
    with pytest.raises(ValueError):
        APIG(ham, nocc, nproj=6)

    nbasis=6
    one_mo = np.arange(36, dtype=float).reshape(nbasis, nbasis)
    two_mo = np.arange(36*36, dtype=float).reshape((nbasis,)*4)
    ham = pyci.restricted_ham(0.0, one_mo, two_mo)
    nocc = 2
    apig = APIG(ham, nocc, nproj=None)
    assert apig.ham == ham
    assert apig.nbasis == nbasis
    assert apig.nocc_up == nocc
    assert apig.nocc_dn == nocc
    #assert apig.nproj == 12
    #assert apig.nproj == 15


def test_apigfanci_init_overlap():
    pass


def test_apigoverlap_init():
    pass


def test_apigoverlap_overlap_deriv():
    pass


def test_apigoverlap_permanent():
    pass

def test_apigoverlap_permanent_deriv():
    pass


def test_apigfanci_compute_objective():
    # Square case
    nocc = 1
    nbasis = 2
    one_mo = np.arange(1, 5, dtype=float).reshape(nbasis, nbasis)
    two_mo = np.arange(1,17, dtype=float).reshape((nbasis,)*4)
    ham = pyci.restricted_ham(0.0, one_mo, two_mo)
    wfn = pyci.doci_wfn(ham.nbasis, nocc)
    wfn.add_all_dets()
    nproj = 2
    params = np.array([3., 2., 1.])

    apig = APIG(ham, nocc, nproj=2)
    #objective = apig.compute_objective(params)
    #assert objective.size == apig.nproj

    op = pyci.sparse_op(ham, wfn, nproj)
    hmlt_mtx = op.to_csr_matrix().toarray()
    # ovlp = [1., 2.]
    # answer = [3.*1. + 4.*2. - 3.*1.,
    #          13.*1. + 24.*2. - 3.*2.]
    ovlp = np.array(np.arange(1, 3, dtype=params.dtype))
    answer = np.dot(hmlt_mtx, ovlp) - params[-1]*ovlp
    assert np.allclose(objective, answer)

    # Rectangular case
    nocc = 2
    nbasis = 6
    one_mo = np.arange(6*6, dtype=float).reshape(nbasis, nbasis)
    two_mo = np.arange(36*36, dtype=float).reshape((nbasis,)*4)
    ham = pyci.restricted_ham(0.0, one_mo, two_mo)
    # nproj = 6 choose 2 = 15
    wfn = pyci.doci_wfn(ham.nbasis, nocc)
    wfn.add_all_dets()
    nproj = nocc * nbasis
    params = np.array([float(i+1) for i in range(13)])

    apig = APIG(ham, nocc, nproj=None)
    #objective = apig.compute_objective(params)
    #assert objective.size == apig.nproj

    op = pyci.sparse_op(ham, wfn, nproj)
    #hmlt_mtx = op.to_csr_matrix().toarray()
    # ovlp = [1, 2, 3, 4, 5, ..., 15]
    ovlp = np.array(np.arange(1, 16, dtype=params.dtype))
    answer = op(ovlp) - params[-1]*ovlp[:nproj]
    assert np.allclose(objective, answer)


def test_apigfanci_compute_jacobian():
    pass


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
    params = np.array([1., 0., -1.])
    ham = pyci.restricted_ham(nuc_nuc, one_int, two_int)
    apig = APIG(ham, nocc, nproj=2)

    #results = least_squares(apig.compute_objective, params)
    apig_energy = results.x[-1]
    assert np.allclose(apig_energy, -1.8590898441488894)


test_apigfanci_init()
test_apigfanci_init_system()
#test_apigoverlap_overlap()
test_apigoverlap_permanent()
test_apigfanci_compute_objective()
test_apig_h2_sto6g_ground()
