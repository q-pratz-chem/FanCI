""" Test APIGOverlap"""
import objective.apig import APIGOverlap


def test_init_system_apig():
    pass


def test_init_overlap():
    pass


def test_apigoverlap():
    """
    """
    nbasis = 15
    nocc = 3
    occs_array = [0, 1, 2]
    x = np.arange(45, dtype=float).reshape(nbasis, nocc)
    test = APIGOverlap(nbasis, nocc)
    test.overlap(x, occs_array)


test_apigoverlap()
