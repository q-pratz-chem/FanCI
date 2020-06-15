"""Refactor objective function"""

import numpy as np

from pyci import doci

from wfns.backend.slater import create
from wfns.wfn.geminal.apig import APIG


class NonLinearSystemAPIG(FanCIBase):
    r"""
    """

    def init_wavefunction(self):
        r"""
        """
        # Create your CI wfn here and return it
        #
        nelec = self.nocc_up + self.nocc_dn
        nspin = 2 * self.nbasis
        return APIG(nelec, nspin)

    def init_overlap(self, x, occs, deriv=None):
        r"""
        """
        # Create your callable() here and return it
        # def overlap(x):
        #     r"""
        #     """
        #     pass
        # return overlap
        params = x[:-1].reshape(self.wfn.ngem, self.wfn.norbpair)
            self.wfn.assign_params = params
        if deriv == None:
            pass
        else:
            pass
