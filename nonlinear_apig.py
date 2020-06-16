"""Refactor objective function"""

import numpy as np

from pyci import doci

from wfns.backend.slater import create
from wfns.wfn.geminal.apig import APIG


class APIGFanCI(FanCIBase):
    r"""
    """

    def init_wavefunction(self):
        r"""
        """
        # Create your CI wfn here and return it
        # PyCI wfn
        pass

    def init_overlap(self):
        r"""
        """
        # Create your callable() here and return it
        # def overlap(x):
        #     r"""
        #     """
        #     pass
        def get_overlap(x, occs_array, deriv=None):
            apig = APIG(self.nocc_up, 2 * self.nbasis)
            apig.assign_params = x.reshape(self.nocc_up, self.nbasis)
            ovlp = np.empty(occs_array.shape[0], dtype=x.dtype)
            for idx, occ in enumerate(occs_array):
                # PyCI occ to FanPy Slater determinant
                temp = [i + self.nbasis for i in occ]
                occ = np.append(occ, temp, axis=0)
                sd = create(0, *occ)
                if deriv == None:
                    # Compute overlap with FanCI
                    ovlp[idx] = apig.get_overlap(sd)
                else:
                    # Compute delta_overlap with FanCI
                    ovlp[idx] = apig.get_overlap(sd, deriv=deriv)
            return ovlp

        return get_overlap
