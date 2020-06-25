r"""
FanCI test module.

"""

import os


__all__ = [
    "find_datafile",
    ]


def find_datafile(file_name):
    return os.path.join(os.path.dirname(__file__), file_name)
