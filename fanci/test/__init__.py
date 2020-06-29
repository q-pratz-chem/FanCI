r"""
FanCI test module.

"""

from os import path


__all__ = [
    'find_datafile',
    ]


DATAPATH = path.join(path.abspath(path.dirname(__file__)), 'data/')


def find_datafile(filename):
    r"""
    Return the full path of a FanCI test data file.

    Parameters
    ----------
    filename : str
        Name of data file.

    Returns
    -------
    datafile : str
        Path to data file.

    """
    return path.abspath(path.join(DATAPATH, filename))
