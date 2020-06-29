r"""
FanCI setup script.

Run `python setup.py --help` for help.

"""

from io import open

from setuptools import setup


name = 'fanci'


version = '0.0.0'


license = 'GPLv3'


author = ''


author_email = ''


url = ''


description = ''


long_description = open('README.rst', 'r', encoding='utf-8').read()


classifiers = [
    'Environment :: Console',
    'Intended Audience :: Science/Research',
    'Programming Language :: Python :: 2',
    'Programming Language :: Python :: 3',
    'Topic :: Science/Engineering :: Molecular Science',
    ]


packages = [
    'fanci',
    'fanci.test',
    ]


package_data = {
    'fanci.test': ['data/*.npy'],
    }


if __name__ == '__main__':

    setup(
        name=name,
        version=version,
        license=license,
        author=author,
        author_email=author_email,
        url=url,
        description=description,
        long_description=long_description,
        classifiers=classifiers,
        packages=packages,
        package_data=package_data,
        include_package_data=True,
        )
