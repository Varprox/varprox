import os
import re
from setuptools import setup, find_packages


def read(*paths):
    """Build a file path from *paths* and return the contents."""
    with open(os.path.join(*paths), 'r') as f:
        return f.read()


__version__ = re.search(
    r'__version__\s*=\s*[\'"]([^\'"]*)[\'"]',
    open('varprox/__init__.py').read()).group(1)

setup(
    # name of the package
    name='varprox',
    # You can specify all the packages manually or use the find_package
    # function
    packages=find_packages(),
    # See PEP440 for defining a proper version number
    version=__version__,
    # Small description of the package
    description='Optimization algorithms for solving penalized non-linear least squares problems',
    # Long description
    long_description=(read('README.rst') + '\n\n' +
                      read('AUTHORS.rst') + '\n\n'),
    # Project home page:
    url='https://github.com/Varprox/varprox',
    # license, author and author email
    license='GPLv3',
    author='Arthur Marmin and Frederic Richard',
    author_email='frederic.richard@univ-amu.fr',
    # If any packages contains data which are not python files, include them
    # package_data={'myapp': 'data'},
    install_requires=['numpy>=1.16.1', 'scipy>=1.16.1'],
    extras_require={'examples': ['pyafbf>=0.2.9', 'matplotlib>=3.10.0']},
    # classifiers is needed for uploading package on pypi.
    # The list of classifiers elements can be found at :
    # https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'Intended Audience :: End Users/Desktop',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering :: Mathematics',
    ],
    project_urls={
      'Documentation': 'https://Varprox.github.io/varprox/',
      'Source': 'https://github.com/Varprox/varprox/',
      'Tracker': 'https://github.com/Varprox/varprox/issues',
    },
    # What does your project relate to?
    keywords={'optimization', 'separable non linear least squares',
              'data fitting'},
    # Platforms on which the package can be installed:
    platforms='Linux, MacOSX, Windows',
    # To provide executable scripts, use entry points in preference to the
    # "scripts" keyword. Entry points provide cross-platform support and allow
    # pip to create the appropriate form of executable for the target platform.
    entry_points={
        'console_scripts': [
            'varprox=varprox',
        ],
    },
)