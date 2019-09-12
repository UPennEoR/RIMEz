=====
RIMEz
=====

.. start-badges
.. image:: https://travis-ci.org/UPennEOR/RIMEz.svg
    :target: https://travis-ci.org/UPennEOR/RIMEz
.. image:: https://coveralls.io/repos/github/UPennEOR/RIMEz/badge.svg
    :target: https://coveralls.io/github/UPennEOR/RIMEz
.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/ambv/black
.. image:: https://readthedocs.org/projects/rimez/badge/?version=latest
    :target: https://rimez.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status
.. end-badges

**Radio Interferometric Measurement Equation solver**

Installation
============

Dependencies
------------
`RIMEz` has a child dependency on both `gfortran` and `fftw`. These can usually be
installed with your package manager. Note that `fftw` must be installed as a *shared*
library (i.e. if compiling yourself, use `--enable-shared`).

If you are a `conda` user, you may wish to manually install the following dependencies
with `conda` so they are not installed with `pip`::

    $ conda install numpy numba cffi gitpython h5py scipy

Then, installation should be as simple as `pip install .` from the top-level directory,
or `pip install git+git://github.com/UPennEoR/RIMEz`. If you installed `FFTW` to a
non-default location, then you can point to its location using the environment variable
`FFTW_PATH`, which should be the path to the `lib` folder, eg.:
`FFTW_PATH=/usr/lib pip install .`.


User Install
------------
If you just want to install `RIMEz` for general use, install with
`pip install git+git://github.com/UPennEOR/RIMEz`.

Developer Install
-----------------
If you are developing `RIMEz`, clone the repo, install the dependencies as above, and
then do (in the repo):

    $ pip install -e .[dev]
