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
``RIMEz`` has a child dependency on both ``gfortran`` and ``fftw``. These can
usually be installed with your package manager. For macOS users, ``gfortran`` is
available through ``conda``. If you have ``gfortran`` installed from another
source (like MacPorts or Homebrew), you may have to create a symlink called
``gfortran`` to compile successfully. Note that ``fftw`` must be installed as a
*shared* library (i.e. if compiling yourself, use ``--enable-shared``).

If you are a ``conda`` user, you may wish to manually install the following
dependencies with ``conda`` so they are not installed with ``pip``::

  $ conda install numpy numba cffi h5py scipy astropy

If you are happy to use the `conda-forge` channel, you can additionally do::

  $ conda install -c conda-forge pyuvdata healpy

Then, installation should be as simple as ``pip install .`` from the top-level
directory, or ``pip install git+git://github.com/UPennEoR/RIMEz``. Note that the
above manual installation of dependencies via `conda` is entirely optional -- simply
doing ``pip install .`` should work regardless.

If you installed ``FFTW`` to a non-default location, then you can point to its location
using the environment variable ``FFTW_PATH``, which should be the path to the
``lib`` folder, e.g.::

  $ FFTW_PATH=/usr/lib pip install .

User Install
------------
If you just want to install ``RIMEz`` for general use, install with::

  $ pip install git+git://github.com/UPennEOR/RIMEz

Developer Install
-----------------
If you are developing ``RIMEz``, clone the repo, install the dependencies as
above, and then do (in the repo)::

  $ pip install -e ".[all,dev]"


Optional Extras
---------------
There are a number of optional extras that can be installed along with `RIMEz`,
including `gsm` (which permits using the GSM as a sky model). To use all (user-focused)
optional extras, install with ``pip install ".[all]"``, otherwise you can pick and
choose by using a comma-separated list in the square brackets.

There are also a number of development-related groups of optional extras. If you are
developing, we recommend using *all* of them by installing the ``dev`` extra
(as specified above).

Development
===========
To install `RIMEz` for development, see above.

Testing
-------
To run tests locally, use
