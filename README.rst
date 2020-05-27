=====
RIMEz
=====

.. start-badges
.. image:: https://github.com/UPennEoR/RIMEz/workflows/Tests/badge.svg
    :target: https://github.com/UPennEOR/RIMEz/actions
.. image:: https://codecov.io/gh/UPennEoR/RIMEz/branch/master/graph/badge.svg
    :target: https://codecov.io/gh/UPennEoR/RIMEz
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
There are a number of optional extras that can be installed along with ``RIMEz``,
including ``gsm`` (which permits using the GSM as a sky model). To use all (user-focused)
optional extras, install with ``pip install ".[all]"``, otherwise you can pick and
choose by using a comma-separated list in the square brackets.

There are also a number of development-related groups of optional extras. If you are
developing, we recommend using *all* of them by installing the ``dev`` extra
(as specified above).

Development
===========
To install ``RIMEz`` for development, see above.
We *strongly* recommend installing the provided pre-commit hooks the first time you
clone the repo::

  $ pre-commit install

This will allow linting checks (and auto-fixes!) to be performed automatically
whenever you commit.

Testing
-------
To run tests locally, use ``tox``. This is preferred over using ``pytest`` directly
because it also tests the package installation and setup. You can run a single
tox environment by using ``tox -e ENVNAME``. In particular, to run the linting
checks, use ``tox -e lint`` (however, these exact checks will be run when you commit,
if you have installed the pre-commit hooks, as above).

Versioning
----------
We use ``setuptools_scm`` for versioning the code. To create a new version, we recommend
creating a new dedicated branch to bump the version. On this branch, update the
``CHANGELOG.rst``, and make a commit with an associated git tag with the format
``vMAJOR.MINOR.PATCH``. Once merged into master, the new version will be active.
