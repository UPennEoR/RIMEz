#!/bin/bash
# This script is meant to be called by the "install" step defined in
# .travis.yml. See http://docs.travis-ci.com/ for more details.
# The behavior of the script is controlled by environment variabled defined
# in the .travis.yml in the top level folder of the project.
#
# This script is inspired by Scikit-Learn (http://scikit-learn.org/)

set -e

# Determine whether to install full env or just flake8
if [[ "$LINTER" != "true" ]]; then
    # Deactivate the travis-provided virtual environment and setup a
    # conda-based environment instead
    deactivate

    if [[ -f "$HOME/miniconda/bin/conda" ]]; then
        echo "Skip install conda [cached]"
    else
        # By default, travis caching mechanism creates an empty dir in the
        # beginning of the build, but conda installer aborts if it finds an
        # existing folder, so let's just remove it:
        rm -rf "$HOME/miniconda"

        # Use the miniconda installer for faster download / install of conda
        # itself
        wget http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh \
             -O miniconda.sh
        chmod +x miniconda.sh && ./miniconda.sh -b -p $HOME/miniconda
    fi
    export PATH=$HOME/miniconda/bin:$PATH
    # Make sure to use the most updated version
    conda update --yes conda

    # Configure the conda environment and put it in the path using the
    # provided versions
    # (prefer local venv, since the miniconda folder is cached)
    conda create -p ./.venv --yes python=${PYTHON_VERSION} pip virtualenv
    source activate ./.venv

    # Install dependencies
    conda install pytest pytest-runner coverage numpy numba cffi astropy h5py scipy
    conda install -c conda-forge healpy pyuvdata
    pip install git+https://github.com/UPennEoR/ssht_numba.git
    pip install git+https://github.com/UPennEoR/spin1_beam_model.git
    pip install pytest-cov codecov
else
    pip install flake8
fi
