# -*- coding: utf-8 -*-
# Copyright (c) 2019 UPennEoR
# Licensed under the MIT License

import numpy as np
import spin1_beam_model
import ssht_numba

import RIMEz
from RIMEz import management


def test_get_versions():
    """Test _get_versions function"""
    rimez_version = RIMEz.__version__
    ssht_numba_version = ssht_numba.__version__
    spin1_beam_model_version = spin1_beam_model.__version__
    repo_versions = management._get_versions()

    assert repo_versions["RIMEz"] == rimez_version
    assert repo_versions["ssht_numba"] == ssht_numba_version
    assert repo_versions["spin1_beam_model"] == spin1_beam_model_version

    return
