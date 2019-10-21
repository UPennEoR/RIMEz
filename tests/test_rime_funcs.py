# -*- coding: utf-8 -*-
# Copyright (c) 2019 UPennEoR
# Licensed under the MIT License

import numba as nb
import numpy as np

from RIMEz import rime_funcs


def test_make_sigma_tensor():
    """Test make_sigma_tensor function"""
    sigma = rime_funcs.make_sigma_tensor()
    assert sigma.shape == (2, 2, 4)
    assert sigma.dtype == np.complex128
    assert np.allclose(
        sigma[:, :, 0], np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.complex128)
    )
    assert np.allclose(
        sigma[:, :, 1], np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
    )
    assert np.allclose(
        sigma[:, :, 2], np.array([[0.0, 1.00], [1.0, 0.0]], dtype=np.complex128)
    )
    assert np.allclose(
        sigma[:, :, 3], np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128)
    )
    return
