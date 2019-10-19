# -*- coding: utf-8 -*-
# Copyright (c) 2019 UPennEoR
# Licensed under the MIT License

import numpy as np

from RIMEz import beam_models


def test_theta_hat():
    """Test theta_hat function"""
    theta = np.asarray([np.pi / 4.0, np.pi / 3.0])
    phi = np.asarray([np.pi / 4.0, np.pi / 4.0])
    theta_hat = beam_models.theta_hat(theta, phi)

    theta_x = -np.cos(phi) * np.sin(theta)
    theta_y = -np.sin(phi) * np.sin(theta)
    theta_z = np.cos(theta)
    assert np.allclose(theta_x, theta_hat[0, :])
    assert np.allclose(theta_y, theta_hat[1, :])
    assert np.allclose(theta_z, theta_hat[2, :])

    return


def test_phi_hat():
    """Test phi_hat function"""
    theta = np.asarray([np.pi / 4.0, np.pi / 3.0])
    phi = np.asarray([np.pi / 4.0, np.pi / 4.0])
    phi_hat = beam_models.phi_hat(theta, phi)

    phi_x = -np.sin(phi)
    phi_y = np.cos(phi)
    phi_z = np.zeros_like(theta)
    assert np.allclose(phi_x, phi_hat[0, :])
    assert np.allclose(phi_y, phi_hat[1, :])
    assert np.allclose(phi_z, phi_hat[2, :])

    return
