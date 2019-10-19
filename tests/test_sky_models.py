# -*- coding: utf-8 -*-
# Copyright (c) 2019 UPennEoR
# Licensed under the MIT License

import numpy as np

from RIMEz import sky_models


def test_random_power_law():
    """Test random_power_law function"""
    # initialize RNG seed
    np.random.seed(1)

    # get a power law
    S_min = 1e1
    S_max = 1e2
    alpha = -2.7
    pl = sky_models.random_power_law(S_min, S_max, alpha)
    # for S_min=1e1, S_max=1e2, alpha=-2.7, we get 12.20579531
    assert np.isclose(pl[0], 12.20579531)
    assert pl.size == 1

    return
