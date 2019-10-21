# -*- coding: utf-8 -*-
# Copyright (c) 2019 UPennEoR
# Licensed under the MIT License

import numpy as np
from astropy import constants as const
from astropy import coordinates as coord
from astropy import units

from RIMEz import utils


def test_coords_to_location():
    """Test coords_to_location function"""
    array_lat = utils.HERA_LAT
    array_lon = utils.HERA_LON
    array_height = utils.HERA_HEIGHT
    location = utils.coords_to_location(array_lat, array_lon, array_height)
    assert isinstance(location, coord.EarthLocation)
    assert np.isclose(location.lat.rad, array_lat)
    assert np.isclose(location.lon.rad, array_lon)
    assert np.isclose(location.height.to("m").value, array_height)

    return


def test_kernel_cutoff_estimate():
    """Test kernel_cutoff_estimate"""
    max_bl = 875.0  # meters
    max_freq = 2.5e6  # hertz
    width_estimate = 100
    c_mks = const.c.to("m/s").value
    max_ell = 2 * np.pi * max_freq * max_bl / c_mks
    max_ell = int(np.ceil(max_ell + width_estimate))
    ell_cutoff = utils.kernel_cutoff_estimate(max_bl, max_freq, width_estimate)
    assert ell_cutoff == max_ell

    width_estimate = 101
    # need to add 2 to max_ell for values above because function always returns
    # an even value
    max_ell += 2
    ell_cutoff = utils.kernel_cutoff_estimate(max_bl, max_freq, width_estimate)
    assert ell_cutoff == max_ell

    return
