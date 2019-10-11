# -*- coding: utf-8 -*-
# Copyright (c) 2019 UPennEoR
# Licensed under the MIT License

from . import management
from . import rime_funcs
from . import sky_models
from . import beam_models
from . import utils

from pkg_resources import get_distribution, DistributionNotFound

try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    # package is not installed
    pass
