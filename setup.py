# -*- coding: utf-8 -*-
# Copyright (c) 2019 UPennEoR
# Licensed under the MIT License

import os
import sys

from setuptools import setup
from distutils import spawn
import distutils.command.build as _build


req = [
    "numpy",
    "numba",
    "cffi",
    "astropy",
    "h5py",
    "scipy",
    "healpy",
    "pyuvdata",
    "ssht_numba @ git+git://github.com/UPennEoR/ssht_numba",
    "spin1_beam_model @ git+git://github.com/UPennEoR/spin1_beam_model",
]

req_gsm = ["pygsm @ git+git://github.com/telegraphic/PyGSM"]

req_all = req_gsm

req_dev = ["pytest", "sphinx", "bump2version"]

# make sure the fortran library is built before installing
class CustomBuild(_build.build):
    def run(self):
        cwd = os.getcwd()
        if spawn.find_executable("make") is None:
            sys.stderr.write("make is required to build this package.\n")
            sys.exit(-1)
        _source_dir = os.path.join(
            os.path.split(__file__)[0], "RIMEz", "dfitpack_wrappers"
        )
        try:
            os.chdir(_source_dir)
            spawn.spawn(["make", "clean"])
            spawn.spawn(["make"])
            os.chdir(cwd)
        except spawn.DistutilsExecError:
            sys.stderr.write("Error while building with make\n")
            sys.exit(-1)
        _build.build.run(self)


setup(
    name="RIMEz",
    description="Methods and input models for computing visibilities.",
    url="https://github.com/UPennEOR/RIMEz",
    author="Zachary Martinot",
    author_email="zmarti@sas.upenn.edu",
    packages=["RIMEz"],
    package_data={"RIMEz": ["dfitpack_wrappers/dfitpack_wrappers.so"]},
    zip_safe=False,
    install_requires=req,
    # fmt: off
    extras_require={
        "dev": req_dev + req_all,
        "gsm": req_gsm,
        "all": req_all
    },
    # fmt: on
    cmdclass={"build": CustomBuild},
    use_scm_version=True,
    setup_requires=["setuptools_scm"],
)
