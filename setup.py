# -*- coding: utf-8 -*-
"""
    Setup file for RIMEz.
    Use setup.cfg to configure your project.

    This file was generated with PyScaffold 3.2.3.
    PyScaffold helps you to put up the scaffold of your new Python project.
    Learn more under: https://pyscaffold.org/
"""
import distutils.command.build as _build
import os
import sys
from distutils import spawn
from pkg_resources import VersionConflict, require
from setuptools import setup

try:
    require("setuptools>=38.3")
except VersionConflict:
    print("Error: version of setuptools is too old (<38.3)!")
    sys.exit(1)


# make sure the fortran library is built before installing
class CustomBuild(_build.build):
    def run(self):
        cwd = os.getcwd()
        if spawn.find_executable("make") is None:
            sys.stderr.write("make is required to build this package.\n")
            sys.exit(-1)
        _source_dir = os.path.join(
            os.path.split(__file__)[0], "src", "RIMEz", "dfitpack_wrappers"
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


if __name__ == "__main__":
    setup(
        cmdclass={"build": CustomBuild},
        package_data={"RIMEz": ["dfitpack_wrappers/dfitpack_wrappers.so"]},
        use_pyscaffold=True,
    )
