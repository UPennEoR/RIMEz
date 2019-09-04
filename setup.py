import os
import subprocess
from setuptools import setup

exec_dir = os.path.join(os.getcwd(), "RIMEz", "dfitpack_wrappers")

subprocess.call("make clean; make", cwd=exec_dir, shell=True)

setup(
    name="RIMEz",
    description="Methods and input models for computing visibilities.",
    url="https://github.com/zacharymartinot/RIMEz",
    author="Zachary Martinot",
    author_email="zmarti@sas.upenn.edu",
    packages=["RIMEz"],
    package_data={"RIMEz": ["dfitpack_wrappers/dfitpack_wrappers.so"]},
    zip_safe=False,
)
