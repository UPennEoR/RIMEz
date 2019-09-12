import os
import subprocess
from setuptools import setup, find_packages

# Make the extension. This is not the "standard" way to do it, but it works.
exec_dir = os.path.join(os.getcwd(), 'RIMEz', 'dfitpack_wrappers')
subprocess.call("make clean; make", cwd=exec_dir, shell=True)


req = [
    "numpy",
    "numba",
    "ssht_numba @ git+git://github.com/UPennEoR/ssht_numba",
    "pyssht @ git+git://github.com/UPennEoR/ssht"
    'cffi',
    'gitpython',
    'h5py',
    'scipy',
    'healpy',
    "spin1_beam_model @ git+git://github.com/UPennEoR/spin1_beam_model",
]

req_gsm = [
    "pygsm @ git+git://github.com/telegraphic/PyGSM"
]

req_all = req_gsm

req_dev = [
    'pytest',
    'sphinx',
    'bump2version'
]

setup(
    name='RIMEz',
    version='0.1.0',
    description='Methods and input models for computing visibilities.',
    url='https://github.com/UPennEOR/RIMEz',
    author='Zachary Martinot',
    author_email='zmarti@sas.upenn.edu',
    packages=find_packages(),
    package_data={'RIMEz': ['dfitpack_wrappers/dfitpack_wrappers.so']},
    zip_safe=False,
    install_requires=req,
    extras_require={
        'dev': req_dev + req_all,
        'gsm': req_gsm,
        'all': req_all,
    }
)
