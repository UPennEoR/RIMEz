import os

import git
import h5py

import numpy as np

import ssht_numba as sshtn

import utils
import rime_funcs
import beam_models
import sky_models

from RIMEz import __path__ as RIMEz_path
from ssht_numba import __path__ as ssht_numba_path
from spin1_beam_model import __path__ as spin1_beam_model_path

git_repo_paths = {
    'RIMEz': RIMEz_path[0],
    'ssht_numba': ssht_numba_path[0],
    'spin1_beam_model': spin1_beam_model_path[0],
}

repo_versions = {}
for repo_name in git_repo_paths:
    repo_path = git_repo_paths[repo_name]
    repo = git.Repo(repo_path, search_parent_directories=True)
    repo_versions[repo_name + '_commit_hash'] = repo.head.object.hexsha

class VisibilityCalculation(object):

    required_parameters = (
    'array_latitude',
    'array_longitude',
    'array_height',
    'initial_time_sample_jd',
    'integration_time',
    'frequency_samples_hz',
    'antenna_positions_meters',
    'antenna_pairs_used',
    'antenna_beam_function_map',
    'integral_kernel_cutoff',
    )

    def __init__(self, parameters=None, beam_func=None, Slm=None, restore_file_path=None):

        if parameters is not None:

            if beam_func is None or Slm is None:
                raise ValueError("beam_func and Slm inputs must be provided.")

            for req_key in self.required_parameters:
                if req_key in parameters:
                    pass
                else:
                    raise ValueError("Parameter '" + req_key + "' is missing from input parameters.")

            self.parameters = parameters

            for key in parameters:
                setattr(self, key, parameters[key])

            self.beam_func = beam_func
            self.Slm = Slm

            self.setup()

        elif restore_file_path is not None:
            self.load_visibility_calculation(restore_file_path)
            self.beam_func = None
            self.Slm = None

    def setup(self):

        array_location = utils.coords_to_location(
            self.array_latitude,
            self.array_longitude,
            self.array_height)

        jd0 = self.initial_time_sample_jd
        self.R_0 = utils.get_rotations_realistic_from_JDs(jd0, array_location)

        self.Lss, _ = sshtn.ind2elm(self.Slm.shape[1])

    def compute_fourier_modes(self):

        nu_axis = self.frequency_samples_hz
        r_axis = self.antenna_positions_meters
        ant_pairs = self.antenna_pairs_used
        ant_ind2beam_func = self.antenna_beam_function_map
        beam_func = self.beam_func
        Slm = self.Slm
        R_0 = self.R_0
        Lss = self.Lss
        Lb = self.integral_kernel_cutoff


        Vm = rime_funcs.parallel_mmode_unpol_visibilities(
                nu_axis,
                r_axis,
                ant_pairs,
                beam_func,
                ant_ind2beam_func,
                Slm,
                R_0,
                Lss,
                Lb,
        )

        self.Vm = 0.5*Vm

    def compute_time_series(self, time_sample_jds=None, integration_time=None):

        if time_sample_jds is not None:
            self.time_sample_jds = time_sample_jds
            self.parameters['time_sample_jds'] = time_sample_jds

        if integration_time is not None:

            if integration_time == 0:

                print "Input integration time is identically zero, changing to 1e-9 seconds."
                integration_time = 1e-9 # seconds

            self.integration_time = integration_time
            self.parameters['integration_time'] = integration_time

        if getattr(self, 'Vm', None) is None:
            self.compute_fourier_modes()

        era0 = utils.JD2era_tot(self.initial_time_sample_jd)
        era_axis = utils.JD2era_tot(self.time_sample_jds)

        delta_era_axis = era_axis - era0

        self.parameters['delta_era_axis'] = delta_era_axis

        if self.integration_time <= 1e-9:

            self.V = rime_funcs.parallel_visibility_dft_from_mmodes(delta_era_axis, self.Vm)

        else:

            raise ValueError("None-zero integration time not yet implmented.")

    def write_visibility_fourier_modes(self, file_path, overwrite=False):

        if getattr(self, 'Vm', None) is None:
            raise ValueError("No visibility data available for writing.")

        if overwrite is True and os.path.exists(file_path):
            os.remove(file_path)

        with h5py.File(file_path, 'w') as h5f:
            for key in self.parameters:
                h5f.create_dataset(key, data=self.parameters[key])

            h5f.create_dataset('Vm', data=self.Vm)

            for label in repo_versions:
                commit_hash_str = np.string_(repo_versions[label])
                h5f.create_dataset(label, data=commit_hash_str)

    def write_visibility_time_series(self, file_path):
        raise NotImplmentedError

    def load_visibility_calculation(self, file_path):

        with h5py.File(file_path, 'r') as h5f:
            self.parameters = {}
            for key in h5f.keys():
                setattr(self, key, h5f[key].value)
                if (key in ['V','Vm']) is not True:
                    self.parameters[key] = h5f[key].value

    def write_uvdata_time_series(self,
            uvdata_file_path,
            clobber=False,
            instrument='RIMEz calculation',
            telescope_name='probably HERA, but who knows?',
            history='left blank by user',
            object_name=''):

        if getattr(self, 'V', None) is None:
            raise ValueError("No visibility data available for writing.")

        uvd = utils.uvdata_from_sim_data(
                self.array_latitude,
                self.array_longitude,
                self.array_height,
                self.time_sample_jds,
                self.frequency_samples_hz,
                self.antenna_positions_meters,
                self.antenna_pairs_used,
                self.V,
                integration_time=self.integration_time,
                instrument=instrument,
                telescope_name=telescope_name,
                history=history,
                object_name=object_name,
        )

        uvd.write_uvh5(uvdata_file_path, clobber=clobber)
