import os

import git
import h5py

import numpy as np

import ssht_numba as sshtn

from . import utils
from . import rime_funcs
from . import sky_models

# where does this come from?
from . import __path__ as RIMEz_path
from ssht_numba import __path__ as ssht_numba_path
from spin1_beam_model import __path__ as spin1_beam_model_path

git_repo_paths = {
    "RIMEz": RIMEz_path[0],
    "ssht_numba": ssht_numba_path[0],
    "spin1_beam_model": spin1_beam_model_path[0],
}

repo_versions = {}
for repo_name in git_repo_paths:
    repo_path = git_repo_paths[repo_name]
    repo = git.Repo(repo_path, search_parent_directories=True)
    repo_versions[repo_name + "_commit_hash"] = repo.head.object.hexsha


class VisibilityCalculation(object):

    required_parameters = (
        "array_latitude",
        "array_longitude",
        "array_height",
        "initial_time_sample_jd",
        "integration_time",
        "frequency_samples_hz",
        "antenna_positions_meters",
        "antenna_pairs_used",
        "antenna_beam_function_map",
        "integral_kernel_cutoff",
    )

    def __init__(
        self, parameters=None, beam_func=None, Slm=None, restore_file_path=None
    ):

        if parameters is not None:

            if beam_func is None or Slm is None:
                raise ValueError("beam_func and Slm inputs must be provided.")

            for req_key in self.required_parameters:
                if req_key in parameters:
                    pass
                else:
                    raise ValueError(
                        "Parameter '" + req_key + "' is missing from input parameters."
                    )

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
            self.array_latitude, self.array_longitude, self.array_height
        )

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
            nu_axis, r_axis, ant_pairs, beam_func, ant_ind2beam_func, Slm, R_0, Lss, Lb
        )

        self.Vm = 0.5 * Vm

    def compute_time_series(self, time_sample_jds=None, integration_time=None):

        if time_sample_jds is not None:
            self.time_sample_jds = time_sample_jds
            self.parameters["time_sample_jds"] = time_sample_jds

        if integration_time is not None:

            if integration_time == 0:

                print(
                    "Input integration time is identically zero, changing to 1e-15 seconds."
                )
                integration_time = 1e-15  # seconds

            self.integration_time = integration_time
            self.parameters["integration_time"] = integration_time

        if getattr(self, "Vm", None) is None:
            self.compute_fourier_modes()

        era0 = utils.JD2era_tot(self.initial_time_sample_jd)
        era_axis = utils.JD2era_tot(self.time_sample_jds)

        delta_era_axis = era_axis - era0

        self.parameters["delta_era_axis"] = delta_era_axis

        self.V = rime_funcs.parallel_visibility_dft_from_mmodes(
            delta_era_axis, self.Vm, self.integration_time
        )

    def write_visibility_fourier_modes(self, file_path, overwrite=False):

        if getattr(self, "Vm", None) is None:
            raise ValueError("No visibility data available for writing.")

        if overwrite is True and os.path.exists(file_path):
            os.remove(file_path)

        with h5py.File(file_path, "w") as h5f:
            for key in self.parameters:
                h5f.create_dataset(key, data=self.parameters[key])

            h5f.create_dataset("Vm", data=self.Vm)

            for label in repo_versions:
                commit_hash_str = np.string_(repo_versions[label])
                h5f.create_dataset(label, data=commit_hash_str)

    def write_visibility_time_series(self, file_path):
        raise NotImplementedError

    def load_visibility_calculation(self, file_path):

        with h5py.File(file_path, "r") as h5f:
            self.parameters = {}
            for key in h5f.keys():
                setattr(self, key, h5f[key].value)
                if (key in ["V", "Vm"]) is not True:
                    self.parameters[key] = h5f[key].value

    def to_uvdata(
        self,
        channel_width="derived",
        antenna_numbers="derived",
        antenna_names="derived",
        instrument="RIMEz calculation",
        telescope_name="probably HERA, but who knows?",
        history="",
        object_name="",
    ):

        if getattr(self, "V", None) is None:
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
            channel_width=channel_width,
            antenna_numbers=antenna_numbers,
            antenna_names=antenna_names,
            integration_time=self.integration_time,
            instrument=instrument,
            telescope_name=telescope_name,
            history=history,
            object_name=object_name,
        )

        return uvd

    def write_uvdata_time_series(
        self,
        uvdata_file_path,
        clobber=False,
        channel_width="derived",
        antenna_numbers="derived",
        antenna_names="derived",
        instrument="RIMEz calculation",
        telescope_name="probably HERA, but who knows?",
        history="",
        object_name="",
    ):

        uvd = self.to_uvdata(
            channel_width=channel_width,
            antenna_numbers=antenna_numbers,
            antenna_names=antenna_names,
            instrument=instrument,
            telescope_name=instrument,
            history=history,
            object_name=object_name,
        )

        uvd.write_uvh5(uvdata_file_path, clobber=clobber)

    # def write_uvdata_time_series(self,
    #         uvdata_file_path,
    #         clobber=False,
    #         instrument='RIMEz calculation',
    #         telescope_name='probably HERA, but who knows?',
    #         history='',
    #         object_name=''):
    #
    #     if getattr(self, 'V', None) is None:
    #         raise ValueError("No visibility data available for writing.")
    #
    #     uvd = utils.uvdata_from_sim_data(
    #             self.array_latitude,
    #             self.array_longitude,
    #             self.array_height,
    #             self.time_sample_jds,
    #             self.frequency_samples_hz,
    #             self.antenna_positions_meters,
    #             self.antenna_pairs_used,
    #             self.V,
    #             integration_time=self.integration_time,
    #             instrument=instrument,
    #             telescope_name=telescope_name,
    #             history=history,
    #             object_name=object_name,
    #     )
    #
    #     uvd.write_uvh5(uvdata_file_path, clobber=clobber)


class PointSourceSpectraSet(object):
    """
    Manages converting a discrete set of point sources to their
    spherical harmonic representation

    Parameters:
    ----------
    nu_mhz : ndarray
        1D float array of frequencies (units of MHz) at which flux density data
        is provided in the `I` array parameter.

    Iflux : ndarray
        2D float array of flux density data (units of Jy) with shape
        (Nfreq, Nsrc) where Nfreq is the length of `nu_mhz` and Nsrc is the
        length of the `RA` and `Dec` parameters

    RA : ndarray
        1D float array of Right Ascension coordinates (in radians) with length
        Nsrc, in the coordinate frame specified by the `coordinates` parameter.

    Dec : ndarray
        1D float array of Declination coordinates (in radians) with length Nsrc,
        in the coordinate frame specified by the `coordinates` parameter.

    coordinates : str
        A string identifying the frame of the `RA` and `Dec` coordinate parameters.
    """

    def __init__(
        self, nu_mhz=None, Iflux=None, RA=None, Dec=None, coordinates="GCRS", file_path=None
    ):
        if file_path is None:

            if any([x is None for x in [nu_mhz, Iflux, RA, Dec]]):
                raise ValueError(
                    "One of the inputs (nu_mhz, Iflux, RA, Dec) has not been provided."
                )

            self.nu_mhz = nu_mhz
            self.Iflux = Iflux
            self.RA = RA
            self.Dec = Dec

            self.coordinates = coordinates

            self.L = None
            self.Ilm = None

        else:
            self.file_path = file_path
            self.load_from_file()

    def generate_harmonics(self, L, N_blocks=1):
        """
        Compute the harmonic coefficients for this set of point sources up to
        the spatial bandlimit L. If harmonics have already been computed up to
        a limit L0, then the harmonics in the range [L0, L) are computed
        and appended.
        """

        if self.Ilm is None:
            self.L = L
            self.Ilm = sky_models.threaded_point_sources_harmonics(
                self.Iflux, self.RA, self.Dec, self.L, N_blocks=N_blocks
            )
        else:
            L0 = self.L
            self.L = L

            Ilm_new = sky_models.threaded_point_sources_harmonics(
                self.Iflux, self.RA, self.Dec, self.L, ell_min=L0, N_blocks=N_blocks
            )

            Ilm_init = np.zeros((self.Ilm.shape[0], self.L ** 2), dtype=np.complex128)
            Ilm_init[:, : L0 ** 2] = self.Ilm
            Ilm_init[:, L0 ** 2 :] = Ilm_new

            self.Ilm = np.copy(Ilm_init)

            del Ilm_init, Ilm_new

    def __add__(self, other):
        if not isinstance(other, PointSourceSpectraSet):
            raise ValueError(
                "Adding anything other than a PointSourceSpectraSet"
                "makes no sense and is not implemented"
            )

        if not np.allclose(self.nu_mhz, other.nu_mhz):
            raise ValueError("These sets have different frequency axes.")

        if self.coordinates != other.coordinates:
            raise ValueError(
                "The coordinate system of these sets are different"
                ", they must be the same to add."
            )
        if self.L != other.L:
            raise ValueError("The bandlimits of the two sets are different.")

        Iflux = np.concatenate((self.Iflux, other.Iflux), axis=1)
        RA = np.concatenate((self.RA, other.RA))
        Dec = np.concatenate((self.Dec, other.Dec))

        new = PointSourceSpectraSet(
            self.nu_mhz, Iflux, RA, Dec, coordinates=self.coordinates
        )

        if self.Ilm is not None and other.Ilm is not None:
            new.Ilm = self.Ilm + other.Ilm
            new.L = self.L

        return new

    def __radd__(self, other):

        return self.__add__(other)

    def save_to_file(self, file_path=None, overwrite=False):
        if file_path is None:
            if getattr(self, "file_path", None) is None:
                raise ValueError("No file path set, must provide an input file_path")

        else:
            self.file_path = file_path

        if os.path.exists(self.file_path):
            if not overwrite:
                raise ValueError("File exists and overwrite not set.")

            else:
                print("Overwriting file:", file_path)
                os.remove(file_path)

        else:
            pass

        with h5py.File(file_path, "w") as h5f:
            h5f.create_dataset("nu_mhz", data=self.nu_mhz)
            h5f.create_dataset("Iflux", data=self.Iflux)
            h5f.create_dataset("RA", data=self.RA)
            h5f.create_dataset("Dec", data=self.Dec)
            h5f.create_dataset("coordinates", data=np.string_(self.coordinates))

            if not (self.Ilm is None):
                h5f.create_dataset("Ilm", data=self.Ilm)
                h5f.create_dataset("L", data=self.L)

    def load_from_file(self):
        if self.file_path is None:
            raise ValueError("File path not set")

        with h5py.File(self.file_path, "r") as h5f:
            self.nu_mhz = h5f["nu_mhz"].value
            self.Iflux = h5f["Iflux"].value
            self.RA = h5f["RA"].value
            self.Dec = h5f["Dec"].value
            self.coordinates = h5f["coordinates"].value

            if "Ilm" in h5f.keys():
                self.Ilm = h5f["Ilm"].value
                self.L = h5f["L"].value
