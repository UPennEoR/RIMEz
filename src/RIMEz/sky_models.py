# -*- coding: utf-8 -*-
# Copyright (c) 2019 UPennEoR
# Licensed under the MIT License

import h5py
import healpy as hp
import numba as nb
import numpy as np
import ssht_numba as sshtn
from scipy import interpolate

from . import utils

try:
    import pygsm
except ImportError:
    pygsm = None


# simple made-up point source catalog generation, with GLEAM-ish dN/dS and spectral indices


def random_power_law(S_min, S_max, alpha, size=1):
    # for pdf(S) ~ S**(alpha-1)  for S in [S_min,S_max]
    r = np.random.random(size=size)
    a = S_min ** alpha
    b = S_max ** alpha
    return (a + (b - a) * r) ** (1.0 / alpha)


def generate_point_source_flux(Nsrc, F_min, F_max, gamma):
    # a distribution with dN/dS ~ S**(-gamma), with S in [F_min, F_max]
    return random_power_law(F_min, F_max, 1.0 - gamma, Nsrc)


def generate_point_source_catalog(Nsrc, seed, F_min=0.5, F_max=100.0, gamma=1.8):

    np.random.seed(seed)

    Flux_150 = generate_point_source_flux(Nsrc, F_min, F_max, gamma)

    RA = 2 * np.pi * np.random.uniform(0.0, 1.0, Nsrc)
    codec = np.arccos(2.0 * np.random.uniform(0.0, 1.0, Nsrc) - 1.0)
    dec = np.pi / 2.0 - codec

    # GLEAM-ish, from glancing at Fig. 16 of Hurley-Walker 2016
    spectral_indices = -0.8 + 0.2 * np.random.randn(Nsrc)

    catalog = {
        "RA": RA,
        "dec": dec,
        "Flux_150": Flux_150,
        "spectral_indices": spectral_indices,
    }
    return catalog


def sky_from_catalog(catalog, nu_axis):
    F_nu = (
        catalog["Flux_150"][None, :]
        * (nu_axis[:, None] / 150.0) ** catalog["spectral_indices"][None, :]
    )
    RA = catalog["RA"]
    dec = catalog["dec"]
    S = np.zeros(F_nu.shape + (4,), dtype=np.float)
    S[:, :, 0] = F_nu
    return S, RA, dec


@nb.njit
def spin0_spherical_harmonics(ell, theta, phi, delta):
    """
    Returns the 2*ell + 1 spin-0 spherical harmonics of order ell evaluated at (theta, phi).
    """

    m_axis = np.arange(-ell, ell + 1)
    phases = m_axis * phi

    Y_elm = np.sqrt((2.0 * ell + 1.0) / 4.0 / np.pi) * (
        np.cos(phases) + 1j * np.sin(phases)
    )
    Y_elm *= np.conj(sshtn.dl_m(ell, 0, theta, delta))
    return Y_elm


def point_sources_harmonics(flux_density, ra, dec, L, ell_min=0):
    """
    Compute the spherical harmonic coefficents for a set of point sources.

    The input data defines a discrete set of delta functions on the sphere, one
    at each point (dec[i], RA[i]), and a flux density for each frequency for
    each of those directions.

    Parameters
    ----------
    flux_density : float, 2d-array, shape (Nfreq, Nsrc)
        The flux in Jansky for each source at each frequency. The first axis
        indexes frequency channels, the second axis indexes sources i.e. `I[:,0]`
        is the flux density spectrum of the 0th source.

    ra : float, 1d-array, shape (Nsrc,)
        The right ascension of each source, in radians (angle in [0, 2*pi)).

    dec : float, 1d-array, shape (Nsrc,)
        The declination of each source, in radians (angle in [-pi/2, pi/2]).

    L : int
        The spatial bandlimit up to which harmonic coefficients will be computed.

    ell_min : int
        The starting order for the sequence of harmonic coefficients.

    Returns
    -------
    Ilm : complex 2d-array, shape (Nfreq, L**2)
        The harmonic coefficients, summed over sources.
    """

    ra = np.array(ra)
    dec = np.array(dec)

    delta = sshtn.generate_dl(np.pi / 2.0, L)

    Ilm = inner_point_source_harmonics(flux_density, ra, dec, L, ell_min, delta)
    return Ilm


@nb.njit
def inner_point_source_harmonics(flux_density, ra, dec, L, ell_min, delta):

    codec = np.pi / 2 - dec

    Ilm = np.zeros((flux_density.shape[0], L ** 2 - ell_min ** 2), dtype=np.complex128)

    for ell in range(ell_min, L):
        m = np.arange(-ell, ell + 1)
        indices = (
            sshtn.elm2ind(ell, m) - ell_min ** 2
        )  # shift indices incase ell_min is not zero

        for ii in range(ra.shape[0]):
            Ylm_conj_ii = np.conj(
                spin0_spherical_harmonics(ell, codec[ii], ra[ii], delta)
            )

            for kk in range(Ilm.shape[0]):
                for jj in range(m.size):

                    Ilm[kk, indices[jj]] += flux_density[kk, ii] * Ylm_conj_ii[jj]

    return Ilm


def threaded_point_sources_harmonics(flux_density, ra, dec, L, ell_min=0, n_blocks=2):
    """
    Same inputs/outputs as `point_sources_harmonics`.

    There has been no speed up beyond N_blocks=3, at which point the
    computation is ~2 times faster than running in a single thread. Something
    to do with memory and inefficient looping over `sshtn.dl_m`?
    """
    ra = np.array(ra)
    dec = np.array(dec)

    delta = sshtn.generate_dl(np.pi / 2.0, L)

    I_split = list(np.array_split(flux_density, n_blocks, axis=1))
    RA_split = list(np.array_split(ra, n_blocks))
    dec_split = list(np.array_split(dec, n_blocks))

    Ilm_split = np.zeros(
        (n_blocks, flux_density.shape[0], L ** 2 - ell_min ** 2), dtype=np.complex128
    )

    @nb.njit(nogil=True, parallel=True)
    def alt_inner_blocks(I_s, RA_s, dec_s, L, ell_min, delta, Ilm_s):
        N_blocks = len(RA_s)
        for nn in nb.prange(N_blocks):
            Ilm_s[nn] = inner_point_source_harmonics(
                I_s[nn], RA_s[nn], dec_s[nn], L, ell_min, delta
            )

        Ilm = np.sum(Ilm_s, axis=0)

        return Ilm

    Ilm = alt_inner_blocks(I_split, RA_split, dec_split, L, ell_min, delta, Ilm_split)

    return Ilm


# diffuse sky model generation
def hp2ssht_index(hp_flm_in, lmax=None):
    """
    Map from healpy indexed harmonic coefficients to ssht index.

    There is a little more to it than just rearanging indices, because the
    spherical coordinate conventions between healpy and ssht are different -
    the azimuthal coordinates have opposite handedness.
    """

    R_xflip = np.array([[-1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]])

    if lmax is None:
        lmax = hp.Alm.getlmax(hp_flm_in[0, :].size)

    hp_flm = np.copy(hp_flm_in)
    for i in range(hp_flm.shape[0]):
        hp.rotate_alm(hp_flm[i, :], matrix=R_xflip, lmax=lmax)

    L = lmax + 1

    ssht_flm = np.zeros((hp_flm.shape[0], L ** 2), dtype=np.complex)
    for el in range(L):
        for m in range(-el, el + 1):
            hp_ind = hp.Alm.getidx(lmax, el, abs(m))
            ssht_ind = sshtn.elm2ind(el, m)
            if m >= 0:
                ssht_flm[:, ssht_ind] = np.exp(1j * m * np.pi) * hp_flm[:, hp_ind]
            else:
                ssht_flm[:, ssht_ind] = (-1.0) ** (m) * np.conj(
                    np.exp(1j * m * np.pi) * hp_flm[:, hp_ind]
                )

    return ssht_flm


def diffuse_sky_model_from_GSM2008(nu_axis, smooth_deg=0.0, ssht_index=True):
    if pygsm is None:
        raise ImportError(
            "You need pygsm to use this function. "
            "Install RIMEz with `pip install .[gsm]`"
        )

    k_b = 1.38064852e-23  # joules/kelvin
    c = 299792458.0  # meters/second
    A_Jy = 1e26  # Jy / (Watt/meter^2/Hz)

    Jy_per_K = A_Jy * 2 * k_b * (nu_axis * 1e6 / c) ** 2.0

    R_g2c = utils.get_galactic_to_gcrs_rotation_matrix()

    gsm8 = pygsm.GlobalSkyModel(
        freq_unit="MHz", basemap="haslam", interpolation="cubic"
    )

    I_init = Jy_per_K[:, None] * gsm8.generate(nu_axis)

    lmax = 3 * 512 // 2
    Ilm_init = hp.map2alm(I_init, lmax=lmax, pol=False, use_pixel_weights=True)

    for i in range(Ilm_init.shape[0]):
        hp.rotate_alm(Ilm_init[i, :], matrix=R_g2c, lmax=lmax)

    if ssht_index:
        flm = hp2ssht_index(Ilm_init, lmax=lmax)

    else:
        flm = Ilm_init

    return flm


def diffuse_sky_model_egsm_preview(nu_axis):
    egsm_harmonics_file = "/users/zmartino/zmartino/eGSM_preview/egsm_harmonics.h5"
    with h5py.File(egsm_harmonics_file, "r") as h5f:
        freqs = h5f["freqs"].value
        Ilm_init = h5f["Ilm"].value

    # 5th order spline interpolation
    Ilm = np.zeros((nu_axis.size, Ilm_init.shape[1]), dtype=np.complex128)
    for ii in range(Ilm.shape[1]):

        tck_re = interpolate.splrep(
            freqs, Ilm_init[:, ii].real, k=5, s=0, full_output=0
        )
        tck_im = interpolate.splrep(
            freqs, Ilm_init[:, ii].imag, k=5, s=0, full_output=0
        )

        Ilm[:, ii] = np.array(interpolate.splev(nu_axis, tck_re)) + 1j * np.array(
            interpolate.splev(nu_axis, tck_im)
        )

    return Ilm


def rotate_sphr_coords(R, theta, phi):
    """
    Returns the spherical coordinates of the point specified by vp = R . v,
    where v is the 3D position vector of the point specified by (theta,phi) and
    R is the 3D rotation matrix that relates two coordinate charts.
    """
    rhx = np.cos(phi) * np.sin(theta)
    rhy = np.sin(phi) * np.sin(theta)
    rhz = np.cos(theta)
    r = np.stack((rhx, rhy, rhz))
    rP = np.einsum("ab...,b...->a...", R, r)
    thetaP = np.arccos(rP[-1, :])
    phiP = np.arctan2(rP[1, :], rP[0, :])
    phiP[phiP < 0] += 2.0 * np.pi
    return (thetaP, phiP)


def linear_interp_rotation(hmap, R):
    """
    Performs a scalar rotation of the map relative to the Healpix coordinate
    frame by interpolating the map at the coordinates of new coordinate frame.
    """
    npix = len(hmap)
    nside = hp.npix2nside(npix)
    hpxidx = np.arange(npix)
    c, a = hp.pix2ang(nside, hpxidx)
    t, p = rotate_sphr_coords(R, c, a)
    return hp.get_interp_val(hmap, t, p)


# old thing
def diffuse_sky_model(nu_axis, R_g2c=None, ssht_index=True, smth_deg=0.0):
    if pygsm is None:
        raise ImportError(
            "You need pygsm to use this function. "
            "Install RIMEz with `pip install .[gsm]`"
        )

    if R_g2c is None:
        R_g2c = hp.rotator.Rotator(coord=["G", "C"]).mat

    gsm_low = pygsm.GlobalSkyModel2016(freq_unit="MHz", unit="MJysr", resolution="low")
    Jy_per_MJy = 1e6

    I_init = Jy_per_MJy * gsm_low.generate(nu_axis)

    nside = 64
    lmax = 3 * nside - 1

    Ilm_init = hp.map2alm(I_init, lmax=lmax, pol=False, use_pixel_weights=True)

    if smth_deg != 0.0:
        Ilm_init = hp.smoothalm(
            Ilm_init, fwhm=np.radians(smth_deg), pol=False, verbose=False, inplace=True
        )

    for i in range(Ilm_init.shape[0]):
        hp.rotate_alm(Ilm_init[i, :], matrix=R_g2c, lmax=lmax)

    if ssht_index:
        flm = hp2ssht_index(Ilm_init, lmax=lmax)
    else:
        flm = Ilm_init

    return flm
