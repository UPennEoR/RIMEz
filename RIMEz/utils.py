from functools import reduce

import numpy as np
from scipy import optimize, linalg
from astropy import coordinates as coord
from astropy import units
from astropy import _erfa
from astropy.time import Time

import healpy as hp
import pyssht
from spin1_beam_model.cst_processing import ssht_power_spectrum

import pyuvdata
from pyuvdata import UVData
from pyuvdata.utils import polstr2num

from beam_models import az_shiftflip

# misc.

HERA_LAT = np.radians(-30.72152777777791)
HERA_LON = np.radians(21.428305555555557)
HERA_HEIGHT = 1073.0000000093132  # meters


def coords_to_location(array_lat, array_lon, array_height):
    return coord.EarthLocation(
        lat=array_lat * units.rad,
        lon=array_lon * units.rad,
        height=array_height * units.meter,
    )


def kernel_cutoff_estimate(max_baseline_length_meters, max_freq_hz, width_estimate=100):
    c_mps = 299792458.0
    ell_peak_est = 2 * np.pi * max_freq_hz * max_baseline_length_meters / c_mps

    ell_cutoff = int(np.ceil(ell_peak_est + width_estimate))
    if ell_cutoff % 2 != 0:
        ell_cutoff += 1
    return ell_cutoff


# antenna array stuff

# This method of finding reducencies is aimed at the HERA hexagonal lattices
# since it first finds clusters in baseline length, then for each length
# group finds clusters in angle.
# Not sure how well it would do with general arrays. Might work?


def b_arc(b, precision=3):
    b = np.array(b)
    if b[0] == 0.0 and b[1] == 0.0:
        arc = np.nan
    else:
        if b[0] == 0.0:
            # i.e. b[1]/b[0] is -np.inf or np.inf
            arc = np.pi / 2.0
        else:
            b_grp = np.around(np.linalg.norm(b), precision)
            arc = np.around(b_grp * np.arctan(b[1] / b[0]), precision)
    return arc


def B(b, precision=3):
    return np.around(np.linalg.norm(b), precision)


def get_minimal_antenna_set(r_axis, precision=3):
    ant_inds = np.arange(0, r_axis.shape[0])

    a2u = {}
    for (ri, ai) in zip(r_axis, ant_inds):
        for (rj, aj) in zip(r_axis, ant_inds):
            b_v = ri - rj
            b, arc = B(b_v, precision=precision), b_arc(b_v, precision=precision)
            a2u[(ai, aj)] = (b, arc)

    u2a = {}
    for (ai, aj) in a2u:
        b, arc = a2u[(ai, aj)]
        if b not in u2a:
            u2a[b] = {}
        if arc not in u2a[b]:
            u2a[b][arc] = []
        u2a[b][arc].append((ai, aj))

    minimal_ant_pairs = []
    for b in u2a:
        for arc in u2a[b]:
            for (i, j) in u2a[b][arc]:
                ri = r_axis[i]
                rj = r_axis[j]

                bv_ij = ri - rj
                if bv_ij[0] >= 0.0:
                    if (bv_ij[1] < 0.0) and (bv_ij[0] == 0.0):
                        pass
                    else:
                        minimal_ant_pairs.append([i, j])
                        break

    return np.array(minimal_ant_pairs), u2a, a2u


def generate_hex_positions(lattice_scale=14.7, u_lim=3, v_lim=3, w_lim=3):
    """
    Generates antenna position on a hexagonal lattice.

    The lattice is centered at the origin coordinates so that there is always
    an antenna with coordinates [0,0,0]. The coordinate values intended to
    be in units of meters

    Parameters
    ----------
    lattice_scale : float
        The distance between any adjacent points in the lattice.

    u_lim, v_lim, w_lim : int
        The extent of the array in each hexagonal coordinate.

    Returns
    -------
    r_axis : ndarray, shape (N_antennas, 3)
        A list of coordinates for antennas in the array.
    """
    u_ang = np.radians(-30.0 + 30.0)
    v_ang = np.radians(210.0 + 30.0)
    w_ang = np.radians(90.0 + 30.0)

    e_u = np.array([np.cos(u_ang), np.sin(u_ang)])
    e_v = np.array([np.cos(v_ang), np.sin(v_ang)])
    e_w = np.array([np.cos(w_ang), np.sin(w_ang)])

    u_axis = np.arange(0, u_lim)
    v_axis = np.arange(0, v_lim)
    w_axis = np.arange(0, w_lim)

    r_vecs = []
    for u in u_axis:
        for v in v_axis:
            for w in w_axis:
                r_vecs.append(u * e_u + v * e_v + w * e_w)

    r_vecs = np.unique(np.around(r_vecs, 8), axis=0)
    r_axis = lattice_scale * np.append(r_vecs, np.zeros((r_vecs.shape[0], 1)), 1)

    return r_axis


# coordinates and visibility function parameters


def JD2era(JD):
    JD_time = Time(JD, format="jd", scale="ut1")
    era = _erfa.era00(JD_time.jd1, JD_time.jd2)
    return era


def JD2era_tot(JD):
    jd_time = Time(JD, format="jd", scale="ut1")
    # from USNO Circular 179, Eqn 2.10
    D_U = jd_time.jd - 2451545.0
    theta = 2 * np.pi * (0.7790572732640 + 1.00273781191135448 * D_U)
    return theta


def era2JD(era, nearby_JD):
    def f(jd):
        return era - JD2era_tot(jd)

    JD_out = optimize.newton(f, nearby_JD, tol=1e-10)
    return JD_out


def get_rotations_realistic(era_axis, JD_INIT, array_location):
    p1 = np.array([1.0, 0.0, 0.0])
    p2 = np.array([0.0, 1.0, 0.0])
    p3 = np.array([0.0, 0.0, 1.0])

    jd_axis = map(lambda era: era2JD(era, JD_INIT), era_axis)

    JDs = Time(jd_axis, format="jd", scale="ut1")

    rotations_axis = np.zeros((JDs.size, 3, 3), dtype=np.float)
    for i in range(JDs.size):
        gcrs_axes = coord.SkyCoord(
            x=p1 * units.one,
            y=p2 * units.one,
            z=p3 * units.one,
            location=array_location,
            obstime=JDs[i],
            frame="gcrs",
            representation="cartesian",
        )

        transf_gcrs_axes = gcrs_axes.transform_to("altaz")
        M = np.zeros((3, 3))
        M[:, 0] = transf_gcrs_axes.cartesian.x.value
        M[:, 1] = transf_gcrs_axes.cartesian.y.value
        M[:, 2] = transf_gcrs_axes.cartesian.z.value

        # the matrix M is generally not an orthogonal matrix
        # to working precision, but it is very close. This procedure
        # finds the orthogonal matrix that is nearest to M,
        # as measured by the Frobenius norm.
        Rt, _ = linalg.orthogonal_procrustes(M, np.eye(3))
        rotations_axis[i] = np.transpose(Rt)

    return rotations_axis, JDs.jd


def get_rotations_realistic_from_JDs(jd_axis, array_location):

    jd_axis = np.atleast_1d(jd_axis)

    p1 = np.array([1.0, 0.0, 0.0])
    p2 = np.array([0.0, 1.0, 0.0])
    p3 = np.array([0.0, 0.0, 1.0])

    #     jd_axis = map(lambda era: era2JD(era, JD_INIT), era_axis)

    JDs = Time(jd_axis, format="jd", scale="ut1")

    rotations_axis = np.zeros((JDs.size, 3, 3), dtype=np.float)
    for i in range(JDs.size):
        gcrs_axes = coord.SkyCoord(
            x=p1 * units.one,
            y=p2 * units.one,
            z=p3 * units.one,
            location=array_location,
            obstime=JDs[i],
            frame="gcrs",
            representation="cartesian",
        )

        transf_gcrs_axes = gcrs_axes.transform_to("altaz")
        M = np.zeros((3, 3))
        M[:, 0] = transf_gcrs_axes.cartesian.x.value
        M[:, 1] = transf_gcrs_axes.cartesian.y.value
        M[:, 2] = transf_gcrs_axes.cartesian.z.value

        # the matrix M is generally not an orthogonal matrix
        # to working precision, but it is very close. This procedure
        # finds the orthogonal matrix that is nearest to M,
        # as measured by the Frobenius norm.
        Rt, _ = linalg.orthogonal_procrustes(M, np.eye(3))
        rotations_axis[i] = np.transpose(Rt)

    return rotations_axis


def get_rotations_idealized(era_axis, array_location):
    Hlat = array_location.lat.rad
    Hlon = array_location.lon.rad

    R_hd2aa = np.array(
        [[-np.sin(Hlat), 0, np.cos(Hlat)], [0, -1, 0], [np.cos(Hlat), 0, np.sin(Hlat)]]
    )

    rotations_axis = np.zeros((era_axis.size, 3, 3))
    for i in range(era_axis.shape[0]):
        ang_i = era_axis[i] + Hlon
        R_ad2hd = np.array(
            [
                [np.cos(ang_i), np.sin(ang_i), 0],
                [np.sin(ang_i), -np.cos(ang_i), 0],
                [0, 0, 1],
            ]
        )

        R_ad2aa = np.dot(R_hd2aa, R_ad2hd)

        rotations_axis[i] = R_ad2aa.T

    return rotations_axis


def get_icrs_to_gcrs_rotation_matrix():
    p1 = np.array([1.0, 0.0, 0.0])
    p2 = np.array([0.0, 1.0, 0.0])
    p3 = np.array([0.0, 0.0, 1.0])

    radec_icrs = coord.SkyCoord(
        x=p1, y=p2, z=p3, frame="icrs", representation="cartesian"
    )

    radec_gcrs = radec_icrs.transform_to("gcrs")

    M = np.zeros((3, 3))
    M[:, 0] = radec_gcrs.cartesian.x.value
    M[:, 1] = radec_gcrs.cartesian.y.value
    M[:, 2] = radec_gcrs.cartesian.z.value

    Rt, _ = linalg.orthogonal_procrustes(M, np.eye(3))

    R = np.transpose(Rt)

    return R


def get_galactic_to_gcrs_rotation_matrix():
    x_c = np.array([1.0, 0, 0])  # unit vectors to be transformed by astropy
    y_c = np.array([0, 1.0, 0])
    z_c = np.array([0, 0, 1.0])

    axes_gcrs = coord.SkyCoord(
        x=x_c, y=y_c, z=z_c, frame="gcrs", representation="cartesian"
    )
    axes_gal = axes_gcrs.transform_to("galactic")
    axes_gal.representation = "cartesian"

    M = np.zeros((3, 3))
    M[:, 0] = axes_gal.cartesian.x.value
    M[:, 1] = axes_gal.cartesian.y.value
    M[:, 2] = axes_gal.cartesian.z.value

    # the matrix M is generally not an orthogonal matrix
    # to working precision, but it is very close. This procedure
    # finds the orthogonal matrix that is nearest to M,
    # as measured by the Frobenius norm.
    Rt, _ = linalg.orthogonal_procrustes(M, np.eye(3))

    R_g2gcrs = np.transpose(Rt)
    # R_g2gcrs = np.array(axes_gal.cartesian.xyz).T # The 3D rotation matrix that defines the coordinate transformation.
    return R_g2gcrs


# Things derived from beam functions


def beam_func_to_Omegas_pyssht(nu_hz, beam_func, L_use=200, beam_index=0):

    ttheta, pphi = pyssht.sample_positions(L_use, Method="MWSS", Grid=True)

    alt = np.pi / 2.0 - ttheta.flatten()
    az = pphi.flatten()

    v_inds = np.where(alt > 0.0)[0]

    alt_v = alt[v_inds]
    az_v = az[v_inds]

    Omega = np.zeros_like(nu_hz)
    Omegapp = np.zeros_like(nu_hz)

    for ii in range(nu_hz.size):
        nu_i = nu_hz[ii]

        J_i = np.zeros((alt.size, 2, 2), dtype=np.complex128)
        J_i[v_inds] = beam_func(beam_index, nu_i, alt_v, az_v)

        B = np.abs(J_i[:, 0, 0]) ** 2.0 + np.abs(J_i[:, 0, 1]) ** 2.0
        B = B.reshape(ttheta.shape)

        Blm = pyssht.forward(B, L_use, Spin=0, Method="MWSS", Reality=True)

        Omega[ii] = np.sqrt(4 * np.pi) * np.real(Blm[0])
        Omegapp[ii] = np.sum(np.abs(Blm) ** 2.0)

    return Omega, Omegapp


def beam_func_to_Omegas_healpix_sum(nu_axis, beam_func, nside=128, beam_index=0):
    npix = 12 * nside ** 2
    hpxidx = np.arange(npix)
    hbeta, halpha = hp.pix2ang(nside, hpxidx)

    halt = np.pi / 2.0 - hbeta

    Omega = np.zeros_like(nu_axis)
    Omegapp = np.zeros_like(nu_axis)

    for i in range(nu_axis.size):
        nu_i = nu_axis[i]

        J_i = beam_func(beam_index, nu_i, halt, halpha)
        nv_inds = np.where(halt <= 0.0)[0]

        B = np.abs(J_i[:, 0, 0]) ** 2.0 + np.abs(J_i[:, 0, 1]) ** 2.0
        B[nv_inds] = 0.0

        Omega[i] = (4.0 * np.pi / npix) * np.sum(B)
        Omegapp[i] = (4.0 * np.pi / npix) * np.sum(B ** 2.0)

    return Omega, Omegapp


def beam_func_to_kernel_power_spectrum(nu_hz, b_m, beam_func):
    """
    Compute the angular power spectrum of the Stokes-I -> Vokes-I
    integral kernel function.

    Parameters
    ----------
    nu_hz : float
        The frequency in Hz at which to evaluate the beam and fringe.
    b_m : float
        The baseline length in meters at which to evaluate the fringe.
    beam_func : function
        Function with inputs (nu, alt, az) which returns the Jones matrix
        of an antenna. Example: beam_models.airy_dipole

    Returns
    -------
    Cl_K00 : float 1d-array
        Angular power spectrum of the Stokes-I -> Vokes-I
        integral kernel function.

    Note: This can be used to get the angular power spectrum of just the beam
    by setting the baseline length to zero, or the angular power spectrum
    of just the fringe by inputing a beam_func that returns 1 everywhere.
    """

    c_mps = 299792458.0  # meter/second

    b_m = b_m * np.array([1.0, 0, 0])

    # approximate peak of the power spectrum from the baseline length and
    # frequency
    ell_peak_est = np.ceil(2 * np.pi * nu_hz / c_mps * np.linalg.norm(b_m)).astype(int)

    # L_use = 3*ell_peak_est/2
    L_use = 2 * ell_peak_est

    # for HERA-sized beam widths this will be sufficient. Significantly narrower
    # directivies may need to reconsider what this minimum should be
    if L_use < 350:
        L_use = 350

    theta, phi = pyssht.sample_positions(L_use, Method="MWSS", Grid=True)

    s = np.zeros(theta.shape + (3,))
    s[..., 0] = np.cos(phi) * np.sin(theta)
    s[..., 1] = np.sin(phi) * np.sin(theta)
    s[..., 2] = np.cos(theta)

    b_dot_s = np.sum(b_m * s, axis=-1)

    phase = 2 * np.pi * nu_hz / c_mps * b_dot_s
    fringe = np.exp(-1j * phase)

    alt = np.pi / 2.0 - theta.flatten()
    az = az_shiftflip(phi.flatten())

    v_inds = np.where(alt > 0.0)[0]

    alt_v = alt[v_inds]
    az_v = az[v_inds]

    jones = np.zeros((alt.size,) + (2, 2), dtype=np.complex128)
    jones[v_inds] = beam_func(nu_hz, alt_v, az_v)

    jones = jones.reshape(theta.shape + (2, 2))

    M00 = 0.5 * np.einsum("...ab,...ab", jones, jones.conj()).real

    K00 = M00 * fringe

    K00_lm = pyssht.forward(K00, L_use, Spin=0, Method="MWSS", Reality=False)

    Cl_K00 = ssht_power_spectrum(K00_lm)

    return Cl_K00


# UVData construction from sim parameters

# needed sim data:
# HERA_LOC v
# r_axis
# ant_pairs
# nu_axis
# jd_axis
# visibility data


def uvdata_from_sim_data(
    array_lat,
    array_lon,
    array_height,
    jd_axis,
    nu_axis,
    r_axis,
    ant_pairs,
    V_sim,
    integration_time="derived",
    channel_width="derived",
    antenna_numbers="derived",
    antenna_names="derived",
    instrument="left blank by user",
    telescope_name="left blank by user",
    history="left blank by user",
    object_name="left blank by user",
):

    HERA_LAT = array_lat
    HERA_LON = array_lon
    HERA_HEIGHT = array_height

    HERA_LAT_LON_ALT = (HERA_LAT, HERA_LON, HERA_HEIGHT)
    HERA_LOC = coord.EarthLocation(
        lat=HERA_LAT * units.rad,
        lon=HERA_LON * units.rad,
        height=HERA_HEIGHT * units.meter,
    )

    jd_obj = Time(jd_axis, format="jd", location=HERA_LOC)
    lst_axis = jd_obj.sidereal_time("apparent").radian
    if integration_time == "derived":
        del_jd = jd_obj[1] - jd_obj[0]
        integration_time = del_jd.sec

    uvd = UVData()

    uvd.telescope_location = HERA_LOC.value
    uvd.telescope_location_lat_lon_alt = HERA_LAT_LON_ALT

    if antenna_numbers == "derived":
        uvd.antenna_numbers = np.arange(r_axis.shape[0], dtype=np.int64)

        ant_num_pairs = ant_pairs

    else:
        uvd.antenna_numbers = antenna_numbers

        ant_num_pairs = np.array(
            [(antenna_numbers[ap[0]], antenna_numbers[ap[1]]) for ap in ant_pairs]
        )

    if antenna_names == "derived":
        uvd.antenna_names = [str(ant_ind) for ant_ind in uvd.antenna_numbers]
    else:
        uvd.antenna_names = antenna_names

    ant_pos_ECEF = pyuvdata.utils.ECEF_from_ENU(
        r_axis, HERA_LOC.lat.rad, HERA_LOC.lon.rad, HERA_LOC.height.value
    )
    uvd.antenna_positions = ant_pos_ECEF - uvd.telescope_location

    bls = [
        uvd.antnums_to_baseline(ant_num_pair[0], ant_num_pair[1])
        for ant_num_pair in ant_num_pairs
    ]

    uvd.freq_array = nu_axis.reshape((1, nu_axis.size))

    pols = ["xx", "yy", "xy", "yx"]
    uvd.polarization_array = np.array([polstr2num(pol_str) for pol_str in pols])

    uvd.x_orientation = "east"

    if channel_width == "derived":
        uvd.channel_width = nu_axis[1] - nu_axis[0]
    else:
        uvd.channel_width = channel_width

    uvd.Nfreqs = nu_axis.size
    uvd.Nspws = 1
    uvd.Npols = len(pols)

    bl_arr, lst_arr = np.meshgrid(np.array(bls), lst_axis)
    uvd.baseline_array = bl_arr.flatten()
    uvd.lst_array = lst_arr.flatten()

    _, time_arr = np.meshgrid(np.array(bls), jd_axis)
    uvd.time_array = time_arr.flatten()

    ant1_arr, _ = np.meshgrid(ant_num_pairs[:, 0], lst_axis)
    ant2_arr, _ = np.meshgrid(ant_num_pairs[:, 1], lst_axis)
    uvd.ant_1_array = ant1_arr.flatten()
    uvd.ant_2_array = ant2_arr.flatten()

    # Nants_data might be less than r_axis.shape[0] for redundant arrays
    uvd.Nants_data = len(np.unique(np.r_[ant1_arr.flatten(), ant2_arr.flatten()]))
    uvd.Nants_telescope = r_axis.shape[0]

    uvd.set_uvws_from_antenna_positions()

    uvd.Nbls = len(bls)
    uvd.Ntimes = lst_axis.size
    uvd.Nblts = bl_arr.size

    uvd.data_array = np.zeros(
        (uvd.Nblts, uvd.Nspws, uvd.Nfreqs, uvd.Npols), dtype=np.complex128
    )

    uvd.flag_array = np.zeros_like(uvd.data_array, dtype=np.bool)
    uvd.nsample_array = np.ones(
        (uvd.Nblts, uvd.Nspws, uvd.Nfreqs, uvd.Npols), dtype=np.float64
    )
    uvd.spw_array = np.ones(1, dtype=np.int64)
    uvd.integration_time = integration_time * np.ones(uvd.Nblts)

    uvd.phase_type = "drift"

    # (0,0) <-> 'xx' <-> East-East, etc.
    # matches order of uvd.polarization_array
    pol_map = {(0, 0): 0, (1, 1): 1, (0, 1): 2, (1, 0): 3}

    for i_a in range(2):
        for i_b in range(2):
            i_p = pol_map[(i_a, i_b)]

            for k in range(ant_num_pairs.shape[0]):
                a_i, a_j = ant_num_pairs[k]
                bl_num = uvd.antnums_to_baseline(a_i, a_j)
                bl_ind = np.where(uvd.baseline_array == bl_num)[0]

                uvd.data_array[bl_ind, 0, :, i_p] = np.copy(V_sim[:, :, k, i_a, i_b])

    uvd.vis_units = "Jy"

    uvd.instrument = instrument
    uvd.telescope_name = telescope_name
    uvd.history = history
    uvd.object_name = object_name

    uvd.check()

    return uvd


def inflate_uvdata_redundancies(uvd, red_gps):
    """
    red_gps is a list of lists of redundant groups of uvdata baseline numbers
    """
    #
    # antenna_positions, antenna_numbers = uvd.get_ENU_antpos()
    #
    # red_gps, centers, lengths = pyuvdata.utils.get_antenna_redundancies(
    #     antenna_numbers, antenna_positions, tol=tol, include_autos=include_autos)

    # number of baselines in the inflated dataset
    Nbls_full = reduce(lambda x, y: x + y, map(len, red_gps))

    # number of baselines = (Nants*(Nants+1))/2, including autos
    Nants = uvd.Nants_telescope
    assert Nbls_full == (Nants * (Nants + 1)) / 2

    # the baseline numbers in the compressed dataset
    bl_array_comp = uvd.baseline_array
    # strip away repeats for different times
    uniq_bl = np.unique(bl_array_comp)

    group_index, bl_array_full = zip(
        *[(i, bl) for i, gp in enumerate(red_gps) for bl in gp]
    )

    group_blti = []
    for gp in red_gps:
        for bl in gp:
            if bl in uniq_bl:
                group_blti.append(np.where(bl == bl_array_comp)[0])
                break
    group_blti = np.array(group_blti)  # shape (N_uniq_bl, Ntimes), transposed below

    Nblts_full = uvd.Ntimes * sum(map(len, red_gps))

    blt_map = np.transpose(group_blti)[..., group_index]
    blt_map = blt_map.flatten()
    assert Nblts_full == blt_map.size

    full_baselines = np.tile(bl_array_full, uvd.Ntimes)
    assert Nblts_full == full_baselines.size

    uvd.data_array = uvd.data_array[blt_map, ...]
    uvd.nsample_array = uvd.nsample_array[blt_map, ...]
    uvd.flag_array = uvd.flag_array[blt_map, ...]
    uvd.time_array = uvd.time_array[blt_map]
    uvd.lst_array = uvd.lst_array[blt_map]
    uvd.integration_time = uvd.integration_time[blt_map]
    uvd.uvw_array = uvd.uvw_array[blt_map, ...]

    uvd.baseline_array = full_baselines
    uvd.ant_1_array, uvd.ant_2_array = uvd.baseline_to_antnums(uvd.baseline_array)
    uvd.Nants_data = np.unique(uvd.ant_1_array.tolist() + uvd.ant_2_array.tolist()).size
    uvd.Nbls = np.unique(uvd.baseline_array).size
    uvd.Nblts = Nblts_full

    uvd.check()

    return
