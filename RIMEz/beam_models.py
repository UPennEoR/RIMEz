import numpy as np
import numba as nb
import ctypes

from spin1_beam_model.jones_matrix_field import AntennaFarFieldResponse

from dfitpack_numba import bispeu_nb


def model_data_to_spline_beam_func(full_file_name, nu_axis, L_synth=180, indexed=False):
    """
    Parameters:
    full_file_name: string : the full path to the file containing the spin1
        model data.
    nu_axis: numpy array of floats, frequencies in Hz.
    L_synth: the bandlimit of the maps used to derive the spline approximation.

    Returns:
    A numba.njit decorated function to evaluate the spline approximation of
    instrumental HERA Jones such that the elements are:

    """

    if np.any(nu_axis < 1e6):
        msg = (
            "Warning: input frequencies look like they might not be in units of Hz."
        )
        print(msg)

    AR = AntennaFarFieldResponse(full_file_name)
    AR.derive_symmetric_rotated_feed(rotation_angle_sign="positive")

    nu_axis_MHz = 1e-6 * nu_axis
    AR.compute_spatial_spline_approximations(nu_axis_MHz, L_synth=L_synth)

    E_coeffs = AR.E_spl_coeffs
    rE_coeffs = AR.rE_spl_coeffs

    tx = AR.xknots
    ty = AR.yknots
    kx = AR.kx
    ky = AR.ky

    spline_beam_func = construct_spline_beam_func(
        nu_axis, tx, ty, kx, ky, E_coeffs, rE_coeffs
    )

    if indexed:

        @nb.njit
        def spline_beam_funcs(i, nu, alt, az):
            if i == 0:
                jones_matrix = spline_beam_func(nu, alt, az)
            return jones_matrix

        return spline_beam_funcs

    else:

        return spline_beam_func


def construct_spline_beam_func(
    nu_axis, tx, ty, kx, ky, E_coeffs, rE_coeffs, imap="default"
):
    """
    Constructor of an njit-ed function to evaluate a Jones matrix.

    Parameters
    ----------
    nu_axis : real float array
        A 1d array of frequency sample points in MHz
    tx, ty : real float arrays
        The 2D spline knots obtained from scipy.interpolate.RectBivariateSpline.
        These knots are the same for each frequency.
    kx, ky : int
        The order of the spline in the x and y parameters, as defined by
        scipy.interpolate.RectBivariateSpline
    E_coeffs, rE_coeffs : real float array
        Arrays of spline coefficients, one for each antenna feed. Each array of
        coefficients has dimensions (Nfreq, 2, 2, 2, N_coeff) where
        Nfreq == nu_axis.size, and N_coeff the number of coeffcients that define
        a 2D spline with knots tx, ty and order kx, ky.

    Returns
    -------
    spline_beam_func : numba.njit wrapped function
        A function with inputs (nu, alt, az), where nu must be one of the
        elements of nu_axis (any other values may cause crashes!), and (alt, az)
        are each 1d arrays of angle coordinates, the Altitude
        -pi/2 <= alt <= pi/2 and Azimuth 0 <= az < 2*pi. This function returns
        a Jones matrix evaluated at the input points.

        NOTE: currently hard codeded to return E-field vector components in a
        basis aligned with intermediate RA/Dec

    """
    if imap == "default":
        imap = np.array([0, 1])
    elif imap == "alt":
        imap = np.array([1, 0])

    HERA_COLAT = 90.0 + 30.72152777777791
    R = rotation_matrix(np.array([0, 1.0, 0]), np.radians(HERA_COLAT))

    @nb.njit
    def spline_beam_func(nu, alt, az):
        i_nu = np.where(nu == nu_axis)[0][0]

        theta = np.pi / 2.0 - alt
        mu = np.cos(theta)
        phi = az_shiftflip(az)

        J_aa = np.zeros(theta.shape + (2, 2), dtype=nb.complex128)
        u = [-1.0, -1j]  # negative for components in alt/az basis
        #         u = [1., 1j]

        for kk in range(2):
            for aa in range(2):

                J_aa[..., imap[0], aa] += (
                    u[kk]
                    * bispeu_nb(tx, ty, E_coeffs[i_nu, aa, kk, :], kx, ky, mu, phi)[0]
                )
                J_aa[..., imap[1], aa] += (
                    u[kk]
                    * bispeu_nb(tx, ty, rE_coeffs[i_nu, aa, kk, :], kx, ky, mu, phi)[0]
                )

        # bleh, need to go back and check why the components have these signs
        # It can be discovered by computing all 4 possible dot products between
        # unit vectors in basis_transform_components. Something to do with
        # RA/dec being right-handed, Alt/Az being left-handed.
        cosX, sinX = basis_transform_components(alt, az, R)
        Umat = np.zeros((cosX.size, 2, 2), dtype=np.float64)
        Umat[:, 0, 0], Umat[:, 0, 1] = cosX, sinX
        Umat[:, 1, 0], Umat[:, 1, 1] = sinX, -cosX

        J_eq = apply_basis_transform(J_aa, Umat)

        tukeyW = tukey_window(alt, np.radians(2.0))
        for n in range(J_eq.shape[0]):
            J_eq[n] *= tukeyW[n]

        return J_eq

    return spline_beam_func


# a NJIT-able Bessel function of the first kind
J1_addr = nb.extending.get_cython_function_address("scipy.special.cython_special", "j1")
J1_functype = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double)
J1_fn = J1_functype(J1_addr)


@nb.vectorize("float64(float64)")
def njit_J1(x):
    return J1_fn(x)


@nb.njit
def airy_dipole(nu, alt, az, a):
    c = 299792458.0
    k = 2 * np.pi * nu / c
    ka = k * a
    calt, salt = np.cos(alt), np.sin(alt)
    caz, saz = np.cos(az), np.sin(az)

    J = np.zeros(alt.shape + (2, 2), dtype=nb.complex128)

    # multiply by 2*J_1(arg)/arg, J_1(x) is the bessel function
    # of the first kind

    arg = ka * calt
    zero_inds = np.where(arg == 0.0)[0]
    arg[zero_inds] = 1e-20
    G = 2.0 * njit_J1(arg) / arg
    #     G = np.ones_like(arg)

    # '00' <-> 'East,Alt', '01' <-> 'East,Az',
    # '10' <-> 'North,Alt', '11' <-> 'North,Az'
    J[..., 0, 0], J[..., 0, 1] = -saz * salt * G, caz * G
    J[..., 1, 0], J[..., 1, 1] = -caz * salt * G, -saz * G

    return J


@nb.njit
def gaussian_dipole(alt, az, a):

    salt = np.sin(alt)
    caz, saz = np.cos(az), np.sin(az)

    J = np.zeros(alt.shape + (2, 2), dtype=nb.complex128)

    # multiply by 2*J_1(arg)/arg, J_1(x) is the bessel function
    # of the first kind

    G = np.exp(-(np.pi / 2.0 - alt) ** 2.0 / 2.0 / a ** 2.0)

    # '00' <-> 'East,Alt', '01' <-> 'East,Az',
    # '10' <-> 'North,Alt', '11' <-> 'North,Az'
    J[..., 0, 0], J[..., 0, 1] = -saz * salt * G, caz * G
    J[..., 1, 0], J[..., 1, 1] = -caz * salt * G, -saz * G

    return J


@nb.njit
def heraish_beam_func(i, nu, alt, az):
    a = 7.0
    J_aa = airy_dipole(nu, alt, az, a)

    HERA_COLAT = 90.0 + 30.72152777777791
    R = rotation_matrix(np.array([0, 1.0, 0]), np.radians(HERA_COLAT))

    cosX, sinX = basis_transform_components(alt, az, R)
    Umat = np.zeros((cosX.size, 2, 2), dtype=np.float64)
    Umat[:, 0, 0], Umat[:, 0, 1] = cosX, sinX
    Umat[:, 1, 0], Umat[:, 1, 1] = sinX, -cosX

    # J_eq = np.einsum('...ab,...cb->...ac', J_aa, Umat)
    result = apply_basis_transform(J_aa, Umat)
    tukeyW = tukey_window(alt, np.radians(2.0))
    for n in range(result.shape[0]):
        result[n] *= tukeyW[n]
    return result


# #### basis transformation and misc.


@nb.njit
def altaz2cartENU(alt, az):
    E = np.sin(az) * np.cos(alt)
    N = np.cos(az) * np.cos(alt)
    U = np.sin(alt)
    return np.stack((E, N, U), axis=0)


@nb.njit
def cartENU2altaz(E, N, U):
    alt = np.arctan(U / np.sqrt(E ** 2.0 + N ** 2.0))
    az = np.arctan2(E, N)
    inds = np.where(az < 0.0)[0]

    az[inds] = az[inds] + 2.0 * np.pi
    return alt, az


@nb.njit
def alt_hat(alt, az):
    alth_E = -np.sin(az) * np.sin(alt)
    alth_N = -np.cos(az) * np.sin(alt)
    alth_U = np.cos(alt)
    return np.stack((alth_E, alth_N, alth_U), axis=0)


@nb.njit
def az_hat(alt, az):
    azh_E = np.cos(az)
    azh_N = -np.sin(az)
    azh_U = np.zeros_like(alt)
    return np.stack((azh_E, azh_N, azh_U), axis=0)


@nb.njit
def thetaphi2cartXYZ(theta, phi):
    X = np.cos(phi) * np.cos(theta)
    Y = np.sin(phi) * np.cos(theta)
    Z = np.sin(theta)
    return np.stack((X, Y, Z), axis=0)


@nb.njit
def cartXYZ2thetaphi(X, Y, Z):
    theta = np.arctan(Z / np.sqrt(X ** 2.0 + Y ** 2.0))
    phi = np.arctan2(Y, X)
    inds = np.where(phi < 0.0)[0]

    phi[inds] = phi[inds] + 2 * np.pi

    return theta, phi


@nb.njit
def theta_hat(theta, phi):
    thetah_x = -np.cos(phi) * np.sin(theta)
    thetah_y = -np.sin(phi) * np.sin(theta)
    thetah_z = np.cos(theta)

    return np.stack((thetah_x, thetah_y, thetah_z), axis=0)


@nb.njit
def phi_hat(theta, phi):
    phih_x = -np.sin(phi)
    phih_y = np.cos(phi)
    phih_z = np.zeros_like(theta)

    return np.stack((phih_x, phih_y, phih_z), axis=0)


@nb.njit
def rotation_matrix(axis, angle):
    """
    Rodrigues' rotation matrix formula
    """
    K = np.array(
        [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]]
    )

    Imat = np.identity(3)

    R = Imat + np.sin(angle) * K + (1.0 - np.cos(angle)) * np.dot(K, K)

    return R


@nb.njit
def basis_transform_components(alt, az, R):
    """
    Basis transform between an intermediate RA/Dec coordinates basis and
    an Alt/Az basis. The matrix R is the rotation matrix by an angle in (0,pi)
    about the +y-axis.
    """
    ENU_hat = altaz2cartENU(alt, az)

    temp = ENU_hat[0, :].copy()
    ENU_hat[0, :] = -ENU_hat[1, :]
    ENU_hat[1, :] = temp

    XYZ_hat = np.dot(R, ENU_hat)

    theta, phi = cartXYZ2thetaphi(XYZ_hat[0], XYZ_hat[1], XYZ_hat[2])

    R_flip = np.array([[0, 1.0, 0], [-1.0, 0, 0], [0, 0, 1.0]])

    alth = np.dot(R_flip.T, alt_hat(alt, az))
    azh = np.dot(R_flip.T, az_hat(alt, az))

    th = np.dot(R.T, theta_hat(theta, phi))

    cosX = np.sum(th * alth, axis=0)
    sinX = np.sum(th * azh, axis=0)

    return cosX, sinX


@nb.njit
def apply_basis_transform(Ma, Mb):
    """
    Acctually just 2x2 matrix multiplication, between two lists of matrices. The
    second matrix (Mb) is transposed. Intended usage is to apply a
    basis transformation matrix U to an instrumental jones matrix J.

    The Jones matrix transforms as: J -> J . U^T
    and this function performs that calculation when Ma[n] = J[n] and
    Mb[n] = U[n].
    """
    res = np.zeros_like(Ma)
    for n in range(Ma.shape[0]):
        for i_a in range(2):
            for i_b in range(2):
                for i_c in range(2):
                    res[n, i_a, i_c] += Ma[n, i_a, i_b] * Mb[n, i_c, i_b]
    return res


@nb.njit
def az_shiftflip(az_in):
    """
    Function to map between the azimuthal angle of Aliitude/Azimuth, where
    the azimuth is measured "east of north", and the azimuthal angle
    of a right-handed coordinate system measured north-of-east.

    This function is it's own inverse.
    """
    az_out = np.pi / 2.0 - az_in
    az_out[az_out < 0.0] += 2 * np.pi
    return az_out


@nb.njit
def tukey_window(alt, start_rad):
    """
    A Tukey tapering function over the upper hemisphere, as a function of the
    angle alt, which is 0 at the equator, pi/2 at the z-axis. The parameter
    start_rad is the anglular distance in radians above the equator for which
    the function is identically equal to 1 when alt > start_rad.
    """
    flat_inds = np.where(alt >= start_rad)[0]
    flat_indsC = np.where(alt < start_rad)[0]  # complment indices of flat_inds

    tw = np.zeros_like(alt)
    tw[flat_inds] = 1.0

    a = np.pi / 2.0 / start_rad

    for ind in flat_indsC:
        if alt[ind] >= 0.0:
            tw[ind] = np.sin(a * alt[ind]) ** 2.0
    return tw


#######
