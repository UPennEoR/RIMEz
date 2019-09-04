import numpy as np
import numba as nb

import pyssht
import ssht_numba as sshtn


@nb.njit
def make_sigma_tensor():
    sigma = np.zeros((2, 2, 4), dtype=nb.complex128)
    sigma[:, :, 0] = np.array([[1.0, 0], [0.0, 1.0]], dtype=nb.complex128)
    sigma[:, :, 1] = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=nb.complex128)
    sigma[:, :, 2] = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=nb.complex128)
    sigma[:, :, 3] = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=nb.complex128)
    return sigma


@nb.njit
def make_bool_sigma_tensor():
    bsigma = np.empty((2, 2, 4), dtype=nb.boolean)
    bsigma[:, :, 0] = np.array([[True, False], [False, True]])
    bsigma[:, :, 1] = np.array([[True, False], [False, True]])
    bsigma[:, :, 2] = np.array([[False, True], [True, False]])
    bsigma[:, :, 3] = np.array([[False, True], [True, False]])
    return bsigma


@nb.njit
def fast_approx_radec2altaz(ra, dec, R):
    p1 = np.cos(ra) * np.cos(dec)
    p2 = np.sin(ra) * np.cos(dec)
    p3 = np.sin(dec)

    p = np.stack((p1, p2, p3), axis=0)

    q = np.dot(R, p)
    # q.shape is (3, ra.size)

    # alt = np.arcsin(q[2])
    alt = np.arctan(q[2] / np.sqrt(q[0] ** 2.0 + q[1] ** 2.0))

    az = np.arctan2(q[1], q[0])
    inds = np.where(az < 0.0)[0]
    az[inds] = 2 * np.pi + az[inds]
    return q.T, alt, az


@nb.njit
def RIME_sum(J1, J2_conj, F1, F2_conj, S, sigma, bsigma):
    Np, Nstokes = S.shape
    V = np.zeros((2, 2), dtype=np.complex128)

    for n in range(Np):
        for a in range(2):
            for b in range(2):
                for c in range(2):
                    for d in range(2):
                        for g in range(Nstokes):
                            if bsigma[b, c, g] == True:
                                # note that the transpose of J2 is taken
                                V[a, d] += (
                                    F1[n]
                                    * J1[n, a, b]
                                    * sigma[b, c, g]
                                    * S[n, g]
                                    * J2_conj[n, d, c]
                                    * F2_conj[n]
                                )
    return V


def vec_psv_constructor(beam_funcs, compile_target="parallel"):
    @nb.guvectorize(
        "float64[:,:], float64[:], float64[:,:],\
                        int64[:,:], int64[:],\
                        float64[:,:,:], float64[:], float64[:], complex128[:,:,:,:]",
        "(c,c), (f), (a,c),(b,j),(a),(f,n,s),(n),(n)->(f,b,j,j)",
        nopython=True,
        fastmath=False,
        target=compile_target,
    )
    def vec_psv(R_i, nu_axis, r_axis, ant_pairs, ant_ind2beam_func, S, RA, dec, V_i):
        c = 299792458.0  # meter/second
        # S.shape is (Nfreq, Nsrc, 4)
        # RA.shape is (Nsrc,)
        # dec.shape is (Nsrc,)

        Nbl = ant_pairs.shape[0]
        beam_func_indices = np.unique(ant_ind2beam_func)

        sigma = make_sigma_tensor()
        bsigma = make_bool_sigma_tensor()

        # s is unit vector toward each source at alpha_i, in ENU basis
        # s.shape is (Nsrc,3)
        s, alt, phi = fast_approx_radec2altaz(RA, dec, R_i.T)

        s[:, 0] = np.sin(phi) * np.cos(alt)
        s[:, 1] = np.cos(phi) * np.cos(alt)
        s[:, 2] = np.sin(alt)

        v_inds = np.where(alt > 0.0)[0]

        # theta = np.pi/2. - alt
        #
        # s[:,0] = np.sin(phi)*np.sin(theta)
        # s[:,1] = np.cos(phi)*np.sin(theta)
        # s[:,2] = np.cos(theta)
        #
        # # find which sources are visible above the horizon
        # # v_inds = np.where(theta < np.pi)[0]
        # v_inds = np.where(theta < np.pi/2.)[0]

        if v_inds.size == 0:
            V_i = np.zeros((nu_axis.shape[0], Nbl, 2, 2), dtype=nb.complex128)

        else:
            s_v = s[v_inds]
            # theta_v = theta[v_inds]
            alt_v = alt[v_inds]
            phi_v = phi[v_inds]

            S_v = S[:, v_inds, :]

            tau_g_v = -2 * np.pi / c * np.dot(r_axis, s_v.T)
            # tau_g_v = -2*np.pi/c * r_dot_s_func(r_axis, s_v)

            for j in range(nu_axis.shape[0]):
                nu_j = nu_axis[j]
                S_j_v = S_v[j]

                #             phases = -2*np.pi*nu_j/c * np.dot(r_axis, s_v.T)
                phases = nu_j * tau_g_v
                F_arr = np.cos(phases) + 1j * np.sin(phases)

                J_arr = np.zeros(
                    (beam_func_indices.shape[0], alt_v.shape[0], 2, 2),
                    dtype=nb.complex128,
                )

                for i_bf in range(beam_func_indices.shape[0]):
                    index = beam_func_indices[i_bf]
                    J_arr[i_bf] = beam_funcs(index, nu_j, alt_v, phi_v)

                for k in range(Nbl):
                    p, q = ant_pairs[k]

                    F_p = F_arr[p]
                    F_q_conj = np.conj(F_arr[q])

                    ind_p = ant_ind2beam_func[p]
                    J_p = J_arr[ind_p]

                    ind_q = ant_ind2beam_func[q]
                    J_q_conj = np.conj(J_arr[ind_q])

                    V_i[j, k] = RIME_sum(
                        J_p, J_q_conj, F_p, F_q_conj, S_j_v, sigma, bsigma
                    )

    return vec_psv


def parallel_point_source_visibilities(
    rotations_axis,
    nu_axis,
    r_axis,
    ant_pairs,
    beam_funcs,
    ant_ind2beam_func,
    S,
    RA,
    dec,
):

    Nbl = ant_pairs.shape[0]
    V = np.zeros(
        (rotations_axis.shape[0], nu_axis.shape[0], Nbl, 2, 2), dtype=np.complex128
    )

    vec_point_source_visibilities = vec_psv_constructor(beam_funcs)

    vec_point_source_visibilities(
        rotations_axis, nu_axis, r_axis, ant_pairs, ant_ind2beam_func, S, RA, dec, V
    )

    return V


def vec_muv_constructor(beam_funcs, compile_target="parallel"):
    @nb.guvectorize(
        "float64, float64[:,:],\
                    int64[:,:], int64[:],\
                    complex128[:,:], float64[:,:], int64, int64, float64[:], complex128[:,:,:,:]",
        "(),(a,c),(b,j),(a),(n,s), (c,c),(),(),(m)->(b,m,j,j)",
        nopython=True,
        target=compile_target,
    )
    def vec_mmode_unpol_visibilities(
        nu, r_axis, ant_pairs, ant_ind2beam_func, Slm, R_0, Ls, Lb, dummy_maxis, V_nu_m
    ):

        c = 299792458.0
        sigma = make_sigma_tensor()
        bsigma = make_bool_sigma_tensor()

        Nbl = ant_pairs.shape[0]
        beam_func_indices = np.unique(ant_ind2beam_func)

        if Ls < Lb:
            Lm = Ls
        else:
            Lm = Lb

        beta_t_init, alpha_t_init = sshtn.mw_sample_positions(Lb)
        alpha_t, beta_t = sshtn.bad_meshgrid(alpha_t_init, beta_t_init)

        ra = alpha_t
        dec = np.pi / 2.0 - beta_t

        s, alt, az = fast_approx_radec2altaz(ra.flatten(), dec.flatten(), R_0.T)

        s[:, 0] = np.sin(az) * np.cos(alt)
        s[:, 1] = np.cos(az) * np.cos(alt)
        s[:, 2] = np.sin(alt)

        v_inds = np.where(alt > 0.0)[0]
        alt_v = alt[v_inds]
        az_v = az[v_inds]

        s_dot_r = np.dot(r_axis, s.T)
        phases = -2 * np.pi * nu / c * s_dot_r

        F_arr = np.cos(phases) + 1j * np.sin(phases)

        J_arr = np.zeros(
            (beam_func_indices.shape[0], alt.shape[0]) + (2, 2), dtype=nb.complex128
        )
        for i_bf in range(beam_func_indices.shape[0]):
            index = beam_func_indices[i_bf]
            J_i = J_arr[index]
            J_i[v_inds] = beam_funcs(index, nu, alt_v, az_v)

        for k in range(Nbl):
            p, q = ant_pairs[k]
            ind_p = ant_ind2beam_func[p]
            ind_q = ant_ind2beam_func[q]

            K_CI = np.zeros((2, 2) + ra.shape, dtype=nb.complex128)
            for i_th in range(ra.shape[0]):
                for i_ph in range(ra.shape[1]):
                    for i_a in range(2):
                        for i_b in range(2):
                            for i_c in range(2):
                                for i_d in range(2):
                                    if bsigma[i_b, i_c, 0] == True:
                                        h = i_a + 2 * i_d
                                        i_rav = (
                                            i_ph + ra.shape[1] * i_th
                                        )  # equiv to np.ravel_multi_index((i_th, i_ph), (Lb, 2*Lb-1), order='C')

                                        A_p_ab = (
                                            F_arr[p, i_rav]
                                            * J_arr[ind_p, i_rav, i_a, i_b]
                                        )
                                        A_q_conj_dc = np.conj(
                                            J_arr[ind_q, i_rav, i_d, i_c]
                                            * F_arr[q, i_rav]
                                        )

                                        K_CI[i_a, i_d, i_th, i_ph] += (
                                            A_p_ab * sigma[i_b, i_c, 0] * A_q_conj_dc
                                        )

            Klm_cI = np.empty((2, 2, Lb ** 2), dtype=nb.complex128)
            for i_a in range(2):
                for i_b in range(2):
                    K_CI_ab = K_CI[i_a, i_b]

                    Klm_ab = Klm_cI[i_a, i_b]

                    sshtn.mw_forward_sov_conv_sym(K_CI_ab, Lb, 0, Klm_ab)

            for i_a in range(2):
                for i_b in range(2):
                    for m in range(-Lm, Lm + 1):
                        i_m = m + Lm - 1
                        for el in range(abs(m), Lm):
                            index = sshtn.elm2ind(el, m)

                            V_nu_m[k, i_m, i_a, i_b] += Klm_cI[
                                i_a, i_b, index
                            ] * np.conj(Slm[index, 0])

    return vec_mmode_unpol_visibilities


def parallel_mmode_unpol_visibilities(
    nu_axis, r_axis, ant_pairs, beam_funcs, ant_ind2beam_func, Slm, R_0, Ls, Lb
):
    Nbl = ant_pairs.shape[0]
    if Ls < Lb:
        Lm = Ls
    else:
        Lm = Lb

    Vm = np.zeros((nu_axis.size, Nbl, 2 * Lm - 1, 2, 2), dtype=np.complex128)

    vec_mmode_unpol_visibilities = vec_muv_constructor(beam_funcs)

    # a dummy argument for guvectorize so that the m dimension of the output Vm
    # is in the input signature
    dummy_maxis = np.arange(-(Lm - 1), Lm, dtype=np.float64)

    vec_mmode_unpol_visibilities(
        nu_axis, r_axis, ant_pairs, ant_ind2beam_func, Slm, R_0, Ls, Lb, dummy_maxis, Vm
    )

    return Vm


@nb.njit
def mmode_unpol_visibilities(
    nu_axis, r_axis, ant_pairs, beam_funcs, ant_ind2beam_func, Slm, R_0, Ls, Lb
):
    c = 299792458.0
    sigma = make_sigma_tensor()
    bsigma = make_bool_sigma_tensor()

    Nbl = ant_pairs.shape[0]
    beam_func_indices = np.unique(ant_ind2beam_func)

    if Ls < Lb:
        Lm = Ls
    else:
        Lm = Lb

    Vm = np.zeros((nu_axis.size, Nbl, 2 * Lm - 1, 2, 2), dtype=nb.complex128)

    beta_t_init, alpha_t_init = sshtn.mw_sample_positions(Lb)
    alpha_t, beta_t = sshtn.bad_meshgrid(alpha_t_init, beta_t_init)

    ra = alpha_t
    dec = np.pi / 2.0 - beta_t

    s, alt, az = fast_approx_radec2altaz(ra.flatten(), dec.flatten(), R_0.T)

    s[:, 0] = np.sin(az) * np.cos(alt)
    s[:, 1] = np.cos(az) * np.cos(alt)
    s[:, 2] = np.sin(alt)

    v_inds = np.where(alt > 0.0)[0]
    alt_v = alt[v_inds]
    az_v = az[v_inds]

    # theta = np.pi/2. - alt

    # s[:,0] = np.sin(phi)*np.sin(theta)
    # s[:,1] = np.cos(phi)*np.sin(theta)
    # s[:,2] = np.cos(theta)

    # v_inds = np.where(theta < np.pi/2.)

    # theta_v = theta[v_inds]
    # phi_v = phi[v_inds]

    for i in range(nu_axis.size):
        nu_i = nu_axis[i]
        Slm_i = Slm[i]

        # s_dot_r.shape == (r_axis.shape[0],) + ra.shape
        # s_dot_r = np.reshape(np.dot(r_axis, s.T), (r_axis.shape[0],) + ra.shape)
        s_dot_r = np.dot(r_axis, s.T)
        phases = -2 * np.pi * nu_i / c * s_dot_r

        F_arr = np.cos(phases) + 1j * np.sin(phases)

        J_arr = np.zeros(
            (beam_func_indices.shape[0], alt.shape[0]) + (2, 2), dtype=nb.complex128
        )
        for i_bf in range(beam_func_indices.shape[0]):
            index = beam_func_indices[i_bf]
            J_i = J_arr[index]
            J_i[v_inds] = beam_funcs(index, nu_i, alt_v, az_v)

        for k in range(Nbl):
            p, q = ant_pairs[k]
            ind_p = ant_ind2beam_func[p]
            ind_q = ant_ind2beam_func[q]

            K_CI = np.zeros((2, 2) + ra.shape, dtype=nb.complex128)

            for i_th in range(ra.shape[0]):
                for i_ph in range(ra.shape[1]):
                    for i_a in range(2):
                        for i_b in range(2):
                            for i_c in range(2):
                                for i_d in range(2):
                                    if bsigma[i_b, i_c, 0] == True:
                                        h = i_a + 2 * i_d
                                        i_rav = (
                                            i_ph + ra.shape[1] * i_th
                                        )  # equiv to np.ravel_multi_index((i_th, i_ph), (Lb, 2*Lb-1), order='C')

                                        A_p_ab = (
                                            F_arr[p, i_rav]
                                            * J_arr[ind_p, i_rav, i_a, i_b]
                                        )
                                        A_q_conj_dc = np.conj(
                                            J_arr[ind_q, i_rav, i_d, i_c]
                                            * F_arr[q, i_rav]
                                        )

                                        K_CI[i_a, i_d, i_th, i_ph] += (
                                            A_p_ab * sigma[i_b, i_c, 0] * A_q_conj_dc
                                        )

            Klm_cI = np.empty((2, 2, Lb ** 2), dtype=nb.complex128)
            for i_a in range(2):
                for i_b in range(2):
                    K_CI_ab = K_CI[i_a, i_b]

                    Klm_ab = Klm_cI[i_a, i_b]

                    sshtn.mw_forward_sov_conv_sym(K_CI_ab, Lb, 0, Klm_ab)

            for i_a in range(2):
                for i_b in range(2):
                    for m in range(-Lm, Lm + 1):
                        i_m = m + Lm - 1
                        for el in range(abs(m), Lm):
                            index = sshtn.elm2ind(el, m)

                            Vm[i, k, i_m, i_a, i_b] += Klm_cI[
                                i_a, i_b, index
                            ] * np.conj(Slm_i[index, 0])
    return Vm


@nb.njit
def visiblity_dft_from_mmodes(era_axis, Vm):
    """
    Sum the fourier series which defines a visibility as a function of time.

    Parameters
    ----------
    era_axis : float 1d-array
        The differential Earth rotation angle (in radians) relative to a
        particular time for which the m-modes were computed.

    Vm : complex float 5d-array, shape (Nfreq, N_baseline, N_modes, 2, 2)
        The m-modes of a visibility function, output from
        `parallel_mmode_unpol_visibilities`.

    Returns
    -------
    V_dft : complex float 5d-array, shape (era_axis.size, Nfreq, N_baseline, 2, 2)
        The visibility time samples.
    """
    V_dft = np.zeros(
        (era_axis.size,) + Vm.shape[:2] + Vm.shape[3:], dtype=nb.complex128
    )

    Lm = (Vm.shape[2] + 1) / 2
    m_axis = np.arange(-Lm + 1, Lm)

    for i in range(era_axis.size):
        # f_kernel = np.exp(-1j*m_axis*(era_axis[i]))
        phases = m_axis * era_axis[i]
        f_kernel = np.cos(phases) - 1j * np.sin(phases)
        for j in range(Vm.shape[0]):
            for k in range(Vm.shape[1]):
                for a in range(2):
                    for b in range(2):
                        # V_dft[i,j,k,a,b] = np.sum(Vm[j,k,:,a,b] * f_kernel)
                        for n in range(m_axis.shape[0]):
                            V_dft[i, j, k, a, b] += Vm[j, k, n, a, b] * f_kernel[n]

    return V_dft


def parallel_visibility_dft_from_mmodes(era_axis, Vm, delta_t):
    """
    Parallelized version of `visibility_dft_from_mmodes`. This code structure is
    because the parallelized njit function cannot initialize the output array `V_dft`,
    it must be passed in. Not sure why...
    """
    if delta_t == 0.0:
        delta_t = 1e-20

    omega_e = (
        7.292115e-5
    )  # radian/second. Mean angular speed of earth, USNO Circular 179 page 16
    delta_era = omega_e * delta_t
    V_dft = np.zeros(
        (era_axis.size,) + Vm.shape[:2] + Vm.shape[3:], dtype=np.complex128
    )
    inner_parallel_visiblity_dft_from_mmodes(era_axis, Vm, V_dft, delta_era)

    return V_dft


@nb.njit(parallel=True, nogil=True)
def inner_parallel_visiblity_dft_from_mmodes(era_axis, Vm, V_dft, delta_era):

    Lm = (Vm.shape[2] + 1) / 2
    m_axis = np.arange(-Lm + 1, Lm)

    zero_ind = np.where(m_axis == 0)
    x = m_axis * delta_era / 2.0
    x[zero_ind] = 1e-20

    sinc = np.sin(x) / x

    for i in nb.prange(era_axis.size):

        phases = m_axis * era_axis[i]
        f_kernel = sinc * (np.cos(phases) - 1j * np.sin(phases))

        for j in range(Vm.shape[0]):
            for k in range(Vm.shape[1]):
                for a in range(2):
                    for b in range(2):
                        for n in range(m_axis.shape[0]):

                            V_dft[i, j, k, a, b] += Vm[j, k, n, a, b] * f_kernel[n]
    return


# this turns out to be slower than the njited-dft in the case I tested
# (~400 fourier modes -> ~8000 time samples), but I'll leave it here...
#
# import nfft
#
# def visibility_nfft_from_mmodes(era_axis, Vm):
#     x_axis = era_axis/(2*np.pi)
#
#     V_nfft = np.zeros((era_axis.size,) + Vm.shape[:2] + Vm.shape[3:], dtype=np.complex128)
#
#     for jj in range(Vm.shape[0]):
#         for kk in range(Vm.shape[1]):
#             for aa in range(2):
#                 for bb in range(2):
#                     V_nfft[:,jj,kk,aa,bb] = nfft.nfft(x_axis, Vm[jj,kk,:-1,aa,bb], tol=1e-10)
#
#     return V_nfft


def vectorize_vis_mat(vmat_in):
    V_vec = np.zeros((vmat_in.shape[0] + 1, 8), dtype=np.float64)

    f = [np.real, np.imag]
    u = [1, 1j]

    for i in range(2):
        for j in range(2):
            for k in range(2):
                n = k + 2 * (j + 2 * i)
                V_vec[:-1, n] = f[k](vmat_in[:, i, j])
                V_vec[-1, n] = f[k](vmat_in[0, i, j])

    return V_vec


def devectorize_vis_vec(vvec_in):
    Vmat = np.zeros((vvec_in.shape[0], 2, 2), dtype=np.complex128)
    f = [np.real, np.imag]
    u = [1, 1j]

    for i in range(2):
        for j in range(2):
            for k in range(2):
                n = k + 2 * (j + 2 * i)
                Vmat[:, i, j] += u[k] * vvec_in[:, n]

    return Vmat


def visibility_from_mmodes(Vm, era_axis, up_sampling=10):

    Lm = (Vm.shape[2] + 1) / 2
    m_axis = np.arange(-Lm + 1, Lm)

    N_padded_m = up_sampling * (Vm.shape[2] + 1) - 1
    padded_shape = Vm.shape[:2] + (N_padded_m,) + Vm.shape[3:]
    Vm_padded = np.zeros(padded_shape, dtype=np.complex128)

    N_fftup = N_padded_m
    L_fftup = (N_fftup + 1) / 2

    ind0 = m_axis[0] + (N_fftup - 1) / 2
    ind1 = m_axis[-1] + (N_fftup - 1) / 2

    ang_shift_up = np.pi - 2 * np.pi * (L_fftup - 1) / (2 * L_fftup - 1)
    # phase_shift_up = np.exp(-1j*m_axis*ang_shift_up)
    phases = m_axis * ang_shift_up
    phase_shift_up = np.cos(phases) - 1j * np.sin(phases)

    for j in range(Vm.shape[0]):
        for k in range(Vm.shape[1]):
            for a in range(2):
                for b in range(2):
                    Vm_padded[j, k, ind0 : ind1 + 1, a, b] = (
                        Vm[j, k, :, a, b] * phase_shift_up
                    )

    era_fftup_axis = (
        2 * np.pi * np.arange(-(N_fftup - 1) / 2, (N_fftup - 1) / 2 + 1) / N_fftup
    )
    era_fftup_axis += np.pi

    V_fftup = np.fft.fft(np.fft.ifftshift(Vm_padded, axes=2), axis=2)

    V_itp = np.zeros(
        (era_axis.size,) + Vm.shape[:2] + Vm.shape[3:], dtype=np.complex128
    )

    last_era_in = era_fftup_axis[-1] + np.diff(era_fftup_axis)[0]
    per_era_in = np.r_[era_fftup_axis, np.array([last_era_in])]

    for j in range(V_itp.shape[1]):
        for k in range(V_itp.shape[2]):

            V_temp = vectorize_vis_mat(V_fftup[j, k, :, :, :].squeeze())

            tck, _ = interpolate.splprep(
                V_temp.T, u=per_era_in, k=3, s=0.0, full_output=0, per=1
            )
            spl_evaled = np.array(interpolate.splev(era_axis, tck))

            V_itp[:, j, k, :, :] = devectorize_vis_vec(spl_evaled.T)

    return V_itp
