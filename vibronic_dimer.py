"""
This implements the model in O'Rielly and Olaya-Castro Nat. Comms. 2013
with GKSL dissipation
"""

import qutip as qt
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns  # just for a colour palette

plt.rc('text', usetex=True)
plt.rc('font', family='serif')


def make_system_operator(operator, n_vib_modes=4):

    if operator.dims != [[2], [2]]:
        raise Exception('System operator wrong shape')

    operator = qt.tensor(operator, qt.qeye(n_vib_modes))

    return operator


def make_vibrational_operator(operator):
    return qt.tensor(qt.qeye(2), operator)


def build_hamiltonian(g, omega_vib, n_vib_modes=4):
    """See Eq (4) O'Rielly and Olaya-Castro Nat. Comms. 2013"""
    # parameters
    delta_eps = 1042.
    v = 92.

    # make full space operators
    sz = make_system_operator(qt.sigmaz(), n_vib_modes)
    sx = make_system_operator(qt.sigmax(), n_vib_modes)
    b = make_vibrational_operator(qt.destroy(n_vib_modes))

    # electronic 2x2

    h_el = delta_eps * sz + v * sx

    # vibronic

    h_v = omega_vib * b.dag() * b

    # coupling

    h_el_v = g / np.sqrt(2) * sz * (b + b.dag())

    return h_el + h_v + h_el_v


def build_jump_operators(omega_vib, beta, n_vib_modes):
    """ Pure dephasing of each site and thermal noise on vibrational mode"""

    def boltzmann(inv_t):
        return (np.exp(inv_t * omega_vib) - 1.) ** (-1)

    gamma_pd = 1. / (2. * np.pi * 0.03)
    gamma_th = 1. / (2. * np.pi * 0.03)
    eta = boltzmann(beta)

    k1 = np.sqrt(gamma_pd) * make_system_operator(qt.fock(2, 0) * qt.fock(2, 0).dag(), n_vib_modes)
    k2 = np.sqrt(gamma_pd) * make_system_operator(qt.fock(2, 1) * qt.fock(2, 1).dag(), n_vib_modes)
    drd = make_vibrational_operator(qt.destroy(n_vib_modes))
    b = np.sqrt(gamma_th * (eta + 1)) * drd
    bd = np.sqrt(gamma_th * eta) * drd.dag()

    jump_ops = [k1, k2, b, bd]

    return jump_ops


def dynamics():

    beta = 0.0048  # room temp in cm
    n_vib_modes = 4
    # hamiltonian
    g = 267.1
    omega_vib = 1111.

    h = build_hamiltonian(g, omega_vib, n_vib_modes)
    # jump operators
    j_ops = build_jump_operators(omega_vib, beta, n_vib_modes)

    # init state and observables of interest
    rho_0 = qt.tensor(qt.fock(2, 0) * qt.fock(2, 0).dag(), qt.thermal_dm(n_vib_modes, 0.5))
    p1 = make_system_operator(qt.fock(2, 0) * qt.fock(2, 0).dag(), n_vib_modes)
    p2 = make_system_operator(qt.fock(2, 1) * qt.fock(2, 1).dag(), n_vib_modes)
    el_coherence = make_system_operator(qt.fock(2, 1) * qt.fock(2, 0).dag(), n_vib_modes)
    el_vib_coherence = qt.tensor(qt.fock(2, 0) * qt.fock(2, 1).dag(),
                                 qt.fock(n_vib_modes, 1) * qt.fock(n_vib_modes, 0).dag())

    ops = [p1, p2, el_coherence, el_vib_coherence]
    # dynamics
    times = np.linspace(0., 50., 1500)

    result = qt.mesolve(h, rho_0, times, j_ops, ops)

    fig, ax = plt.subplots(1, 1)
    ax.plot(times, result.expect[0], label='Site 1')
    ax.plot(times, result.expect[1], label='Site 2')

    fig, ax = plt.subplots(1, 1)
    ax.plot(times, result.expect[2])
    ax.plot(times, result.expect[3], ':')
    ax.plot(times, result.expect[4], '--')
    plt.show()


def vibronic_resonance():

    beta = 0.0048  # room temp in cm
    n_vib_modes = 4
    # hamiltonian
    g = 267.1
    omega_vibs = [800., 1111., 1300.]
    fig, ax = plt.subplots(1, 1)
    fig2, ax2 = plt.subplots(1, 1)
    fig3, ax3 = plt.subplots(1, 1)
    cp = sns.color_palette('colorblind')
    for i, omega_vib in enumerate(omega_vibs):

        h = build_hamiltonian(g, omega_vib, n_vib_modes)
        # jump operators
        j_ops = build_jump_operators(omega_vib, beta, n_vib_modes)

        # init state and observables of interest
        rho_0 = qt.tensor(qt.fock(2, 0) * qt.fock(2, 0).dag(), qt.thermal_dm(n_vib_modes, 0.5))
        p1 = make_system_operator(qt.fock(2, 0) * qt.fock(2, 0).dag(), n_vib_modes)
        p2 = make_system_operator(qt.fock(2, 1) * qt.fock(2, 1).dag(), n_vib_modes)
        el_coherence = make_system_operator(qt.fock(2, 1) * qt.fock(2, 0).dag(), n_vib_modes)
        el_vib_coherence = qt.tensor(qt.fock(2, 0) * qt.fock(2, 1).dag(),
                                     qt.fock(n_vib_modes, 1) * qt.fock(n_vib_modes, 0).dag())

        ops = [p1, p2, el_coherence, el_vib_coherence]
        # dynamics
        times = np.linspace(0., 50., 1500)

        result = qt.mesolve(h, rho_0, times, j_ops, ops)

        ax.plot(times, result.expect[0], label=f'$\omega_v = {omega_vib}$' + 'cm$^{-1}$', c=cp[i])
        ax.plot(times, result.expect[1], c=cp[i])

        ax2.plot(times, np.abs(result.expect[2]), label=f'$\omega_v = {omega_vib}$' + 'cm$^{-1}$', c=cp[i])
        ax3.plot(times, np.abs(result.expect[3]), label=f'$\omega_v = {omega_vib}$' + 'cm$^{-1}$', c=cp[i])
    ax.legend()
    ax.legend()
    ax.legend()

    fsz = 22
    ax.set_xlabel('$t$', fontsize=fsz)
    ax2.set_xlabel('$t$', fontsize=fsz)
    ax3.set_xlabel('$t$', fontsize=fsz)
    ax.set_ylabel('Site populations', fontsize=fsz)
    ax2.set_ylabel('Site coherence', fontsize=fsz)
    ax3.set_ylabel('Vibronic coherence', fontsize=fsz)
    fig.tight_layout()
    fig2.tight_layout()
    fig3.tight_layout()

    plt.show()


if __name__ == "__main__":
    vibronic_resonance()
