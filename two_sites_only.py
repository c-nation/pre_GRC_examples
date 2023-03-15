import qutip as qt
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
plt.rc('text', usetex=True)
plt.rc('font', family='serif')


def model(V=None, gamma_pd=None, gamma_opt=None, eps1=None, eps2=None):
    # parameters
    # hamiltonian
    if eps1 is None:
        eps1 = 18000
    if eps2 is None:
        delta_eps = 1042.
        eps2 = eps1 - delta_eps
    if V is None:
        V = 92.
    # environment
    if gamma_pd is None:
        gamma_pd = 1. / (2 * np.pi * 0.03)  #
    if gamma_opt is None:
        gamma_opt = 0.01 / (2 * np.pi * 0.03)
    #
    # hamiltonian - includes doubly excited state and the ground state, both are dynamically decoupled
    # in the absence of pumping and decay processes - though kept here to allow for these to be added

    h0 = eps2 * qt.tensor(qt.sigmap() * qt.sigmam(), qt.qeye(2)) + eps1 * qt.tensor(qt.qeye(2), qt.sigmap() * qt.sigmam())
    v = V * (qt.tensor(qt.sigmap(), qt.sigmam()) + qt.tensor(qt.sigmam(), qt.sigmap()))

    h = h0 + v

    # pure dephasing jump operators

    k1 = qt.tensor(qt.fock(2, 1) * qt.fock(2, 1).dag(), qt.qeye(2))  # |1><1| + |D><D|
    k2 = qt.tensor(qt.qeye(2), qt.fock(2, 1) * qt.fock(2, 1).dag())  # |2><2| + |D><D|
    sigma = qt.tensor(qt.qeye(2), qt.fock(2, 0) * qt.fock(2, 1).dag())

    jump_ops = [np.sqrt(gamma_pd) * k1, np.sqrt(gamma_pd) * k2, np.sqrt(gamma_opt) * sigma]
    # initial state and observables

    rho_0 = qt.tensor(qt.fock(2, 0) * qt.fock(2, 0).dag(), qt.fock(2, 1) * qt.fock(2, 1).dag())    # |1><1|

    excitons = h.eigenstates()
    e1 = excitons[1][1] * excitons[1][1].dag()
    e2 = excitons[1][2] * excitons[1][2].dag()
    ex_coherence = excitons[1][2] * excitons[1][1].dag()
    site_coherence = qt.tensor(qt.fock(2, 1) * qt.fock(2, 0).dag(), qt.fock(2, 0) * qt.fock(2, 1).dag())
    opers = [k1, k2, e1, e2, site_coherence, ex_coherence]

    return h, rho_0, jump_ops, opers


def single_plot():

    h, rho_0, jump_ops, opers = model()
    times = np.linspace(0., 5., 1000)

    result = qt.mesolve(h, rho_0, times, jump_ops, opers)

    # populations plot
    fsz = 22

    fig, ax = plt.subplots(1, 1)
    fig2, ax2 = plt.subplots(1, 1)
    figc, axc = plt.subplots(1, 1)
    figc2, axc2 = plt.subplots(1, 1)

    ax.plot(times, result.expect[0], label='Site 1')
    ax.plot(times, result.expect[1], label='Site 2')
    ax2.plot(times, result.expect[2])
    ax2.plot(times, result.expect[3])

    ax.set_xlabel('$t$', fontsize=fsz)
    ax2.set_xlabel('$t$', fontsize=fsz)

    ax.set_ylabel('Site Populations', fontsize=fsz)
    ax2.set_ylabel('Excitonic Populations', fontsize=fsz)

    # coherences plot

    axc.plot(times, np.abs(result.expect[4]))
    axc2.plot(times, np.abs(result.expect[5]))

    axc.set_xlabel('$t$', fontsize=fsz)
    axc2.set_xlabel('$t$', fontsize=fsz)

    axc.set_ylabel('Site Coherence', fontsize=fsz)
    axc2.set_ylabel('Excitonic Coherence', fontsize=fsz)

    ax.legend(fontsize=22)

    fig.tight_layout()
    figc.tight_layout()
    fig2.tight_layout()
    figc2.tight_layout()

    plt.show()


def v_dependence():

    Vs = [100, 200]

    fig, ax = plt.subplots(1, 1)
    fig2, ax2 = plt.subplots(1, 1)
    figc, axc = plt.subplots(1, 1)
    figc2, axc2 = plt.subplots(1, 1)
    times = np.linspace(0., 2., 1000)
    cs = sns.color_palette('colorblind')
    lss = ['-', '--']

    for i, v in enumerate(Vs):
        h, rho_0, jump_ops, opers = model(V=v, gamma_pd=1. / (2 * np.pi * 0.03))

        result = qt.mesolve(h, rho_0, times, jump_ops, opers)

        ax.plot(times / (2. * np.pi * 0.03), result.expect[0], c=cs[0], label=f'$V$ = {v}' + 'cm$^{-1}$', linestyle=lss[i])
        ax.plot(times / (2. * np.pi * 0.03), result.expect[1], c=cs[1], linestyle=lss[i])
        ax2.plot(times / (2. * np.pi * 0.03), result.expect[2], c=cs[0], linestyle=lss[i])
        ax2.plot(times / (2. * np.pi * 0.03), result.expect[3], c=cs[1], linestyle=lss[i])
        axc.plot(times / (2. * np.pi * 0.03), np.abs(result.expect[4]), c=cs[0], linestyle=lss[i])
        axc2.plot(times / (2. * np.pi * 0.03), np.abs(result.expect[5]), c=cs[0], linestyle=lss[i])

    fsz = 22
    ax.set_xlabel('$t$ [ps]', fontsize=fsz)
    ax2.set_xlabel('$t$ [ps]', fontsize=fsz)

    ax.set_ylabel('Site Populations', fontsize=fsz)
    ax2.set_ylabel('Excitonic Populations', fontsize=fsz)

    axc.set_xlabel('$t$ [ps]', fontsize=fsz)
    axc2.set_xlabel('$t$ [ps]', fontsize=fsz)

    axc.set_ylabel('Site Coherence', fontsize=fsz)
    axc2.set_ylabel('Excitonic Coherence', fontsize=fsz)

    ax.legend(fontsize=22)

    fig.tight_layout()
    figc.tight_layout()
    fig2.tight_layout()
    figc2.tight_layout()

    plt.show()


def rate_dependence():
    V = 92.
    gammas = np.array([0.1, 1., 1000000.])

    fig, ax = plt.subplots(1, 1)
    fig2, ax2 = plt.subplots(1, 1)
    figc, axc = plt.subplots(1, 1)
    figc2, axc2 = plt.subplots(1, 1)
    times = np.linspace(0., 4.2, 100000)
    cs = sns.color_palette('colorblind')
    lss = ['--', '-', ':']
    labels = ['Weak coupling', 'Intermediate coupling', 'Strong coupling']

    for i, gamma in enumerate(gammas):
        h, rho_0, jump_ops, opers = model(V=V, gamma_pd=gamma / (2 * np.pi * 0.03))

        result = qt.mesolve(h, rho_0, times, jump_ops, opers)

        ax.plot(times / (2. * np.pi * 0.03), result.expect[0], c=cs[0], label=labels[i], linestyle=lss[i])
        ax.plot(times / (2. * np.pi * 0.03), result.expect[1], c=cs[1], linestyle=lss[i])
        ax2.plot(times / (2. * np.pi * 0.03), result.expect[2], c=cs[0], linestyle=lss[i])
        ax2.plot(times / (2. * np.pi * 0.03), result.expect[3], c=cs[1], linestyle=lss[i])
        axc.plot(times / (2. * np.pi * 0.03), np.abs(result.expect[4]), c=cs[0], linestyle=lss[i])
        axc2.plot(times / (2. * np.pi * 0.03), np.abs(result.expect[5]), c=cs[0], linestyle=lss[i], label=labels[i])

    fsz = 22
    ax.set_xlabel('$t$ [ps]', fontsize=fsz)
    ax2.set_xlabel('$t$ [ps]', fontsize=fsz)

    ax.set_ylabel('Site Populations', fontsize=fsz)
    ax2.set_ylabel('Excitonic Populations', fontsize=fsz)

    axc.set_xlabel('$t$ [ps]', fontsize=fsz)
    axc2.set_xlabel('$t$ [ps]', fontsize=fsz)

    axc.set_ylabel('Site Coherence', fontsize=fsz)
    axc2.set_ylabel('Excitonic Coherence', fontsize=fsz)

    ax.legend(fontsize=22)
    axc2.legend(fontsize=22)

    fig.tight_layout()
    figc.tight_layout()
    fig2.tight_layout()
    figc2.tight_layout()

    plt.show()


if __name__ == "__main__":
    # single_plot()
    v_dependence()

    # rate_dependence()

