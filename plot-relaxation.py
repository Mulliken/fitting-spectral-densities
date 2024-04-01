import numpy as np
import matplotlib.pyplot as plt

# Load data

pop_subOhmic = np.load("Leggett_s=0.5_A=0.10_cutoff10.npy")
pop_ohmic = np.load("Leggett_s=1.0_A=0.10_cutoff10.npy")
pop_superOhmic = np.load("Leggett_s=2.0_A=0.10_cutoff10.npy")

# polarization function based on the relaxation theory

def polarization(t, η, Δ):
    pz = np.exp(-t * η / 2) * (
        np.cos(t * np.sqrt(4 * Δ**2 - η**2) / 2) + 
        η / np.sqrt(4 * Δ**2 - η**2) * np.sin(t * np.sqrt(4 * Δ**2 - η**2) / 2)
    )
    return 0.5 * (pz + 1)

# define parameters
def ohmic(w, A, s, ν_c):
    return A * np.power(np.abs(w), s, dtype=float) * np.power(ν_c, 1-s, dtype=float) * np.exp(-np.abs(w) / ν_c) * np.sign(w)

ωc = 10
Δ = 1
β = 2.3741537354439313
sim_time = 12.8 * 2*np.pi
steps = 1280
tlist = np.linspace(0, sim_time, steps+1)

fig, ax = plt.subplots(1, 3, figsize=(12,9/3))
pops = [pop_subOhmic, pop_ohmic, pop_superOhmic]
for i, s in enumerate([0.5, 1.0, 2.0]):
    η = 0.5 * ohmic(Δ, A=0.1, s=s, ν_c=ωc) / np.tanh(β * Δ / 2)
    ax[i].plot(tlist/2/np.pi, pops[i], '-', label='DMRG')
    ax[i].plot(tlist/2/np.pi, polarization(tlist, η, Δ), '--', label='Relaxation theory')
    ax[i].set_xlabel('t')
    ax[i].set_ylabel('Donor pop. s= {}'.format(s))
    ax[i].set_title('s = {}'.format(s))
    ax[i].legend()

# Save the figure
plt.savefig("relaxation.png", dpi=300, bbox_inches='tight')

# plot the correlation function of sub-ohmic, ohmic and super-ohmic baths

from legendre_discretization import get_vn_squared, get_approx_func


def correlation_func(J, n, domain):
    import scipy.integrate as spi
    corr = lambda t, T: spi.quad(lambda w: J(w) * (np.cos(w*t) / np.tanh(w/2/T) - 1j * np.sin(w*t)), *domain)[0]
    return np.vectorize(corr)


def correlation_func_sum(J, n, domain):
    freq, vs_squared = get_vn_squared(J, n, domain)
    # select frquences within [0.85, 1.15]
    # selector = (freq > 0.85) & (freq < 1.15)
    # freq = freq[selector]
    # vs_squared = vs_squared[selector]
    # print(freq)

    corr_single = lambda v_squared, w, t, T:  v_squared * (np.cos(w*t) / np.tanh(w/2/T) - 1j * np.sin(w*t))
    corr = lambda t, T: np.sum(corr_single(vs_squared, freq, t, T))
    return np.vectorize(corr)

s_list = [0.01, 0.1, 0.5, 1.0, 2.0]

fig, ax = plt.subplots(2, len(s_list), figsize=(12*1.,9/2*1.))

cost_integration = 0
cost_summation = 0
# measure the cost of integration and summation
import time

for i, s in enumerate(s_list):
    domain = [0, 100]
    resonance_domain = [0.85, 1.15]
    n_modes = 10
    wlist = np.linspace(*domain, 1000)
    tlist = np.linspace(0, 0.5, 100)
    J = lambda w: ohmic(w, A=0.1, s=s, ν_c=ωc)
    corr_integral = correlation_func(J, n_modes, domain)
    corr_sum = correlation_func_sum(J, n_modes, domain)

    t_0 = time.time()
    ax[0][i].plot(tlist, np.real(corr_integral(tlist, 0.000001)), '-', label='Continuous')
    cost_integration += time.time() - t_0

    t_0 = time.time()
    ax[0][i].plot(tlist, np.real(corr_sum(tlist, 0.000001)), '--', label='Discretized')
    cost_summation += time.time() - t_0

    # ax[0][i].plot(tlist, np.imag(corr(tlist, 0.000001)), '--', label='Imaginary')
    # ax[0][i].set_ylim(-6, 20)
    ax[0][i].set_xlabel('t')
    ax[0][0].set_ylabel('Correlation Function $C(t)$')
    ax[0][i].set_title('s = {}'.format(s))
    ax[0][i].legend()

    J_approx = get_approx_func(J, n_modes, domain, 0.6)
    ax[1][i].plot(wlist, J(wlist), label='Continuous')
    ax[1][i].plot(wlist, J_approx(wlist), '--', label='Discretized')
    # ax[1][i].set_ylim(0, 0.6)
    ax[1][i].set_xlabel('$\omega$')
    ax[1][0].set_ylabel('$J(\omega)$')
    ax[1][i].legend()

# Save the figure
plt.savefig("correlation.png", dpi=300, bbox_inches='tight')
print("Integration cost: ", cost_integration)
print("Summation cost: ", cost_summation)

