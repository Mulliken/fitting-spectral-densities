import numpy as np
import matplotlib.pyplot as plt
from common import correlation_func_sum, correlation_func_integral


def coth(x):
    return 1 / np.tanh(x)


print(coth(np.array([1, 2, 3]) * 1j))


def J(omega, pk, Omega_k, Gamma_k):
    return (
        np.pi
        / 2
        * pk
        * omega
        / ((omega + Omega_k) ** 2 + Gamma_k**2)
        / ((omega - Omega_k) ** 2 + Gamma_k**2)
    )


def a(t, pk, Omega_k, Gamma_k, beta):
    nu_k = 2 * np.pi * np.arange(1, 5000 + 1) / beta
    nu_k = nu_k[:, np.newaxis]
    term1 = (
        pk
        / (Omega_k * Gamma_k)
        * coth(beta * (Omega_k + 1j * Gamma_k) / 2)
        * np.exp((1j * Omega_k - Gamma_k) * t)
    )
    term2 = (
        pk
        / (Omega_k * Gamma_k)
        * coth(beta * (Omega_k - 1j * Gamma_k) / 2)
        * np.exp((-1j * Omega_k - Gamma_k) * t)
    )
    term3 = np.zeros_like(t, dtype=np.complex128)
    for nu in nu_k:
        term3 += 2j / beta * J(1j * nu, pk, Omega_k, Gamma_k) * np.exp(-nu * t)
    print(term3.shape)
    return term1 + term2 + term3


def b(t, pk, Omega_k, Gamma_k):
    return (
        1j
        * pk
        / (Omega_k * Gamma_k)
        * (np.exp(1j * Omega_k - Gamma_k * t) - np.exp(-1j * (Omega_k - Gamma_k) * t))
    )


# Parameters
n = 3
pk = 1
Omega_k = 0.1
Gamma_k = 1
beta = 1000000

# Time points
t_min = 0
t_max = 10
num_points = 132
t = np.linspace(t_min, t_max, num_points)

# Frequency points
omega_min = 0
omega_max = 10
num_omega_points = 1000
omega = np.linspace(omega_min, omega_max, num_omega_points)

# Calculate the correlation function
correlation_real = np.real(a(t, pk, Omega_k, Gamma_k, beta))
correlation_imag = np.imag(b(t, pk, Omega_k, Gamma_k))

correlation_sum_real = (
    np.pi
    / 2
    * correlation_func_sum(lambda w: J(w, pk, Omega_k, Gamma_k), 1000, [0, 100])(
        t, 1 / beta
    )
)

J_omega = J(omega, pk, Omega_k, Gamma_k)

# Plot the correlation function and spectral density function in subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))

ax1.plot(t, correlation_real.real, label="Real part")
ax1.plot(t, correlation_sum_real, label="Real part (sum)")
# ax1.plot(t, correlation_real / correlation_sum_real, label='Ratio')
# ax1.plot(t, correlation_imag, label='Imaginary part')
ax1.set_xlabel("Time")
ax1.set_ylabel("Correlation function")
ax1.set_title("Correlation Function")
ax1.legend()
ax1.grid(True)

ax2.plot(omega, J_omega)
ax2.set_xlabel("Frequency (ω)")
ax2.set_ylabel("J(ω)")
ax2.set_title("Spectral Density Function")
ax2.grid(True)

plt.tight_layout()
plt.savefig("lorentzian-correlation.png", dpi=300)
