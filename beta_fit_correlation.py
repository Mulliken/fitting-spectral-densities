import numpy as np
import matplotlib.pyplot as plt
from legendre_discretization import get_vn_squared, get_coups_sq, get_freqs
from common import (
    correlation_func_sum,
    _correlation_func_sum,
    correlation_func_integral,
)
from scipy.optimize import LinearConstraint, minimize, Bounds
from fcmaes import bitecpp as biteopt
from fcmaes import cmaescpp as cmaes

# def lorentz(ω, g, gamma, omega0):
#     return g**2 * (
#         gamma / 2 / ((gamma / 2) ** 2 + (ω - omega0) ** 2)
#         - gamma / 2 / ((gamma / 2) ** 2 + (ω + omega0) ** 2)
#     )


def ohmic(omega, alpha, omega_c):
    return alpha * omega ** 0.5 * np.exp(-omega / omega_c)


def lorentz(omega, g, gamma, omega0, occupation):
    # let beta = 2.3741537354439313 (target temperature 300K)
    
    return g**2 * (
        gamma / 2 / ((gamma / 2) ** 2 + (omega - omega0) ** 2)
        - gamma / 2 / ((gamma / 2) ** 2 + (omega + omega0) ** 2)
    )

def lorentz_(omega, g, gamma, omega0_div_gamma, occupation):
    return g**2 * (
        1 / 2 / ((1 / 2) ** 2 * gamma + (omega / gamma - omega0_div_gamma) ** 2)
        - 1 / 2 / ((1 / 2) ** 2 * gamma + (omega / gamma + omega0_div_gamma) ** 2)
    )

def lorentz_thermal(omega, g, gamma, omega0, occupation):
    beta_omega0 = np.log(1/occupation + 1)
    return g**2 * (
        gamma / 2 / ((gamma / 2) ** 2 + (omega - omega0) ** 2)
        - gamma / 2 / ((gamma / 2) ** 2 + (omega + omega0) ** 2)
    ) / np.tanh(beta_omega0 / 2) * np.tanh(omega / 2 / T)


def lorentz_thermal_(omega, g, gamma, omega0_div_gamma, occupation):
    beta_omega0 = np.log(1/occupation + 1)
    return g**2 * (
        1 / 2 / ((1 / 2) ** 2 * gamma + (omega / gamma - omega0_div_gamma) ** 2)
        - 1 / 2 / ((1 / 2) ** 2 * gamma + (omega / gamma + omega0_div_gamma) ** 2)
    ) / np.tanh(beta_omega0 / 2) * np.tanh(omega / 2 / T)


def approx_func(omega, params, lorentz_func):
    g = params[::4]
    gamma = params[1::4]
    omega0 = params[2::4]
    occupation = params[3::4]
    approx = lorentz_func(omega[:, np.newaxis], g, gamma, omega0, occupation).sum(axis=1)
    return approx


def corr_integral(ws, vs_squared, T, t_max):
    return vs_squared * (
        np.sin(ws * t_max) / np.tanh(ws / 2 / T) -
        1j * (1 - np.cos(ws * t_max))
    )

def objective_func_sd(params):
    approx_vs_squared_real = get_coups_sq(lambda omega: approx_func(omega, params, lorentz_func=lorentz_thermal), freqs, weights)
    approx_vs_squared_imag = get_coups_sq(lambda omega: approx_func(omega, params, lorentz_func=lorentz), freqs, weights)
    diff_imag = np.abs(target_vs_squared - approx_vs_squared_imag)
    diff_real = np.abs(target_vs_squared - approx_vs_squared_real)
    # overall diff should be equally weighted
    diff = diff_real + diff_imag
    # diff /= np.abs(target_vs_squared)
    # # divide by a gaussian centered at 1 with variance 0.1
    # diff /= np.exp(-((freqs - 1) ** 2) / 10)
    return diff.sum()


# Set the parameters
alpha = 0.1
omega_c = 10
n_grids = 2000
num_modes = 10
num_param = 4
freq_domain = (0, 5)
# g, gamma, omega0_div_gamma, occupation
bounds = np.array([(0, 20), (0.01, 2), (0.1, 2), (0.01, 5)] * num_modes)
bounds = Bounds(bounds[:, 0], bounds[:, 1])
lorentz = lorentz_
lorentz_thermal = lorentz_thermal_


T = 1 / 2.3741537354439313

objective_func = objective_func_sd

# Define the grids in the frequency domain
freqs, weights = get_freqs(n_grids, freq_domain)
target_vs_squared = get_coups_sq(lambda omega: ohmic(omega, alpha, omega_c), freqs, weights)
# target_corr = _correlation_func_sum(freqs, target_vs_squared)(t_grid, T)

initial_params = np.array([1, 2, 1, 5] * num_modes)


if __name__ == "__main__":
    # Optimize the parameters

    minimizer_kwargs = {
        "method": "SLSQP",
        "bounds": bounds,
        # "constraints": linear_constraint,
    }

    # Optimize the parameters
    from scipy.optimize import (dual_annealing,  basinhopping, direct, differential_evolution, shgo, brute)

    n_iter = 0
    def print_fun(x, f, accepted):
        global n_iter
        n_iter += 1
        print(f"iter {n_iter} at minimum %.4f accepted %d" % (f, int(accepted)))

    # result = basinhopping(objective_func, initial_params, niter=100, seed=None, T=100, stepsize=0.5, minimizer_kwargs=minimizer_kwargs, callback=print_fun)
    # result = dual_annealing(objective_func, bounds)
    result = biteopt.minimize(objective_func, bounds, M=16)
    # result = differential_evolution(objective_func, bounds, maxiter=2 * 10**4, popsize=20, mutation=(0.5, 1.5), tol=1e-6, atol=0, updating="deferred", workers=-1)
    # result = shgo(objective_func, bounds)
    # result = direct(objective_func, bounds, maxiter=int(1e8), maxfun=int(1e8), eps=0.5, locally_biased=False)
    # result = shgo(objective_func, bounds, workers=-1)
    optimized_params = result.x

    # Print the optimized parameters
    print("Optimized parameters:\n", result)
    for i in range(num_modes):
        param = optimized_params[i * num_param: (i + 1) * num_param]
        print(f"Basis {i+1:02d}: g={param[0]:>8.4f}, gamma={param[1]:>8.4f}, omega0={param[2]*param[1]:>8.4f}, ratio={param[2]:>8.4f}, occupation={param[3]:>8.4f}")
    # print(f"Optimized Correlation Function Value: {objective_func_corr(optimized_params):.6f}")
    print(f"Optimized Spectral Density Value    : {objective_func_sd(optimized_params):.6f}")

    # Create subplots
    fig,((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 12), sharex=False)


    omega_vals = np.linspace(0, freq_domain[1], 1000)
    def target_func(omega_vals): return ohmic(omega_vals, alpha, omega_c)
    def fitted_func(omega_vals): return approx_func(omega_vals, optimized_params, lorentz_thermal)
    ax1.plot(omega_vals, target_func(omega_vals), label="Target")
    ax1.plot(omega_vals, fitted_func(omega_vals), label="Fitted")
    for i in range(num_modes):
        param = optimized_params[i * num_param: (i + 1) * num_param]
        ax1.plot(omega_vals, lorentz(omega_vals, *param), "--", label=f"Lorentz {i+1:02d}")
    ax1.set_xlabel("$\omega$")
    ax1.set_ylabel("$J(\omega)$")
    # ax1.legend(loc='lower right')
    ax1.set_title("Spectral Densities from Re{C(t)}")
    ax1.grid(True)

    omega_vals = np.linspace(0, freq_domain[1], 1000)
    def fitted_func(omega_vals): return approx_func(omega_vals, optimized_params, lorentz)
    ax2.plot(omega_vals, target_func(omega_vals), label="Target")
    ax2.plot(omega_vals, fitted_func(omega_vals), label="Fitted")
    for i in range(num_modes):
        param = optimized_params[i * num_param: (i + 1) * num_param]
        ax2.plot(omega_vals, lorentz(omega_vals, *param),
                 "--", label=f"Lorentz {i+1:02d}")
    ax2.legend()
    ax2.set_xlabel("$\omega$")
    ax2.set_ylabel("$J(\omega)$")
    # ax2.legend(loc='lower right')
    ax2.set_title("Spectral Densities from Im{C(t)}")



    # Plot the correlation functions

    t_max = 12
    t_vals = np.linspace(0, t_max, 1000)
    target_corr = correlation_func_sum(lambda omega: ohmic(omega, alpha, omega_c), n_grids, freq_domain)
    fitted_corr_real = correlation_func_sum(lambda omega: approx_func(omega, optimized_params, lorentz_thermal), n_grids, freq_domain)
    fitted_corr_imag = correlation_func_sum(lambda omega: approx_func(omega, optimized_params, lorentz), n_grids, freq_domain)
    _ax3 = ax3.twinx()
    ax3.plot(t_vals, target_corr(t_vals, T).real, label="Target BCF real")
    ax3.plot(t_vals, fitted_corr_real(t_vals, T).real, label="Fitted BCF real")
    _ax3.plot(t_vals, target_corr(t_vals, T).imag, label="Target BCF imag", lw=4)
    _ax3.plot(t_vals, fitted_corr_imag(t_vals, T).imag, label="Fitted BCF imag")
    _ax3.set_ylabel(r"$\mathrm{Re}\{C(t)\}$")
    _ax3.legend()
    ax3.set_xlabel("Time $t$")
    ax3.set_ylabel(r"$\mathrm{Im}\{C(t)\}$")
    ax3_legend = ax3.legend(); ax3_legend.set_zorder(10)
    _ax3.legend()
    ax3.set_title("Bath Correlation Functions (BCF)")
    ax3.grid(True)

    omega_vals = np.linspace(0, 3*omega_c, 1000)
    ax4.plot(omega_vals, target_func(omega_vals), label="Target")
    ax4.plot(omega_vals, fitted_func(omega_vals), label="Fitted")
    for i in range(num_modes):
        param = optimized_params[i * num_param: (i + 1) * num_param]
        ax4.plot(omega_vals, lorentz(omega_vals, *param),
                 "--", label=f"Lorentz {i+1:02d}")
    ax4.legend()
    ax4.set_xlabel("$\omega$")
    ax4.set_ylabel("$J(\omega)$")
    # ax4.legend(loc='lower right')
    ax4.set_title(r"Spectral Densities over [0, 3$\omega_\mathrm{c}$]")

    plt.tight_layout()
    plt.savefig(f"beta_fitting_{freq_domain[0]}-{freq_domain[1]}.png", dpi=300, bbox_inches='tight')
