import numpy as np
import matplotlib.pyplot as plt
from legendre_discretization import get_vn_squared, get_coups_sq, get_freqs
from common import (
    correlation_func_sum,
    _correlation_func_sum,
    correlation_func_integral,
)
from scipy.optimize import LinearConstraint, minimize


# def lorentz(ω, g, gamma, omega0):
#     return g**2 * (
#         gamma / 2 / ((gamma / 2) ** 2 + (ω - omega0) ** 2)
#         - gamma / 2 / ((gamma / 2) ** 2 + (ω + omega0) ** 2)
#     )


def ohmic(omega, alpha, omega_c):
    return alpha * omega ** 0.5 * np.exp(-omega / omega_c)


# define new lorentz function where gamma is 0.5*omega0


def lorentz_reduced(ω, g, omega0):
    gamma = 1 * omega0
    return g**2 * (
        gamma / 2 / ((gamma / 2) ** 2 + (ω - omega0) ** 2)
        - gamma / 2 / ((gamma / 2) ** 2 + (ω + omega0) ** 2)
    )


def lorentz_reduced_thermal(ω, g, omega0, occupation):
    beta_omega0 = np.log(1/occupation + 1)
    gamma = 0.5 * omega0
    return g**2 * (
        gamma / 2 / ((gamma / 2) ** 2 + (ω - omega0) ** 2)
        - gamma / 2 / ((gamma / 2) ** 2 + (ω + omega0) ** 2)
    ) / np.tanh(beta_omega0 / 2) * np.tanh(ω / 2 / T)


def approx_func_reduced(omega, params):
    g = params[::3]
    omega0 = params[1::3]
    occupation = params[2::3]
    approx = lorentz_reduced(omega[:, np.newaxis], g, omega0).sum(axis=1)
    return approx


def approx_func_reduced_thermal(omega, params):
    g = params[::3]
    omega0 = params[1::3]
    occupation = params[2::3]
    approx = lorentz_reduced_thermal(omega[:, np.newaxis], g, omega0, occupation).sum(axis=1)
    return approx


def corr_integral(ws, vs_squared, T, t_max):
    return vs_squared * (
        np.sin(ws * t_max) / np.tanh(ws / 2 / T) -
        1j * (1 - np.cos(ws * t_max))
    )


def objective_func_sd(params):
    approx_vs_squared_real = get_coups_sq(
        lambda omega: approx_func_reduced_thermal(omega, params), freqs, weights)
    approx_vs_squared_imag = get_coups_sq(
        lambda omega: approx_func_reduced(omega, params), freqs, weights)
    diff_real = np.abs(target_vs_squared - approx_vs_squared_real)
    diff_imag = np.abs(target_vs_squared - approx_vs_squared_imag)
    diff = diff_real + diff_imag
    diff /= np.abs(target_vs_squared)
    # # divide by a gaussian centered at 1 with variance 0.1
    # diff /= np.exp(-((freqs - 1) ** 2) / 10)
    return diff.sum()


# def objective_func_corr(params):
#     target_vs_squared = get_coups_sq(lambda omega: ohmic(
#         omega, alpha, omega_c), freqs, weights)
#     approx_vs_squared = get_coups_sq(lambda omega: approx_func(
#         omega, params), freqs, weights)
#     target_integral = corr_integral(freqs, target_vs_squared, T, t_max)
#     approx_integral = corr_integral(freqs, approx_vs_squared, T, t_max)

#     return np.abs(target_integral - approx_integral).sum()


# def objective_func_corr(params):
#     approx_vs_squared = get_coups_sq(
#         lambda omega: approx_func(omega, params), freqs, weights
#     )

#     # Evaluate the target and approximate correlation functions on the time grid
#     # target_corr = np.zeros_like(t_grid, dtype=complex)
#     # approx_corr = np.zeros_like(t_grid, dtype=complex)
#     # for i, t in enumerate(t_grid):
#     approx_corr = _correlation_func_sum(freqs, approx_vs_squared)(t_grid, T)

#     # Calculate the absolute (or squared) difference between the correlation functions
#     diff = np.abs(target_corr - approx_corr)
#     # Integrate the absolute (or squared) difference
#     integral_diff = np.dot(diff, exp_t_grid)

#     return integral_diff


# Set the parameters
alpha = 0.1
omega_c = 10
n = 500
num_modes = 10
num_param = 3
freq_domain = (0, 2)
bounds = [(0, 100), (0, freq_domain[1]*1.5), (0.001, 20)] * num_modes

T = 1 / 2.3741537354439313
t_max = 12
# Define a time grid for evaluating the correlation functions
t_grid = np.linspace(0, t_max, 1000)
inv_t_grid = np.exp(1 / (1 + t_grid))
exp_t_grid = np.exp(-5*t_grid)

# linear_constraint = ()
# objective_func = objective_func_corr
objective_func = objective_func_sd

# Define the grids in the frequency domain
freqs, weights = get_freqs(n, freq_domain)
target_vs_squared = get_coups_sq(
    lambda omega: ohmic(omega, alpha, omega_c), freqs, weights)
# target_corr = _correlation_func_sum(freqs, target_vs_squared)(t_grid, T)

initial_params = np.array([1, 2, 5] * num_modes)


if __name__ == "__main__":
    # Optimize the parameters

    minimizer_kwargs = {
        "method": "SLSQP",
        "bounds": bounds,
        # "constraints": linear_constraint,
    }

    # Optimize the parameters
    from scipy.optimize import (
        dual_annealing,
        basinhopping,
        direct,
        differential_evolution,
        shgo,
        brute,
    )

    # result = differential_evolution(objective_func, bounds, maxiter=2 * 10**4, popsize=20, mutation=(0.5, 1.5), tol=1e-6, atol=0, updating="deferred", workers=-1)
    # result = basinhopping(objective_func, initial_params, niter=1000, seed=None, T=100, stepsize=0.5, minimizer_kwargs=minimizer_kwargs)
    result = dual_annealing(objective_func, bounds)
    # result = shgo(objective_func, bounds)
    # result = direct(objective_func, bounds, maxiter=int(1e8), maxfun=int(1e8), eps=0.5, locally_biased=False)
    # result = shgo(objective_func, bounds, workers=-1)
    optimized_params = result.x

    # Print the optimized parameters
    print("Optimized parameters:\n", result)
    for i in range(num_modes):
        param = optimized_params[i * num_param: (i + 1) * num_param]
        print(
            f"Basis {i+1:02d}: g={param[0]:>8.4f}, omega0={param[1]:>8.4f}, occupation={param[2]:>8.4f}"
        )
    # print(f"Optimized Correlation Function Value: {objective_func_corr(optimized_params):.6f}")
    print(
        f"Optimized Spectral Density Value    : {objective_func_sd(optimized_params):.6f}")

    # Create subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

    # Plot the correlation functions

    t_vals = np.linspace(0, 4, 1000)
    target_corr = correlation_func_sum(
        lambda omega: ohmic(omega, alpha, omega_c), n, freq_domain
    )
    fitted_corr = correlation_func_sum(
        lambda omega: approx_func_reduced_thermal(omega, optimized_params), n, freq_domain
    )

    ax1.plot(t_vals, target_corr(t_vals, T).real,
             label="Target Bath Correlation Function")
    ax1.plot(t_vals, fitted_corr(t_vals, T).real,
             label="Fitted Bath Correlation Function")
    ax1.set_xlabel("Time $t$")
    ax1.set_ylabel("C(t)")
    ax1.legend()
    ax1.set_title("Bath Correlation Functions")
    ax1.grid(True)

    omega_vals = np.linspace(0, freq_domain[1], 1000)
    def target_func(omega_vals): return ohmic(omega_vals, alpha, omega_c)

    def fitted_func(omega_vals): return approx_func_reduced_thermal(omega_vals, optimized_params)

    ax2.plot(omega_vals, target_func(omega_vals), label="Target Function")
    ax2.plot(omega_vals, fitted_func(omega_vals), label="Fitted Function")
    for i in range(num_modes):
        param = optimized_params[i * num_param: (i + 1) * num_param]
        ax2.plot(
            omega_vals, lorentz_reduced_thermal(omega_vals, *param), "--", label=f"Lorentz {i+1:02d}"
        )

    ax2.set_xlabel("$\omega$")
    ax2.set_ylabel("$J(\omega)$")
    ax2.legend(loc='lower right')
    ax2.set_title("Spectral Densities")
    ax2.grid(True)

    omega_vals = np.linspace(0, freq_domain[1], 1000)
    def fitted_func(omega_vals): return approx_func_reduced(omega_vals, optimized_params)
    ax3.plot(omega_vals, target_func(omega_vals), label="Target Function")
    ax3.plot(omega_vals, fitted_func(omega_vals), label="Fitted Function")
    for i in range(num_modes):
        param = optimized_params[i * num_param: (i + 1) * num_param]
        param = param[:-1]
        ax3.plot(omega_vals, lorentz_reduced(omega_vals, *param),
                 "--", label=f"Lorentz {i+1:02d}")
    ax3.legend()

    plt.tight_layout()
    plt.savefig("correlation_and_original_functions.png", dpi=300)
