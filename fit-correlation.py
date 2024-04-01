import numpy as np
import matplotlib.pyplot as plt
from legendre_discretization import get_vn_squared, get_coups_sq, get_freqs
from common import (
    correlation_func_sum,
    _correlation_func_sum,
    correlation_func_integral,
)
from scipy.optimize import LinearConstraint, minimize


def lorentz(ω, g, gamma, omega0):
    return g**2 * (
        gamma / 2 / ((gamma / 2) ** 2 + (ω - omega0) ** 2)
        - gamma / 2 / ((gamma / 2) ** 2 + (ω + omega0) ** 2)
    )


def ohmic(omega, alpha, omega_c):
    return alpha * omega**0.5 * np.exp(-omega / omega_c)


# define new lorentz function where gamma is 0.5*omega0


def lorentz_reduced(ω, g, omega0):
    gamma = 0.5 * omega0
    return g**2 * (
        gamma / 2 / ((gamma / 2) ** 2 + (ω - omega0) ** 2)
        - gamma / 2 / ((gamma / 2) ** 2 + (ω + omega0) ** 2)
    )


def approx_func_reduced(omega, params):
    g = params[::2]
    omega0 = params[1::2]
    approx = lorentz_reduced(omega[:, np.newaxis], g, omega0).sum(axis=1)
    return approx


def approx_func(omega, params):
    g = params[::3]
    gamma = params[1::3]
    omega0 = params[2::3]
    approx = lorentz(omega[:, np.newaxis], g, gamma, omega0).sum(axis=1)
    return approx


def corr_integral(ws, vs_squared, T, t_max):
    return vs_squared * (
        np.sin(ws * t_max) / np.tanh(ws / 2 / T) - 1j * (1 - np.cos(ws * t_max))
    )


def corr_integral(ws, vs_squared, T, t_max):
    # returns return vs_squared * (np.sin(ws*t_max) / np.tanh(ws/2/T) - 1j * (1-np.cos(ws*t_max)))
    ws_t_max = ws * t_max
    ws_2T = ws / (2 * T)

    sin_term = np.sin(ws_t_max)
    cos_term = np.cos(ws_t_max)
    exp_term = np.exp(-ws_2T)
    tanh_term = (1 - exp_term) / (1 + exp_term)

    real_part = sin_term / tanh_term
    imag_part = -1j * (1 - cos_term)

    return vs_squared * (real_part + imag_part)


def objective_func_sd(params):
    target_vs_squared = get_coups_sq(
        lambda omega: ohmic(omega, alpha, omega_c), freqs, eig_vecs, domain
    )
    approx_vs_squared = get_coups_sq(
        lambda omega: approx_func(omega, params), freqs, eig_vecs, domain
    )

    return np.abs(approx_vs_squared - target_vs_squared).sum()


# def objective_func_corr(params):
#     target_vs_squared = get_coups_sq(lambda omega: ohmic(
#         omega, alpha, omega_c), freqs, eig_vecs, domain)
#     approx_vs_squared = get_coups_sq(lambda omega: approx_func(
#         omega, params), freqs, eig_vecs, domain)
#     target_integral = corr_integral(freqs, target_vs_squared, T, t_max)
#     approx_integral = corr_integral(freqs, approx_vs_squared, T, t_max)

#     return np.abs(target_integral - approx_integral).sum()


def objective_func_corr(params):
    approx_vs_squared = get_coups_sq(
        lambda omega: approx_func(omega, params), freqs, eig_vecs, domain
    )

    # Evaluate the target and approximate correlation functions on the time grid
    # target_corr = np.zeros_like(t_grid, dtype=complex)
    # approx_corr = np.zeros_like(t_grid, dtype=complex)
    # for i, t in enumerate(t_grid):
    approx_corr = _correlation_func_sum(freqs, approx_vs_squared)(t_grid, T)

    # Calculate the absolute (or squared) difference between the correlation functions
    diff = np.abs(target_corr - approx_corr)
    # Integrate the absolute (or squared) difference
    integral_diff = np.dot(diff, exp_t_grid)

    return integral_diff


# Set the parameters
alpha = 0.1
omega_c = 10
n = 1000
domain = (0, 100)
num_modes = 3
num_param = 3
T = 0.5
t_max = 1
# Define a time grid for evaluating the correlation functions
t_grid = np.linspace(0, t_max, 1000)
inv_t_grid = np.exp(1 / (1 + t_grid))
exp_t_grid = np.exp(-5*t_grid)

# Define the grids in the frequency domain
freqs, eig_vecs = get_freqs(n, domain)
target_vs_squared = get_coups_sq(
    lambda omega: ohmic(omega, alpha, omega_c), freqs, eig_vecs, domain
)
target_corr = _correlation_func_sum(freqs, target_vs_squared)(t_grid, T)

# Define the optimization constraints
bounds = [(0, 100), (0, 100), (0.0001, 100)] * num_modes
initial_params = np.array([1, 5, 10] * num_modes)

# linear_constraint = ()
objective_func = objective_func_corr
objective_func = objective_func_sd

# Define the linear constraint matrix and vector
constraint_matrix = np.zeros((num_modes, 3 * num_modes))
for i in range(num_modes):
    constraint_matrix[i, i * 3 + 1] = 1
    constraint_matrix[i, i * 3 + 2] = -0.5

linear_constraint = LinearConstraint(constraint_matrix, -np.inf, 0)

if num_param == 2:
    bounds = [(0, 100), (0.0001, 100)] * num_modes
    initial_params = np.array([1, 10] * num_modes)
    linear_constraint = ()
    lorentz = lorentz_reduced
    approx_func = approx_func_reduced


# def objective_func_penalized(params, penalty_factor=1e6):
# target_vs_squared = get_coups_sq(lambda omega: ohmic(
#     omega, alpha, omega_c), freqs, eig_vecs, domain)
# approx_vs_squared = get_coups_sq(lambda omega: approx_func(
#     omega, params), freqs, eig_vecs, domain)

# objective_value = np.abs(approx_vs_squared - target_vs_squared).sum()

# # Calculate the constraint violation
# constraint_violation = np.maximum(
#     0, constraint_matrix @ params - constraint_vector)
# penalized_objective_value = objective_value + \
#     penalty_factor * np.sum(constraint_violation)

# return penalized_objective_value


if __name__ == "__main__":
    # Optimize the parameters

    minimizer_kwargs = {
        "method": "SLSQP",
        "bounds": bounds,
        "constraints": linear_constraint,
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

    result = differential_evolution(
        objective_func,
        bounds,
        maxiter=2 * 10**6,
        popsize=20,
        mutation=(0.5, 1),
        tol=1e-12,
        atol=0,
        recombination=0.7,
        constraints=linear_constraint,
        updating="deferred",
        workers=-1,
    )
    # result = dual_annealing(objective_func, bounds)
    # result = shgo(objective_func, bounds)
    # result = basinhopping(objective_func, initial_params, niter=100, T=1.0, stepsize=0.5, minimizer_kwargs=minimizer_kwargs)
    # result = direct(objective_func, bounds, maxiter=int(1e8), maxfun=int(1e8), eps=0.5, locally_biased=False)
    # result = shgo(objective_func, bounds, workers=-1)
    optimized_params = result.x

    # Print the optimized parameters
    print("Optimized parameters:\n", result)
    for i in range(num_modes):
        param = optimized_params[i * num_param : (i + 1) * num_param]
        if num_param == 2:
            print(f"Basis function {i+1}: g={param[0]:>8.4f}, gamma={param[1]:>8.4f}")
        if num_param == 3:
            print(
                f"Basis function {i+1}: g={param[0]:>8.4f}, gamma={param[1]:>8.4f}, omega0={param[2]:>8.4f},ratio={param[2]/param[1]:>8.4f}"
            )
    print(
        f"Optimized Correlation Function Value: {objective_func_corr(optimized_params):.3f}"
    )
    print(
        f"Optimized Spectral Density Value    : {objective_func_sd(optimized_params):.3f}"
    )

    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Plot the correlation functions

    t_vals = np.linspace(0, 4, 1000)
    target_corr = correlation_func_sum(
        lambda omega: ohmic(omega, alpha, omega_c), n, domain
    )
    fitted_corr = correlation_func_sum(
        lambda omega: approx_func(omega, optimized_params), n, domain
    )

    ax1.plot(t_vals, target_corr(t_vals, T).real, label="Target Correlation")
    ax1.plot(t_vals, fitted_corr(t_vals, T).real, label="Fitted Correlation")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Correlation")
    ax1.legend()
    ax1.set_title("Correlation Functions")
    ax1.grid(True)

    omega_vals = np.linspace(0, domain[1] / 2, 1000)
    target_func = ohmic(omega_vals, alpha, omega_c)
    fitted_func = approx_func(omega_vals, optimized_params)

    ax2.plot(omega_vals, target_func, label="Target Function")
    ax2.plot(omega_vals, fitted_func, label="Fitted Function")
    for i in range(num_modes):
        param = optimized_params[i * num_param : (i + 1) * num_param]
        ax2.plot(
            omega_vals, lorentz(omega_vals, *param), "--", label=f"Component {i+1}"
        )

    ax2.set_xlabel("Omega")
    ax2.set_ylabel("Function Value")
    ax2.legend()
    ax2.set_title("Original Functions")
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig("correlation_and_original_functions.png", dpi=300)