import numpy as np
from legendre_discretization import get_vn_squared


def _correlation_func_sum(freqs, vs_squared):
    corr_single = lambda v_squared, w, t, T: v_squared * (
        np.cos(w * t) / np.tanh(w / 2 / T) - 1j * np.sin(w * t)
    )
    corr = lambda t, T: np.sum(corr_single(vs_squared, freqs, t, T))
    return np.vectorize(corr)


def correlation_func_sum(J, n, domain):
    freq, vs_squared = get_vn_squared(J, n, domain)
    return np.vectorize(_correlation_func_sum(freq, vs_squared))


def correlation_func_integral(J, n, domain):
    import scipy.integrate as spi

    corr = lambda t, T: spi.quad(
        lambda w: J(w) * (np.cos(w * t) / np.tanh(w / 2 / T) - 1j * np.sin(w * t)),
        *domain
    )[0]
    return np.vectorize(corr)
