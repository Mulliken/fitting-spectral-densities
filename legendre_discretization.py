#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 15:20:02 2021

@author: michaelwu, Mulliken
"""

import numpy as np

import matplotlib.pyplot as plt


def get_recursion(n, j, domain, g=1, ncap=20000):  # j=weight function
    """
    The old way of calculating. Deprecated due to computational cost.
    Replaced by get_legendre_recursion() where an analytical expression for alpha and beta is used.
    """
    import fishbonett.recurrence_coefficients as rc
    alphaL, sqrt_betaL = rc.recurrenceCoefficients(
        n - 1, lb=domain[0], rb=domain[1], j=j, g=g, ncap=ncap
    )
    j = lambda x: j(x) * np.pi
    alphaL = g * np.array(alphaL)
    sqrt_betaL = g * np.sqrt(np.array(sqrt_betaL))
    sqrt_betaL[0] = sqrt_betaL[0] / g
    return alphaL, sqrt_betaL[1:]  # k=sqrt(beta), w=alpha, sqrt_beta[0] is dropped


def get_legendre_recursion(n, domain):
    l = domain[0]
    r = domain[1]
    assert l < r
    a = (l+r) / 2
    a = np.repeat(a, n)
    _temp = (r-l)/2
    b = (lambda x: _temp * x / np.sqrt(4 * x ** 2 - 1))(np.arange(1, n))
    return a, b

def get_freqs(n, domain):
    alpha, beta = get_legendre_recursion(n, domain)
    M = np.diag(alpha) + np.diag(beta, -1) + np.diag(beta, 1)
    freqs, eig_vec = np.linalg.eigh(M)
    return freqs, eig_vec

def get_coups_sq(j, freqs, eig_ves, domain):
    W = (eig_ves[0, :]) ** 2 * (domain[1] - domain[0])
    V_squared = j(freqs) * W
    return V_squared

def get_vn_squared(j, n: int, domain):
    freqs, eig_vecs = get_freqs(n, domain)
    V_squared = get_coups_sq(j, freqs, eig_vecs, domain)
    return freqs, V_squared


def get_approx_func(J, n, domain, epsilon):
    delta = lambda x: 1 / np.pi * epsilon / (epsilon ** 2 + x ** 2)
    w, V_squared = get_vn_squared(J, n, domain)
    j_approx = lambda x: np.sum([vi * delta(x - wi) for wi, vi in zip(w, V_squared)])
    return np.vectorize(j_approx)


if __name__ == '__main__':
    def lorentzian(eta, w, lambd=5245., omega=77.):
        return 0.5 * lambd * (omega ** 2) * eta * w / ((w ** 2 - omega ** 2) ** 2 + (eta ** 2) * (w ** 2))

    drude = lambda x, gam, lam: 2 * lam * gam * x / (x ** 2 + gam ** 2)
    lorentzian1 = lambda w: lorentzian(100, w, 10, 1000)
    J = lorentzian1

    J_approx = get_approx_func(J, 1000, [0, 5000], 5)
    print("Get approx func:", J_approx(10))

    x = np.linspace(0, 5000, 1000)
    disc = []
    for xi in x:
        disc += [J_approx(xi)]

    plt.plot(x, J(x), 'r-', label='original')
    plt.plot(x, disc, 'k-', label='approx')
    plt.legend()
    plt.show()
