# Fitting Spectral Densities using Legendre Discretization

This project approximates a target spectral density function using Lorentzian basis functions, [1] leveraging the Legendre discretization technique [2] for efficient representation and computation. See the legendre_discretization module

Two main metrics are used to assess the quality of the approximation: (1) the deviation of the fitted spectral density from the target spectral density and (2) the deviation from the target bath correlation function.

Visualization capabilities are included to plot bath correlation functions and the original spectral density functions (target and fitted) for assessing the accuracy of the approximation.


# References:
- [1] Meier, C., & Tannor, D. J. (1999). Non-Markovian evolution of the density operator in the presence of strong laser fields. _The Journal of chemical physics_, 111(8), 3365-3376.
- [2] de Vega, I., Schollwöck, U., & Wolf, F. A. (2015). How to discretize a quantum bath for real-time evolution. _Physical Review B_, 92(15), 155126.