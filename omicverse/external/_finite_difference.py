"""Legacy three-point differences used by bundled SpatialDE/SOMDE.

Preserves scipy.misc.derivative's default dx=1/order=3 numerics for the
first and second derivatives used here; no global SciPy monkey patch.
"""


def derivative(func, x0, dx=1.0, n=1, args=(), order=3):
    if order != 3 or n not in (1, 2) or dx <= 0:
        raise ValueError('Only positive-step, order=3 first/second derivatives are supported.')
    left = func(x0 - dx, *args)
    right = func(x0 + dx, *args)
    if n == 1:
        return (right - left) / (2.0 * dx)
    return (left - 2.0 * func(x0, *args) + right) / dx ** 2
