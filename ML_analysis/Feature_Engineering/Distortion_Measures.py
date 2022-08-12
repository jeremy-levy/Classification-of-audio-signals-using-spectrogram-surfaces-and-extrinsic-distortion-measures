import numpy as np


def quasi_conformal(a, b):
    if b == 0 or a == 0:
        return 0
    return max(a/b, b/a)


def mips(a, b):
    if b == 0 or a == 0:
        return 0
    return a/b + b/a


def dirichlet(a, b):
    return (a**2 + b**2) / 2


def quasi_isometric(a, b):
    if b == 0:
        return a
    return max(a, 1/b)


def rigidity_energy(a, b):
    return (a**2 - 1)**2 + (b**2 - 1)**2


def linear_energy(a, b):
    return (a + b) / 2


def symmetric_rigid_energy(a, b):
    return ((a ** 2) + (a ** (-2)) + (b ** 2) + (b ** (-2))) / 4


def area_distortion(a, b):
    if a == 0 or b == 0:
        return 0
    return max(a*b, 1/(a*b))


def apply_mean_distorsion(distortion_function, first_dimension, second_dimension, weight=None):
    if weight is None:
        return np.mean(list(map(distortion_function, first_dimension, second_dimension)))

    return np.average(list(map(distortion_function, first_dimension, second_dimension)), weights=weight)


def apply_std_distortion(distortion_function, first_dimension, second_dimension):
    return np.std(list(map(distortion_function, first_dimension, second_dimension)))
