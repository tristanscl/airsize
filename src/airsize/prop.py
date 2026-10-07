import numpy as np

from airsize.isa import P0_ISA, GAMMA

BCM = 0.7

# Tables


def alpha_HBP_turbofan(M, sigma, tolerance=1.1, bypass_check: bool = False):
    if not bypass_check:
        assert M / tolerance <= 0.9, "Mach number is too large for HBP turbofan engine."
    return (0.568 + 0.25 * (1.2 - M) ** 3) * sigma**0.6


def alpha_LBP_mixed_turbofan(M, sigma, afterburners=False):
    alpha0 = (not afterburners) * (
        0.72 * (0.88 + 0.245 * np.abs(M - 0.6) ** 1.4) * sigma**0.7
    )
    alpha1 = afterburners * (0.94 + 0.38 * (M - 0.4) ** 2) * sigma**0.7
    return alpha0 + alpha1


def alpha_turbojet(M, sigma, afterburners=False):
    alpha0 = (not afterburners) * (
        0.76 * (0.907 + 0.262 * np.abs(M - 0.5) ** 1.5) * sigma**0.7
    )
    alpha1 = afterburners * (0.952 + 0.3 * (M - 0.4) ** 2) * sigma**0.7
    return alpha0 + alpha1


def alpha_turboprop(M, sigma):
    alpha0 = (M <= 0.1) * np.sqrt(sigma)
    alpha1 = (M > 0.1) * (0.12 / (M + 0.02) * np.sqrt(sigma))
    return alpha0 + alpha1


# Models


def compute_u(D, R, T):
    return (D + R) / T


def compute_u_constant_speed_climb(TW, alpha, beta, CD, CL):
    return CD / CL * beta / alpha / TW


def compute_u_horizontal_acceleration(TW, alpha, beta, CD, CL):
    return CD / CL * beta / alpha / TW


def compute_u_takeoff_acceleration(TW, WS, alpha, beta, xi, mu, q):
    return (xi * q / beta / WS + mu) * beta / alpha / TW


def compute_TSFC_FJ(C, theta):
    return C * np.sqrt(theta)


def compute_TSFC_turboprop(C, M):
    return C * M


def delta_optimal(WS, beta, K1, CD0, M, P0=P0_ISA, gamma=GAMMA):
    return 2 * beta / (gamma * P0 * M**2) / np.sqrt(CD0 / K1) * WS
