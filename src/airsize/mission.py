import numpy as np


def weight_fraq_eq(W, D, R, TSFC, dt):
    """Differential equation dW/W = ... instantaneously exact. Can be iterated when finer analysis is needed."""
    return -TSFC * (D + R) / W * dt


def constant_speed_climb(TW, h0, h1, alpha, beta, CD, CL, V, TSFC):
    u = CD / CL * beta / alpha / TW
    if np.any(u >= 1):
        return 0.0
    return np.exp(-TSFC / V * (h1 - h0) / (1 - u))


def constant_speed_climb_check(theta, u):
    return np.sqrt(theta) / (1 - u)


def horizontal_acceleration(TW, V0, V1, alpha, beta, CD, CL, V, TSFC, g0):
    return np.exp(
        -TSFC / V * ((V1**2 - V0**2) / g0) / (1 - CD / CL * beta / alpha / TW)
    )


def horizontal_acceleration_check(V, u):
    return V / (1 - u)


def takeoff_acceleration(VTO, u, TSFC, g0):
    return np.exp(-TSFC / g0 * VTO / (1 - u))


def climb_and_acceleration(TW, h0, h1, V0, V1, alpha, beta, CD, CL, V, TSFC, g0):
    return np.exp(
        -TSFC
        / V
        * ((V1**2 - V0**2) / g0 + (h1 - h0))
        / (1 - CD / CL * beta / alpha / TW)
    )


def constant_altitude_speed_cruise(s, CD, CL, V, TSFC):
    return np.exp(-TSFC / V * CD / CL * s)


def constant_altitude_speed_turn(n, N, CD, CL, V, TSFC, g0):
    return np.exp(-TSFC * (n * CD / CL) * 2 * n * N * V / (g0 * np.sqrt(n**2 - 1)))


def loiter(K1, K2, CD0, TSFC, t):
    return np.exp(-TSFC * (np.sqrt(4 * CD0 * K1) + K2) * t)


def warm_up(TW, alpha, beta, TSFC, t):
    return np.clip(1 - TSFC * alpha / beta * TW * t, a_min=0, a_max=np.inf)


def takeoff_rotation(TW, alpha, beta, TSFC, tR):
    return np.clip(1 - TSFC * alpha / beta * TW * tR, a_min=0, a_max=np.inf)


def constant_energy_height_maneuver(CD, CL, TSFC, t):
    return np.exp(-TSFC * CD / CL * t)
