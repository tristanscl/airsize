import numpy as np


def energy_equation(WS, alpha, beta, K1, K2, CD0, R, dhdt, V, dVdt, q, S, n, g0):
    """Computes TSL/WTO from the energy equation.
    Args:
        WS: WTO/S weight loading
        alpha: thrust efficiency (T/TSL)
        beta: weight fraction (W/WTO)
        K1: drag polar quadratic coefficient
        K2: drag polar linear coefficient
        CD0: residual drag
        R: additional friction (from ground etc)
        dhdt: climb rate
        V: speed
        dVdt: acceleration
        q: dynamic pressure
        S: wing surface area
        n: load factor
        g0: gravity acceleration at sea level
    """
    x = n * beta / q * WS
    return (
        beta
        / alpha
        * (
            q / beta / WS * (K1 * x**2 + K2 * x + CD0 + R / (q * S))
            + dhdt / V
            + dVdt / g0
        )
    )


def constant_altitude_speed_cruise(WS, alpha, beta, K1, K2, CD0, q):
    return energy_equation(
        WS, alpha, beta, K1, K2, CD0, 0.0, 0.0, 1.0, 0.0, q, 1.0, 1.0, 9.81
    )


def constant_speed_climb(WS, alpha, beta, K1, K2, CD0, dhdt, V, q):
    return energy_equation(
        WS, alpha, beta, K1, K2, CD0, 0.0, dhdt, V, 0.0, q, 1.0, 1.0, 9.81
    )


def constant_altitude_speed_turn(WS, alpha, beta, K1, K2, CD0, q, n):
    return energy_equation(
        WS, alpha, beta, K1, K2, CD0, 0.0, 0.0, 1.0, 0.0, q, 1.0, n, 9.81
    )


def horizontal_acceleration(WS, alpha, beta, K1, K2, CD0, dVdt, q):
    return energy_equation(
        WS, alpha, beta, K1, K2, CD0, 0.0, 0.0, 1.0, dVdt, q, 1.0, 1.0, 9.81
    )


def takeoff_ground_roll_large_thrust(WS, alpha, beta, CLmax, sG, k, rho, g0):
    """Computes TSL/WTO from the energy equation for the takeoff ground roll (large thrust) case.
    Args:
        alpha: thrust efficiency (T/TSL)
        beta: weight fraction (W/WTO)
        CLmax: lift coefficient at stall
        sG: takeoff distance
        k: speed multiplier (V/Vstall)
        rho: air density during takeoff
        g0: gravity acceleration at sea level
    """
    return beta / alpha * k**2 / (sG * rho * g0 * CLmax) * WS


def takeoff_ground_roll_low_thrust(WS, alpha, beta, xi, mu, CLmax, s, k, rho, g0):
    """Computes TSL/WTO from the energy equation for the takeoff ground roll (low thrust) case.
    Args:
        alpha: thrust efficiency (T/TSL)
        beta: weight fraction (W/WTO)
        mu: Coulomb drag coefficient
        CLmax: lift coefficient at stall
        sG: roll distance
        k: speed multiplier (V/Vstall)
        rho: air density during takeoff
        g0: gravity acceleration at sea level
    """
    return (
        beta
        / alpha
        * (mu + (xi * k**2 / CLmax) / (1 - np.exp(-s * rho * g0 * xi / (beta * WS))))
    )


def braking_roll(WS, alpha, beta, xi, mu, CLmax, s, k, rho, g0):
    """Computes TSL/WTO from the energy equation for the takeoff ground roll (low thrust) case.
    Args:
        alpha: thrust efficiency (T/TSL)
        beta: weight fraction (W/WTO)
        mu: Coulomb drag coefficient
        CLmax: lift coefficient at stall
        s: roll distance
        k: speed multiplier (V/Vstall)
        rho: air density during takedown
        g0: gravity acceleration at sea level
    """
    return (
        beta
        / alpha
        * (mu - (xi * k**2 / CLmax) / (np.exp(s * rho * g0 * xi / (beta * WS)) - 1))
    )


def service_ceiling(WS, alpha, beta, K1, K2, CD0, dhdt, V, q):
    return energy_equation(
        WS, alpha, beta, K1, K2, CD0, 0.0, dhdt, V, 0.0, q, 1.0, 1.0, 9.81
    )


def non_stall_landing(beta, CLmax, V, rho):
    return rho * V**2 * CLmax / (2 * beta)
