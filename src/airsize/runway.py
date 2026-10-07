import numpy as np

from airsize.aero import compute_Vstall

# Takeoff


def compute_theta(dhdt, V):
    return np.arcsin(dhdt / V)


def compute_xi(CD, CDR, mu, CL):
    return CD + CDR - mu * CL


def compute_sG_TO(TW, WS, alpha, beta, xi, mu, CLmax, k, rho, g0):
    """Ground roll distance, first phase of takeoff."""
    return (
        -beta
        * WS
        / (rho * g0 * xi)
        * np.log(1 - xi / (((alpha / beta) * TW - mu) * CLmax / k**2))
    )


def compute_sR_TO(WS, beta, CLmax, k, tR, rho):
    """Rotation distance, second phase of takeoff."""
    return tR * k * np.sqrt(2 * beta / (rho * CLmax) * WS)


def compute_sTR_TO(WS, beta, CLmax, k, theta, rho, g0):
    """Transition distance, third phase of takeoff."""
    return (
        k**2 * np.sin(theta) / (g0 * (0.8 * k**2 - 1)) * 2 * beta / (rho * CLmax) * WS
    )


def compute_sOBS_TO(WS, beta, CLmax, k, theta, hOBS, rho, g0):
    """Clearing distance, fourth phase of takeoff."""
    hTR = (
        k**2
        * (1 - np.cos(theta))
        / (g0 * (0.8 * k**2 - 1))
        * 2
        * beta
        / (rho * CLmax)
        * WS
    )
    return (hOBS - hTR) / np.tan(theta)


def compute_sOBS_TR_TO(WS, beta, CLmax, k, hOBS, rho, g0):
    """Clearing distance, third phase of takeoff, when obstacle is cleared during transition."""
    Vstall = compute_Vstall(WS, beta, CLmax, rho)
    VTO = k * Vstall
    RC = VTO**2 / (g0 * (0.8 * k**2 - 1))
    thetaOBS = np.arccos(1 - hOBS / RC)
    return RC * np.sin(thetaOBS)


def compute_sTO(
    TW,
    WS,
    alpha,
    beta,
    xi,
    mu,
    CLmax,
    k,
    tR,
    hOBS,
    theta,
    rho,
    g0,
):
    """Takeoff distance is computed as the minimum between case A: the obstacle is cleared during transition, and case B: the obstacle is cleared during climb."""
    sG = compute_sG_TO(TW, WS, alpha, beta, xi, mu, CLmax, k, rho, g0)
    sR = compute_sR_TO(WS, beta, CLmax, k, tR, rho)
    sTR = compute_sTR_TO(WS, beta, CLmax, k, theta, rho, g0)
    sOBS = compute_sOBS_TO(WS, beta, CLmax, k, theta, hOBS, rho, g0)
    sOBS_TR = compute_sOBS_TR_TO(WS, beta, CLmax, k, hOBS, rho, g0)
    return np.minimum(sG + sR + sOBS_TR, sG + sR + sTR + sOBS)


# Landing


def compute_sB(TW, WS, alpha, beta, xi, mu, CLmax, k, rho, g0):
    return (
        beta
        / (rho * g0 * xi)
        * WS
        * np.log(1 + xi / ((alpha / beta * TW + mu) * CLmax / k**2))
    )
