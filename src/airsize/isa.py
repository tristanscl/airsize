import numpy as np

T0_ISA = 288.15  # K
P0_ISA = 101325  # Pa
rho0_ISA = 1.225  # kg.m-3
L_ISA = 0.0065  # K/m
g_ISA = 9.80665  # m/s²
R_ISA = 287.05  # J.m-1.K-1
TROPO_STOP = 11000.0  # m
GAMMA = 1.4


def T_ISA(h, T0=T0_ISA, L=L_ISA):
    return (T0 - L * h) * (h < TROPO_STOP) + (T0 - L * TROPO_STOP) * (h >= TROPO_STOP)


def P_ISA(h, T0=T0_ISA, P0=P0_ISA, g=g_ISA, R=R_ISA, L=L_ISA):
    T = T_ISA(h, T0, L)  # continuity of Pb handled here
    P_TROPO = P0 * (T / T0) ** (g / (R * L))
    P_CT = P_TROPO * np.exp(-g * (h - TROPO_STOP) / (R * T))
    return P_TROPO * (h < TROPO_STOP) + P_CT * (h >= TROPO_STOP)


def rho_ISA(h, T0=T0_ISA, P0=P0_ISA, R=R_ISA, g=g_ISA, L=L_ISA):
    T = T_ISA(h, T0, L)
    P = P_ISA(h, T0, P0, g, R, L)
    return P / (R * T)


def sigma_ISA(h, rho0=rho0_ISA, T0=T0_ISA, P0=P0_ISA, R=R_ISA, g=g_ISA, L=L_ISA):
    return rho_ISA(h, T0, P0, R, g, L) / rho0


def theta_ISA(h, T0=T0_ISA, L=L_ISA):
    return T_ISA(h, T0, L) / T0_ISA


def compute_cs(T, gamma=GAMMA, R=R_ISA):
    return np.sqrt(gamma * R * T)
