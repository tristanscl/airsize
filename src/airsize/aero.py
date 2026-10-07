import numpy as np

KB_TYPICAL = 0.0055  # geometric mean between typical values 0.001 and 0.03
CLMIN_TYPICAL = 0.2  # arithmetic mean between typical values 0.1 and 0.3
E_TYPICAL = 0.8  # arithmetic mean between typical values 0.75 and 0.85
AR_TYPICAL = 8.5  # arithmetic mean between typical values 7 and 10

# Tables


def CDmin_table(M0):
    CDmin0 = (M0 <= 0.7) * 0.02
    CDmin1 = (M0 > 0.7) * (0.2 * (M0 - 0.7))
    return CDmin0 + CDmin1


def K1_table(M0):
    K10 = (M0 <= 1.3) * 0.2
    K11 = (M0 > 1.3) * (0.36 * (M0 - 1.3))
    return K10 + K11


def CD0_table(M0):
    CD00 = (M0 <= 0.75) * 0.015
    CD01 = (M0 > 0.75) * (M0 <= 1.25) * (0.015 * (M0 - 0.75))
    CD02 = (M0 > 1.25) * 0.03
    return CD00 + CD01 + CD02


# Models


def compute_q(V, rho):
    return 0.5 * rho * V**2


def compute_D(CD, q, S):
    return CD * q * S


def compute_L(CL, q, S):
    return CL * q * S


def compute_KA(AR=AR_TYPICAL, e=E_TYPICAL):
    return 1 / (np.pi * AR * e)


def compute_K1(KA, KB=KB_TYPICAL):
    return KA + KB


def compute_K2(KB=KB_TYPICAL, CLmin=CLMIN_TYPICAL):
    return -2 * KB * CLmin


def compute_CD0(CDmin, CLmin=CLMIN_TYPICAL, KB=KB_TYPICAL):
    return CDmin + KB * CLmin**2


def drag_polar(CL, K1, K2, CD0):
    return K1 * CL**2 + K2 * CL + CD0


def compute_Vstall(WS, beta, CLmax, rho):
    return np.sqrt(2 * beta * WS / (CLmax * rho))
