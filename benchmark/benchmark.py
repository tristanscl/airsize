"""
F-86 Benchmark (test #1 of AirSize)
Author: Tristan Scuiller
"""

# 3rd party
from typing import Callable
import numpy as np
import json
import matplotlib.pyplot as plt

# Airsize
import airsize.mission as mission
import airsize.constraints as cst
import airsize.weights as weights
import airsize.units as units
import airsize.aero as aero
import airsize.isa as isa
import airsize.prop as prop
import airsize.dyn as dyn
import airsize.runway as rw
import airsize.optim as optim

## Global ##


# Global: requirements
WP = units.lb2kg(210 + 432) * 1.1  # kg, ASSUMPTION (10% margin)
min_fuel = 0.1 + 0.05  # ASSUMPTION (5% margin)
sTO_max = units.ft2m(4400)  # m
sL_max = units.ft2m(5000)  # m

# Global: aerodynamics
CLmax = 1.3
CL_max_LD = 0.97960
CD_max_LD = 0.01550
K1 = 1.42e-2
K2 = 0.0
CD0 = 4.59e-3
CDR = 0.0  # ASSUMPTION (drag from landing gear is negligible)

# Global: physics
g0 = isa.g_ISA  # m.s-2

# Global: propulsion
cruise_alpha_model = lambda x: np.maximum(0.0, 1.0 - 5.657e-5 * x)
mil_alpha_model = lambda x: np.maximum(0.0, 1.0 - 5.500e-5 * x)
wet_alpha_model = lambda x: np.maximum(0.0, 1.0 - 4.805e-5 * x)
cruise_avg_tsfc = 4.23e-4  # s-1
mil_avg_tsfc = 3.98e-4  # s-1
wet_avg_tsfc = 7.83e-4  # s-1


## Phases ##

# Phase 1a: warm up
TSFC_1a = 1.13 / 3600  # s-1
alpha_1a = 1.0
beta_1a = 1.0
t_1a = 80.0  # s

# Phase 1b: takeoff ground roll
rho_1b = isa.rho0_ISA  # kg.m-3
k_1b = 1.1  # ASSUMPTION (10% margin over Vstall for takeoff)
alpha_1b = alpha_1a
TSFC_1b = wet_avg_tsfc  # s-1
CL_1 = CLmax * 1.3  # ASSUMPTION (high lift devices, increasing lift by 30%)
drag_penalty_1 = 1.35  # ASSUMPTION (drag is increased by 35% using high lift devices)
mu_1b = 0.02  # ASSUMPTION (from lecture slides)
s_1 = 0.9 * sTO_max  # ASSUMPTION (10% margin)

# Phase 1c: takeoff rotation
rho_1c = isa.rho0_ISA  # kg.m-3
alpha_1c = alpha_1b
TSFC_1c = TSFC_1b  # s-1
tR_1c = 3.0  # s, ASSUMPTION (from lecture slides)
hOBS_1c = units.ft2m(50)  # m

# Phase 2: horizontal acceleration
h2_2 = 0.0  # m
rho_2 = isa.rho0_ISA  # kg.m-3
CL_2 = CLmax * 1.3  # ASSUMPTION (high lift devices, increasing lift by 30%)
V1_2 = units.ft2m(1020 * 1.1)  # m/s, ASSUMPTION (10% more than target speed)
TSFC_2 = wet_avg_tsfc  # s-1
alpha_2 = 1.0
dVdt_2 = 0.2 * g0  # m.s-2 ASSUMPTION (0.2g acceleration)


# Phase 3: initial climb
dhdt_3 = units.ft2m(90) * 0.8  # m/s, ASSUMTION (20% margin wrt requirement)
h0_3 = 0.0  # m/s
h1_3 = units.ft2m(35400)  # m
n_split_3 = 10  # ASSUMPTION (alpha relatively constant on 3540 ft intervals)
CL_3 = CLmax  # ASSUMPTION (climbing with CLmax and no high lift devices)
V_3 = V1_2  # ASSUMPTION (continuity)
TSFC_3 = mil_avg_tsfc  # m.s-1

# Phase 4: cruise-climb
h0_4 = h1_3  # m
h1_4 = units.ft2m(38700)  # m
CD_4 = CD_max_LD  # ASSUMPTION (economy)
CL_4 = CL_max_LD  # ASSUMPTION (economy)
V_4 = units.kt2ms(458)  # m.s-1
TSFC_4 = cruise_avg_tsfc  # s-1
s_4 = units.nm2km(550) * 1000  # m

# Phase 5: loiter
TSFC_5 = cruise_avg_tsfc  # s-1
t_5 = 10 * 60  # s
h_5 = h1_4  # m
V_5 = V_4  # ASSUMPTION (speed continuity)

# Phase 6: climb
h0_6 = h0_4  # m
h1_6 = units.ft2m(47550)  # m
n_split_6 = 3  # ASSUMPTION (alpha relatively constant on 2950 ft intervals)
CL_6 = CLmax  # ASSUMPTION (climbing with CLmax and no high lift devices)
V_6 = V_4  # ASSUMPTION (climb at cruise speed)
TSFC_6 = wet_avg_tsfc  # ASSUMPTION (climb using afterburners)
dhdt_6 = dhdt_3  # ASSUMPTION (climb rate is the same than initial climb)

# Phase 7: combat
V_7 = units.kt2ms(500)  # m.s-1 (from unclassified data)
RC_7 = units.ft2m(17000)  # m, ASSUMPTION (worst case 45k ft unclassified data)
t_7 = 5 * 60  # s
CL_7 = CLmax  # ASSUMPTION (lift necessary to shorten turn radius without requiring immmense S)
TSFC_7 = wet_avg_tsfc  # s-1
h_7 = h1_6  # m


# Phase 8: cruise
h_8 = units.ft2m(37000)  # m
V_8 = units.kt2ms(536)  # m.s-1
CD_8 = CD_max_LD  # ASSUMPTION (economy)
CL_8 = CL_max_LD  # ASSUMPTION (economy)
TSFC_8 = cruise_avg_tsfc  # s-1
s_8 = units.nm2km(550) * 1000  # m


# Phase 9: loiter
h_9 = units.ft2m(35000)  # m
t_9 = 10 * 60  # s
TSFC_9 = cruise_avg_tsfc  # s-1
V_9 = V_8  # m.s-1  # ASSUMPTION (maximum endurance speed close to 458 kt)


# Phase 10: landing
h_10 = 0.0
alpha_10 = 0.65  # ASSUMPTION (fraction of thrust that can be reversed)
CL_10 = CLmax
mu_10 = (
    10 * 0.05
)  # ASSUMPTION (brakes have the same effect than 100 times the Coulomb drag coefficient during takeoff)
k_10 = 1.1  # ASSUMPTION (10% margin wrt Vstall)
rho_10 = isa.rho0_ISA
s_10 = 0.8 * sL_max  # ASSUMPTION (20% margin)


## Optimization settings ##

lr = 3e-2
n_epochs = 1000
mission_penalty_fun = lambda x: 10 * np.maximum(-x + 0.2, 0.0) ** 2
constraints_penalty_fun = lambda x: 10 * np.maximum(-x + 0.2, 0.0) ** 2
sanity_penalty_fun = lambda x: 0.0  # no sanity penalty (for positivity)
Wref = units.lbf2N(16252)  # N
Tref_wet = units.lbf2N(7650)  # N
Tref_mil = units.lbf2N(5550)  # N
Tref_cruise = units.lbf2N(5100)  # N
Sref = units.ft2m(units.ft2m(313.37))  # m2
TWref_cruise = Tref_cruise / Wref
TWref_mil = Tref_mil / Wref
TWref_wet = Tref_wet / Wref
TWref = np.array([TWref_cruise, TWref_mil, TWref_wet])
WSref = Wref / Sref  # Pa
TW0 = (np.array([1, 1, 1]) * TWref).tolist()
WS0 = WSref
S0 = Sref

print("TWref:", TWref)
print("WSref:", WSref, "Pa")
print("Sref:", Sref, "m2")

## Utils ##


def comp_WF0(WS, S):
    WTO = WS * S
    WE = weights.wE_fighter_table(WTO) * WTO
    WF_total = WTO - WE - WP
    WF = WF_total * (1 - min_fuel)
    return WF


def comp_beta(WF, WS, S):
    WTO = WS * S
    WF0_total = comp_WF0(WS, S) / (1 - min_fuel)
    WE = weights.wE_fighter_table(WTO) * WTO
    WF_total = WF + min_fuel * WF0_total  # Total WF
    return (WE + WP + WF_total) / WTO


def phase_header(x: np.ndarray, prev_phase: Callable) -> tuple[float, float]:
    TW = x[:-2] * TWref
    WS = x[-2] * WSref
    S = x[-1] * Sref
    WTO = WS * S
    WF_before = prev_phase(x) * Wref  # Net WF
    beta_before = comp_beta(WF_before, WS, S)
    return TW, WS, S, WTO, beta_before


def phase_footer(
    WS, S, beta_instant, beta_before
) -> float:  # where fuel can become negative
    WTO = WS * S
    beta_after = beta_instant * beta_before
    WE = weights.wE_fighter_table(WTO) * WTO
    W = WTO * beta_after
    WF_total = W - WE - WP
    WF0_total = comp_WF0(WS, S) / (1 - min_fuel)
    WF = WF_total - min_fuel * WF0_total
    return WF


def constraint_header(
    x: np.ndarray, prev_phase: Callable
) -> tuple[float, float, float, float]:
    TW = x[:-2] * TWref
    WS = x[-2] * WSref
    S = x[-1] * Sref
    WF = prev_phase(x) * Wref
    beta = comp_beta(WF, WS, S)
    return TW, WS, S, beta


## Mission analysis constraints ##


def phase_1a(x: np.ndarray) -> float:
    _, _, TW_wet = x[:-2] * TWref
    WS = x[-2] * WSref
    S = x[-1] * Sref
    WTO = WS * S
    WE = weights.wE_fighter_table(WTO) * WTO
    WF0_total = WTO - WE - WP
    beta_instant = mission.warm_up(TW_wet, alpha_1a, beta_1a, TSFC_1a, t_1a)
    W = beta_instant * WTO
    WF_total = W - WP - WE
    WF = WF_total - min_fuel * WF0_total
    return WF / Wref


def phase_1b(x: np.ndarray) -> float:
    TW, WS, S, WTO, beta_before = phase_header(x, phase_1a)
    _, _, TW_wet = TW
    Vstall = aero.compute_Vstall(WS, beta_before, CL_1, rho_1b)
    VTO = k_1b * Vstall
    CD = aero.drag_polar(CLmax, K1, K2, CD0) * drag_penalty_1
    xi = rw.compute_xi(CD, CDR, mu_1b, CL_1)
    q = aero.compute_q(VTO, rho_1b)
    u = prop.compute_u_takeoff_acceleration(
        TW_wet, WS, alpha_1b, beta_before, xi, mu_1b, q
    )
    beta_instant = mission.takeoff_acceleration(VTO, u, TSFC_1b, g0)
    WF = phase_footer(WS, S, beta_instant, beta_before)
    return WF / Wref


def phase_1c(x: np.ndarray) -> float:
    TW, WS, S, WTO, beta_before = phase_header(x, phase_1b)
    _, _, TW_wet = TW
    beta_instant = mission.takeoff_rotation(
        TW_wet, alpha_1c, beta_before, TSFC_1c, tR_1c
    )
    WF = phase_footer(WS, S, beta_instant, beta_before)
    return WF / Wref


def phase_2(x: np.ndarray) -> float:
    TW, WS, S, WTO, beta_before = phase_header(x, phase_1c)
    _, _, TW_wet = TW
    V0 = k_1b * aero.compute_Vstall(WS, beta_before, CLmax, rho_2)
    V = (V0 + V1_2) / 2
    CD_2 = aero.drag_polar(CLmax, K1, K2, CD0) * drag_penalty_1
    beta_instant = mission.horizontal_acceleration(
        TW_wet, V0, V1_2, alpha_2, beta_before, CD_2, CL_2, V, TSFC_2, g0
    )
    WF = phase_footer(WS, S, beta_instant, beta_before)
    return WF / Wref


def phase_3(x: np.ndarray) -> float:
    TW, WS, S, WTO, beta_before = phase_header(x, phase_2)
    _, TW_mil, _ = TW
    h_range = np.linspace(h0_3, h1_3, n_split_3 + 1)
    beta_instant = 1.0
    CD_3 = aero.drag_polar(CL_3, K1, K2, CD0)
    for i in range(n_split_3):
        h0 = h_range[i]
        h1 = h_range[i + 1]
        alpha = cruise_alpha_model(h1)
        beta_instant *= mission.constant_speed_climb(
            TW_mil, h0, h1, alpha, beta_instant, CD_3, CL_3, V_3, TSFC_3
        )
    WF = phase_footer(WS, S, beta_instant, beta_before)
    return WF / Wref


def phase_4(x: np.ndarray) -> float:
    TW, WS, S, WTO, beta_before = phase_header(x, phase_3)
    TW_cruise, _, _ = TW
    alpha = cruise_alpha_model(h1_4)
    beta_instant = mission.constant_speed_climb(
        TW_cruise, h0_4, h1_4, alpha, beta_before, CD_4, CL_4, V_4, TSFC_4
    )
    WF = phase_footer(WS, S, beta_instant, beta_before)
    return WF / Wref


def phase_5(x: np.ndarray) -> float:
    TW, WS, S, WTO, beta_before = phase_header(x, phase_4)
    TW_cruise, _, _ = TW
    beta_instant = mission.loiter(K1, K2, CD0, TSFC_5, t_5)
    WF = phase_footer(WS, S, beta_instant, beta_before)
    return WF / Wref


def phase_6(x: np.ndarray) -> float:
    TW, WS, S, WTO, beta_before = phase_header(x, phase_5)
    _, _, TW_wet = TW
    h_range = np.linspace(h0_6, h1_6, n_split_6 + 1)
    beta_instant = 1.0
    CD_6 = aero.drag_polar(CL_6, K1, K2, CD0)
    for i in range(n_split_6):
        h0 = h_range[i]
        h1 = h_range[i + 1]
        alpha = cruise_alpha_model(h1)
        beta_instant *= mission.constant_speed_climb(
            TW_wet, h0, h1, alpha, beta_instant, CD_6, CL_6, V_6, TSFC_6
        )
    WF = phase_footer(WS, S, beta_instant, beta_before)
    return WF / Wref


def phase_7(x: np.ndarray) -> float:
    TW, WS, S, WTO, beta_before = phase_header(x, phase_6)
    _, _, TW_wet = TW
    n = dyn.compute_n(V_7, g0, RC_7)
    N = dyn.compute_N(t_7, V_7, n, RC_7)
    CD = aero.drag_polar(CL_7, K1, K2, CD0)
    beta_instant = mission.constant_altitude_speed_turn(n, N, CD, CL_7, V_7, TSFC_7, g0)
    WF = phase_footer(WS, S, beta_instant, beta_before)
    return WF / Wref


def phase_8(x: np.ndarray) -> float:
    TW, WS, S, WTO, beta_before = phase_header(x, phase_7)
    TW_cruise, _, _ = TW
    beta_instant = mission.constant_altitude_speed_cruise(s_8, CD_8, CL_8, V_8, TSFC_8)
    WF = phase_footer(WS, S, beta_instant, beta_before)
    return WF / Wref


def phase_9(x: np.ndarray) -> float:
    TW, WS, S, WTO, beta_before = phase_header(x, phase_8)
    TW_cruise, _, _ = TW
    beta_instant = mission.loiter(K1, K2, CD0, TSFC_9, t_9)
    WF = phase_footer(WS, S, beta_instant, beta_before)
    return WF / Wref


## Energy analysis constraints ##


def constraint_1(x: np.ndarray) -> float:
    TW, WS, S, beta = constraint_header(x, phase_1a)
    _, _, TW_wet = TW
    CD = aero.drag_polar(CLmax, K1, K2, CD0) * drag_penalty_1
    xi = rw.compute_xi(CD, CDR, mu_1b, CL_1)
    TWmin = cst.takeoff_ground_roll_low_thrust(
        WS, alpha_1b, beta, xi, mu_1b, CL_1, s_1, k_1b, rho_1b, g0
    )
    return (TW_wet - TWmin) / TWref_wet


def constraint_2(x: np.ndarray) -> float:
    TW, WS, S, beta = constraint_header(x, phase_1c)
    _, _, TW_wet = TW
    Vstall = aero.compute_Vstall(WS, beta, CL_2, rho_2)
    V0 = k_1b * Vstall
    q = aero.compute_q(V0, rho_2)
    TWmin = (
        cst.horizontal_acceleration(WS, alpha_2, beta, K1, K2, CD0, dVdt_2, q)
        * drag_penalty_1
    )
    return (TW_wet - TWmin) / TWref_wet


def constraint_3(x: np.ndarray) -> float:
    TW, WS, S, beta = constraint_header(x, phase_2)
    _, TW_mil, _ = TW
    alpha = mil_alpha_model(h1_3)
    rho = isa.rho_ISA(h1_3)
    q = aero.compute_q(V_3, rho)
    TWmin = cst.constant_speed_climb(WS, alpha, beta, K1, K2, CD0, dhdt_3, V_3, q)
    return (TW_mil - TWmin) / TWref_mil


def constraint_4(x: np.ndarray) -> float:
    TW, WS, S, beta = constraint_header(x, phase_3)
    TW_cruise, _, _ = TW
    alpha = mil_alpha_model(h1_4)
    rho = isa.rho_ISA(h1_4)
    q = aero.compute_q(V_4, rho)
    h = h1_4 - h0_4
    dhdt_4 = V_4 * h / np.sqrt(s_4**2 + h**2)  # ASSUMPTION (climbing a straight line)
    TWmin = cst.constant_speed_climb(WS, alpha, beta, K1, K2, CD0, dhdt_4, V_4, q)
    return (TW_cruise - TWmin) / TWref_cruise


def constraint_5(x: np.ndarray) -> float:
    TW, WS, S, beta = constraint_header(x, phase_4)
    TW_cruise, _, _ = TW
    alpha = cruise_alpha_model(h_5)
    rho = isa.rho_ISA(h_5)
    q = aero.compute_q(V_5, rho)
    TWmin = cst.constant_altitude_speed_cruise(WS, alpha, beta, K1, K2, CD0, q)
    return (TW_cruise - TWmin) / TWref_cruise


def constraint_6(x: np.ndarray) -> float:
    TW, WS, S, beta = constraint_header(x, phase_5)
    _, _, TW_wet = TW
    alpha = mil_alpha_model(h1_6)
    rho = isa.rho_ISA(h1_6)
    q = aero.compute_q(V_6, rho)
    TWmin = cst.constant_speed_climb(WS, alpha, beta, K1, K2, CD0, dhdt_6, V_6, q)
    return (TW_wet - TWmin) / TWref_wet


def constraint_7(x: np.ndarray) -> float:
    TW, WS, S, beta = constraint_header(x, phase_6)
    _, _, TW_wet = TW
    alpha = wet_alpha_model(h_7)
    rho = isa.rho_ISA(h_7)
    q = aero.compute_q(V_7, rho)
    n = dyn.compute_n(V_7, g0, RC_7)
    TWmin = cst.constant_altitude_speed_turn(WS, alpha, beta, K1, K2, CD0, q, n)
    return (TW_wet - TWmin) / TWref_wet


def constraint_8(x: np.ndarray) -> float:
    TW, WS, S, beta = constraint_header(x, phase_7)
    TW_cruise, _, _ = TW
    alpha = cruise_alpha_model(h_8)
    rho = isa.rho_ISA(h_8)
    q = aero.compute_q(V_8, rho)
    TWmin = cst.constant_altitude_speed_cruise(WS, alpha, beta, K1, K2, CD0, q)
    return (TW_cruise - TWmin) / TWref_cruise


def constraint_9(x: np.ndarray) -> float:
    TW, WS, S, beta = constraint_header(x, phase_8)
    TW_cruise, _, _ = TW
    alpha = cruise_alpha_model(h_9)
    rho = isa.rho_ISA(h_9)
    q = aero.compute_q(V_9, rho)
    TWmin = cst.constant_altitude_speed_cruise(WS, alpha, beta, K1, K2, CD0, q)
    return (TW_cruise - TWmin) / TWref_cruise


def constraint_10(x: np.ndarray) -> float:
    TW, WS, S, beta = constraint_header(x, phase_9)
    _, _, TW_wet = TW
    CD = aero.drag_polar(CL_10, K1, K2, CD0)
    xi = rw.compute_xi(CD, CDR, mu_10, CL_10)
    TWmin = cst.braking_roll(
        WS, alpha_10, beta, xi, mu_10, CLmax, s_10, k_10, rho_10, g0
    )
    return (TW_wet - TWmin) / TWref_wet


## Optimization ##


if __name__ == "__main__":
    # Execution
    print("Finding minimum weight design...")
    history = optim.minimum_fuel_design(
        TW0=TW0,
        WS0=WS0,
        S0=S0,
        WP=WP,
        empty_weight_fun=weights.wE_fighter_table,
        constraints=[
            constraint_1,
            constraint_2,
            constraint_3,
            constraint_4,
            constraint_5,
            constraint_6,
            constraint_7,
            constraint_8,
            constraint_9,
            constraint_10,
        ],
        mission=[
            phase_1a,
            phase_1b,
            phase_1c,
            phase_2,
            phase_3,
            phase_4,
            phase_5,
            phase_6,
            phase_7,
            phase_8,
            phase_9,
        ],
        min_fuel=min_fuel,
        TWref=TWref,
        WSref=WSref,
        Sref=Sref,
        n_epochs=n_epochs,
        lr=lr,
        mission_penalty_fun=mission_penalty_fun,
        constraint_penalty_fun=constraints_penalty_fun,
        sanity_penalty_fun=sanity_penalty_fun,
        return_path=True,
        verbose=False,
        print_path=True,
        normalize_print=True,
    )
    TW_cruise_history = []
    TW_mil_history = []
    TW_wet_history = []
    WS_history = []
    S_history = []
    for TW, WS, S in history:
        TW_cruise_history.append(TW[0])
        TW_mil_history.append(TW[1])
        TW_wet_history.append(TW[2])
        WS_history.append(WS)
        S_history.append(S)
    results_dict = {
        "TW_cruise": TW_cruise_history,
        "TW_mil": TW_mil_history,
        "TW_wet": TW_wet_history,
        "WS": WS_history,
        "S": S_history,
    }
    TW_cruise_history = np.array(TW_cruise_history)
    TW_mil_history = np.array(TW_mil_history)
    TW_wet_history = np.array(TW_wet_history)
    WS_history = np.array(WS_history)
    S_history = np.array(S_history)
    with open("results.json", "w") as file:
        json.dump(results_dict, file, indent=4)

    # Diagnostics

    WS_plot_min = np.maximum(np.minimum(WS_history.min(), WSref), 0.0)
    WS_plot_max = np.maximum(np.maximum(WS_history.max(), WSref), 0.0)
    WS_span = WS_plot_max - WS_plot_min
    WS_range = np.linspace(
        np.maximum(WS_plot_min - 0.1 * WS_span, 10.0), WS_plot_max + 0.1 * WS_span
    )
    TW_cruise = TW_cruise_history[-1]  # We pick the optimal value
    TW_mil = TW_mil_history[-1]  # We pick the optimal value
    TW_wet = TW_wet_history[-1]  # We pick the optimal value
    WS = WS_history[-1]  # We pick the optimal value
    S = S_history[-1]  # We pick the optimal value
    x = np.array([TW_cruise, TW_mil, TW_wet, WS, S]) / np.concatenate(
        [TWref, np.array([WSref, Sref])]
    )

    plt.figure(1)
    plt.plot(WS_history, TW_cruise_history, label="optim. path", color="gray")
    plt.scatter([WSref], [TWref[0]], label="ref.", color="blue")
    plt.scatter([WS], [TW_cruise], label="end", color="red")
    WF_3 = Wref * phase_3(x)
    beta_3 = comp_beta(WF_3, WS, S)
    alpha_4 = cruise_alpha_model(h1_4)
    rho_4 = isa.rho_ISA(h1_4)
    q_4 = aero.compute_q(V_4, rho_4)
    dhdt_4 = V_4 * (h1_4 - h0_4) / np.sqrt(s_4**2 + (h1_4 - h0_4) ** 2)
    TW_cruise_4 = cst.constant_speed_climb(
        WS_range, alpha_4, beta_3, K1, K2, CD0, dhdt_4, V_4, q_4
    )
    plt.fill_between(WS_range, TW_cruise_4, 0.0, label="phase 4", alpha=0.3)
    WF_4 = Wref * phase_4(x)
    beta_4 = comp_beta(WF_4, WS, S)
    alpha_5 = cruise_alpha_model(h_5)
    rho_5 = isa.rho_ISA(h_5)
    q_5 = aero.compute_q(V_5, rho_5)
    TW_cruise_5 = cst.constant_altitude_speed_cruise(
        WS_range, alpha_5, beta_4, K1, K2, CD0, q_5
    )
    plt.fill_between(WS_range, TW_cruise_5, 0.0, label="phase 5", alpha=0.3)
    WF_7 = Wref * phase_7(x)
    beta_7 = comp_beta(WF_7, WS, S)
    alpha_8 = cruise_alpha_model(h_8)
    rho_8 = isa.rho_ISA(h_8)
    q_8 = aero.compute_q(V_8, rho_8)
    TW_cruise_8 = cst.constant_altitude_speed_cruise(
        WS_range, alpha_8, beta_7, K1, K2, CD0, q_8
    )
    plt.fill_between(WS_range, TW_cruise_8, 0.0, label="phase 8", alpha=0.3)
    WF_8 = Wref * phase_8(x)
    beta_8 = comp_beta(WF_8, WS, S)
    alpha_9 = cruise_alpha_model(h_9)
    rho_9 = isa.rho_ISA(h_9)
    q_9 = aero.compute_q(V_9, rho_9)
    TW_cruise_9 = cst.constant_altitude_speed_cruise(
        WS_range, alpha_9, beta_8, K1, K2, CD0, q_9
    )
    plt.fill_between(WS_range, TW_cruise_8, 0.0, label="phase 8", alpha=0.3)
    plt.legend()
    plt.grid()
    plt.ylabel("$T_{SL}/W_{TO}$ [-]")
    plt.xlabel("$W_{TO}/S$ [Pa]")
    plt.title("Constraints plot - Cruise")
    plt.ylim(-0.1, None)
    plt.legend(loc="upper left", bbox_to_anchor=(1.02, 1))
    plt.tight_layout()
    plt.savefig("cst_cruise.png")

    plt.figure(2)
    plt.plot(WS_history, TW_mil_history, label="optim. path", color="gray")
    plt.scatter([WSref], [TWref[1]], label="ref.", color="blue")
    plt.scatter([WS], [TW_mil], label="end", color="red")
    WF_2 = Wref * phase_2(x)
    beta_2 = comp_beta(WF_2, WS, S)
    alpha_3 = mil_alpha_model(h1_3)
    rho_3 = isa.rho_ISA(h1_3)
    q_3 = aero.compute_q(V_3, rho_3)
    TW_mil_3 = cst.constant_speed_climb(
        WS_range, alpha_3, beta_2, K1, K2, CD0, dhdt_3, V_3, q_3
    )
    plt.fill_between(WS_range, TW_mil_3, 0.0, label="phase 3", alpha=0.3)
    plt.legend()
    plt.grid()
    plt.ylabel("$T_{SL}/W_{TO}$ [-]")
    plt.xlabel("$W_{TO}/S$ [Pa]")
    plt.title("Constraints plot - Military")
    plt.ylim(-0.1, None)
    plt.legend(loc="upper left", bbox_to_anchor=(1.02, 1))
    plt.tight_layout()
    plt.savefig("cst_mil.png")

    plt.figure(3)
    plt.plot(WS_history, TW_wet_history, label="optim. path", color="gray")
    plt.scatter([WSref], [TWref[2]], label="ref.", color="blue")
    plt.scatter([WS], [TW_wet], label="end", color="red")
    CD_1 = aero.drag_polar(CL_1, K1, K2, CD0) * drag_penalty_1
    xi_1b = rw.compute_xi(CD_1, CDR, mu_1b, CL_1)
    TW_wet_1b = cst.takeoff_ground_roll_low_thrust(
        WS_range, alpha_1b, beta_1a, xi_1b, mu_1b, CL_1, sTO_max, k_1b, rho_1b, g0
    )
    plt.fill_between(WS_range, TW_wet_1b, 0.0, label="phase 1", alpha=0.3)
    WF_1 = Wref * phase_1c(x)
    beta_1 = comp_beta(WF_1, WS, S)
    V_1c = k_1b * aero.compute_Vstall(WS, beta_1, CL_2, rho_1c)
    q_2 = aero.compute_q(V_1c, rho_2)
    TW_wet_2 = (
        cst.horizontal_acceleration(WS_range, alpha_2, beta_1, K1, K2, CD0, dVdt_2, q_2)
        * drag_penalty_1
    )
    plt.fill_between(WS_range, TW_wet_2, 0.0, label="phase 2", alpha=0.3)
    WF_5 = Wref * phase_5(x)
    beta_5 = comp_beta(WF_5, WS, S)
    alpha_6 = cruise_alpha_model(h1_6)
    rho_6 = isa.rho_ISA(h1_6)
    q_6 = aero.compute_q(V_6, rho_6)
    TW_wet_6 = cst.constant_speed_climb(
        WS_range, alpha_6, beta_5, K1, K2, CD0, dhdt_6, V_6, q_6
    )
    plt.fill_between(WS_range, TW_wet_6, 0.0, label="phase 6", alpha=0.3)
    WF_6 = Wref * phase_6(x)
    beta_6 = comp_beta(WF_6, WS, S)
    alpha_7 = wet_alpha_model(h_7)
    rho_7 = isa.rho_ISA(h_7)
    q_7 = aero.compute_q(V_7, rho_7)
    n_7 = dyn.compute_n(V_7, g0, RC_7)
    TW_wet_7 = cst.constant_altitude_speed_turn(
        WS_range, alpha_7, beta_6, K1, K2, CD0, q_7, n_7
    )
    plt.fill_between(WS_range, TW_wet_7, 0.0, label="phase 7", alpha=0.3)
    WF_9 = Wref * phase_9(x)
    beta_9 = comp_beta(WF_9, WS, S)
    CD_10 = aero.drag_polar(CL_10, K1, K2, CD0)
    xi_10 = rw.compute_xi(CD_10, CDR, mu_10, CL_10)
    TW_wet_10 = cst.braking_roll(
        WS_range, alpha_10, beta_9, xi_10, mu_10, CLmax, sL_max, k_10, rho_10, g0
    )
    plt.fill_between(WS_range, TW_wet_10, 0.0, label="phase 10", alpha=0.3)
    plt.legend()
    plt.grid()
    plt.ylabel("$T_{SL}/W_{TO}$ [-]")
    plt.xlabel("$W_{TO}/S$ [Pa]")
    plt.title("Constraints plot - Wet")
    plt.ylim(-0.1, None)
    plt.legend(loc="upper left", bbox_to_anchor=(1.02, 1))
    plt.tight_layout()
    plt.savefig("cst_wet.png")

    plt.figure(4)
    plt.bar(
        [f"phase {i}" for i in range(1, 10 + 1)],
        [
            beta_1,
            beta_2,
            beta_3,
            beta_4,
            beta_5,
            beta_6,
            beta_7,
            beta_8,
            beta_9,
            beta_9,
        ],
    )
    plt.ylabel(r"$\beta$")
    plt.grid(axis="y")
    plt.title("Total weight fraction evolution during design mission")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("beta.png")

    plt.figure(5)
    WTO = WS * S
    WF0_total = comp_WF0(WS, S) / (1 - min_fuel)
    WE = WTO * weights.wE_fighter_table(WTO)
    plt.pie(
        [WE, WP, WF0_total],
        labels=["Empty weight", "Total payload weight", "Initial fuel weight"],
        autopct="%1.1f%%",
    )
    plt.title(f"Initial weight breakdown (total {(WTO / g0 / 1000):.2f} t)")
    plt.tight_layout()
    plt.savefig("weight_breakdown.png")
    print(
        "Sum of all weights FYI (should be close to WTO on plot):",
        (WE + WP + WF0_total) / g0 / 1000,
        "t",
    )

    plt.show()
