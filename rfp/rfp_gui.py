# 3rd party
from typing import Callable
import numpy as np
import json
import matplotlib.pyplot as plt
import streamlit as st

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

st.title("RFP - Single aisle aircraft")

left_col, right_col = st.columns(2)

with left_col:
    st.header("Design parameters")

    st.subheader("Aircraft")

    # Global: requirements
    with st.expander("General requirements"):
        WP = isa.g_ISA * (
            weights.compute_WP(
                st.number_input("Number of passengers", value=180 + 6), long_flight=True
            )
        ) * (
            1 + st.number_input("Payload margin in %", value=10) / 100
        )  # N, ASSUMPTION (10% margin)
        min_fuel = st.number_input("Min. fuel reserve in %", value=5) / 100
        sTO_max = units.ft2m(
            st.number_input("Max. takeoff distance in ft", value=6500)
        )  # m
        sL_max = units.ft2m(
            st.number_input("Max. landing distance in ft", value=6000)
        )  # m
        mission_range = (
            units.nm2km(st.number_input("Min. range in nmi", value=2855)) * 1000
        )  # m

    # Global: aerodynamics
    with st.expander("Aerodynamics"):
        CLmax = st.number_input("$C_L^{max}$", value=1.3)
        AR = st.number_input("Aspect ratio $AR$", value=10.0)
        e = st.number_input("Oswald efficiency factor $e$", value=0.85)
        CD0 = st.number_input("$1000 C_D^0$", value=20.0) * 1e-3
        K1 = 1.0 / (np.pi * e * AR)
        K2 = 0.0
        CL_max_LD = np.sqrt(CD0 / K1)
        CD_max_LD = CD0 + K1 * CL_max_LD**2
        r"""Reminder (quadratic drag polar model):
        $$
        C_{D} = K_1 C_L^2 + K_2 C_L + C_D^0
        $$
        """
        CDR = st.number_input(
            "$C_D^R$", value=0.0
        )  # ASSUMPTION (drag from landing gear is negligible)

    # Global: physics
    g0 = isa.g_ISA  # m.s-2

    # Global: propulsion

    with st.expander("Propulsion"):
        r"""
        * High bypass turbofan engine, model from Mattingly et al. 2002, p. 71, eq. 3.54:
        $$
        TSFC = (0.45 + 0.54 M) \sqrt{\theta}
        $$
        with $M < 0.9$
        * Lapse rate model from Mavris AE6343 lecture at Georgia Tech.
        """

    def cruise_alpha_model(h, V):
        sigma = isa.sigma_ISA(h)
        T = isa.T_ISA(h)
        M = V / isa.compute_cs(T)
        return prop.alpha_HBP_turbofan(M, sigma, bypass_check=False)

    def cruise_tsfc_model(h, V):
        theta = isa.theta_ISA(h)
        T = isa.T_ISA(h)
        M = V / isa.compute_cs(T)
        TSFC = (0.45 + 0.54 * M) * np.sqrt(theta) / 3600
        return TSFC  # s-1

    ## Phases ##

    st.subheader("Mission")

    # Phase 1a: warm up
    with st.expander("Phase 1: Takeoff"):
        TSFC_1a = cruise_tsfc_model(h=0.0, V=0.0)  # s-1
        alpha_1a = 1.0
        beta_1a = 1.0
        t_1a = st.number_input(
            "Warm up time in s", value=80.0
        )  # s, ASSUMPTION (engine takes similar time to warm up than for F86)

        # Phase 1b: takeoff ground roll
        rho_1b = isa.rho0_ISA  # kg.m-3
        k_1b = st.number_input(
            "$k = V_{TO} / V_{stall}$", value=1.1, key="k_1b"
        )  # ASSUMPTION (10% margin over Vstall for takeoff)
        alpha_1b = alpha_1a
        TSFC_1b = cruise_tsfc_model(h=0.0, V=0.0)  # s-1
        HL_mult_1 = st.number_input(
            r"High lift $C_L^{max}$ multiplier", value=1.3, key="HL_mult_1"
        )
        CL_1 = (
            CLmax * HL_mult_1
        )  # ASSUMPTION (high lift devices, increasing lift by 30%)
        drag_penalty_1 = st.number_input(
            "High lift drag multiplier",
            value=1.35,
            key="drag_penalty_1",  # ASSUMPTION (drag is increased by 35% using high lift devices)
        )
        mu_1b = st.number_input(
            r"$\mu$", value=0.02, key="mu_1b"
        )  # ASSUMPTION (from lecture slides)
        s_1 = (
            st.number_input("$s_{GR} / s_{TO}^{max}$", value=0.9) * sTO_max
        )  # ASSUMPTION (10% margin)

        # Phase 1c: takeoff rotation
        rho_1c = isa.rho0_ISA  # kg.m-3
        alpha_1c = alpha_1b
        TSFC_1c = TSFC_1b  # s-1
        tR_1c = st.number_input(
            "$t_R$ in s", value=3.0
        )  # s, ASSUMPTION (from lecture slides)
        hOBS_1c = units.ft2m(st.number_input("$h_{obs}$ in ft", value=35))  # m

    # Phase 2: clearing climb
    with st.expander("Phase 2: Clearing climb"):
        """
        Set parameters:
        * Lift: $C_L^{max}$ without high lift devices
        """
        dhdt_2 = units.ft2m(
            st.number_input("Climb rate in ft/s", value=90 * 0.8, key="dhdt_2")
        )  # m/s, ASSUMTION (maximum climb rate similar than with F86)
        h0_2 = 0.0  # m/s
        h1_2 = units.ft2m(
            st.number_input("$h_{max}$ in ft", value=10000, key="h1_2")
        )  # m
        n_split_2 = st.number_input(
            "$n_{split}$ (number of subdivisions of the climb for more accurate integration)",
            value=10,
        )  # ASSUMPTION (alpha relatively constant on 1000 ft intervals)
        CL_2 = CLmax  # ASSUMPTION (climbing with CLmax and no high lift devices)
        M1_2 = st.number_input("Target $M$", value=0.3, key="M1_2")

    # Phase 3: first climb
    with st.expander("Phase 3: First climb"):
        """
        Set parameters:
        * Aerodynamics: max. $L/D$
        * Climb rate: same than phase 2
        """
        h0_3 = h1_2  # m
        h1_3 = units.ft2m(
            st.number_input("$h_{max}$ in ft", value=35000, key="h1_3")
        )  # m
        CD_3 = CD_max_LD  # ASSUMPTION (economy)
        CL_3 = CL_max_LD  # ASSUMPTION (economy)
        M0_3 = M1_2
        M1_3 = st.number_input("Target $M$", value=0.78, key="M1_3")
        n_split_3 = st.number_input(
            "$n_{split}$ (number of subdivisions of the climb for more accurate integration)",
            value=25,
            key="n_split_3",
        )  # ASSUMPTION (alpha relatively constant on 1000 ft intervals)
        dhdt_3 = dhdt_2  # ASSUMPTION (continuity)

    # Phase 4: first cruise
    with st.expander("Phase 4: First cruise"):
        """
        Set parameters:
        * Aerodynamics: max. $L/D$
        * Speed: same than end of phase 3
        """
        h_4 = units.ft2m(st.number_input("$h$ in ft", value=35000, key="h_4"))  # m
        M_4 = M1_3
        CD_4 = CD_max_LD  # ASSUMPTION (economy)
        CL_4 = CL_max_LD  # ASSUMPTION (economy)
        s_4 = (
            st.number_input("$s_1 / s_{mission}$", value=0.5) * mission_range
        )  # m, ASSUMPTION (first cruise should cover 50% of the mission range)

    # Phase 5: second climb
    with st.expander("Phase 5: Second climb"):
        """
        Set parameters:
        * Lift: $C_L^{max}$ without high lift devices
        * Climb rate: same than phase 3
        * Speed: same than pahse 4
        """
        h0_5 = h_4  # m
        h1_5 = units.ft2m(
            st.number_input("$h_{max}$ in ft", value=37000, key="h1_5")
        )  # m
        M_5 = M_4
        n_split_5 = st.number_input(
            "$n_{split}$ (number of subdivisions of the climb for more accurate integration)",
            value=2,
        )  # ASSUMPTION (alpha relatively constant on 1000 ft intervals)
        dhdt_5 = dhdt_3  # ASSUMPTION (continuity)
        CL_5 = CLmax  # ASSUMPTION (second climb performed like first climb)

    # Phase 6: second cruise
    with st.expander("Phase 6: Second cruise"):
        """
        Set parameters:
        * Aerodynamics: max. $L/D$
        * Speed: same than phase 5
        """
        h_6 = h1_5  # m
        M_6 = M_5
        s_6 = (
            0.5 * mission_range
        )  # m, ASSUMPTION (second cruise should cover 50% of the mission range)
        CL_6 = CL_max_LD  # ASSUMPTION (economy)
        CD_6 = CD_max_LD  # ASSUMPTION (economy)

    with st.expander("Phase 7: Descent"):
        """
        Assumptions:
        * Gliding descent
        * Speed reduced using aero braking
        """
        pass

    # Phase 8: loiter
    with st.expander("Phase 8: Loiter"):
        """
        Assumptions:
        * The load factor requirement has the most impact during the final loiter phase
        """
        t_8 = st.number_input("$t$ in min", value=30, key="t_8") * 60  # s
        s_8 = (
            units.nm2km(st.number_input("$s$ in nmi", value=200, key="s_8")) * 1000
        )  # m
        h_8 = units.ft2m(st.number_input("$h$ in ft", value=5000, key="h_8"))  # m
        M_8 = st.number_input("$M$", value=0.3)
        n_8 = st.number_input("$n$ in $g$", value=3)

    # Phase 9: landing
    with st.expander("Phase 9: Landing"):
        Vapp_9 = units.kt2ms(st.number_input("$V_{app}$ in kt", value=130))  # m.s-1
        s_9 = (
            st.number_input("$s_{B} / s_{L}^{max}$", value=0.9) * sL_max
        )  # m, ASSUMPTION (10% margin)
        k_9 = st.number_input(
            "$k = V_{L} / V_{stall}$", value=1.1
        )  # ASSUMPTION (10% margin over Vstall for landing)
        CL_9 = (
            st.number_input(
                "High lift $C_L^{max}$ multiplier", value=1.5, key="HL_mult_9"
            )
            * CLmax
        )  # ASSUMPTION (high lift devices during landing increase lift by 50%)
        drag_penalty_9 = st.number_input(
            "High lift drag multiplier", value=1.6, key="drag_penalty_9"
        )  # ASSUMPTION (drag increases by 60% due to high lift devices)
        rho_9 = isa.rho0_ISA
        alpha_9 = st.number_input(
            r"$\alpha", value=0.65, key="alpha_9"
        )  # ASSUMPTION (65% of thrust can be reversed for landing)
        mu_9 = st.number_input(
            r"\mu", value=10 * 0.02, key="mu_9"
        )  # ASSUMPTION (Coulomb friction coefficient is 10x greater due to braking than during takeoff)

    ## Optimization settings ##

    st.subheader("Optimization")

    with st.expander("Hyperparameters"):
        lr_val = st.number_input("Learning rate value", value=5)
        lr_mag = st.number_input("Learning rate magnitude", value=-2)
        lr = lr_val * 10**lr_mag
        n_epochs = st.number_input("Epochs", value=500)
        fd_step = 1e-6
        mission_penalty_fun = lambda x: 100 * np.maximum(-x + 0.001, 0.0) ** 2
        constraints_penalty_fun = lambda x: 3 * np.maximum(-x + 0.05, 0.0) ** 2
        sanity_penalty_fun = lambda x: 0.0  # no sanity penalty (for positivity)

    with st.expander("Initialization"):
        Wref = (
            st.number_input("$W_{ref}$ in t", value=79) * 1000 * g0
        )  # N, (max. TO weight of A320 neo, similar specs)
        Tref_cruise = units.lbf2N(
            st.number_input("$T_{ref}$ in lbf", value=2 * 28000)
        )  # N
        Sref = st.number_input(
            "$S_{ref}$ in m$^2$", value=141.25
        )  # m2  # (estimated from official technical sheet of A320, pp. 46-47 on the geometry of neo)
        TWref_cruise = Tref_cruise / Wref
        TWref = [TWref_cruise]
        WSref = Wref / Sref  # Pa
        TW0 = TWref
        WS0 = WSref
        S0 = Sref

    run_optim = st.button(label="Run", width="stretch")

## Utils ##


def comp_WF0(WS, S):
    WTO = WS * S
    WE = weights.wE_cargo_table(WTO) * WTO
    WF_total = WTO - WE - WP
    WF = WF_total * (1 - min_fuel)
    return WF


def comp_beta(WF, WS, S):
    WTO = WS * S
    WF0_total = comp_WF0(WS, S) / (1 - min_fuel)
    WE = weights.wE_cargo_table(WTO) * WTO
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
    WE = weights.wE_cargo_table(WTO) * WTO
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
    beta_before = comp_beta(WF, WS, S)
    return TW, WS, S, beta_before


## Mission analysis constraints ##


def phase_1a(x: np.ndarray) -> float:
    TW_cruise = x[:-2] * TWref
    WS = x[-2] * WSref
    S = x[-1] * Sref
    WTO = WS * S
    WE = weights.wE_cargo_table(WTO) * WTO
    WF0_total = WTO - WE - WP
    beta_instant = mission.warm_up(TW_cruise, alpha_1a, beta_1a, TSFC_1a, t_1a)
    W = beta_instant * WTO
    WF_total = W - WP - WE
    WF = WF_total - min_fuel * WF0_total
    return WF / Wref


def phase_1b(x: np.ndarray) -> float:
    TW, WS, S, WTO, beta_before = phase_header(x, phase_1a)
    TW_cruise = TW[0]
    Vstall = aero.compute_Vstall(WS, beta_before, CL_1, rho_1b)
    VTO = k_1b * Vstall
    CD = aero.drag_polar(CLmax, K1, K2, CD0) * drag_penalty_1
    xi = rw.compute_xi(CD, CDR, mu_1b, CL_1)
    q = aero.compute_q(VTO, rho_1b)
    u = prop.compute_u_takeoff_acceleration(
        TW_cruise, WS, alpha_1b, beta_before, xi, mu_1b, q
    )
    beta_instant = mission.takeoff_acceleration(VTO, u, TSFC_1b, g0)
    WF = phase_footer(WS, S, beta_instant, beta_before)
    return WF / Wref


def phase_1c(x: np.ndarray) -> float:
    TW, WS, S, WTO, beta_before = phase_header(x, phase_1b)
    TW_cruise = TW[0]
    beta_instant = mission.takeoff_rotation(
        TW_cruise, alpha_1c, beta_before, TSFC_1c, tR_1c
    )
    WF = phase_footer(WS, S, beta_instant, beta_before)
    return WF / Wref


def phase_2(x: np.ndarray) -> float:
    TW, WS, S, WTO, beta_before = phase_header(x, phase_1c)
    TW_cruise = TW[0]
    h_range = np.linspace(h0_2, h1_2, n_split_2 + 1)
    Vstall = aero.compute_Vstall(WS, beta_1a, CL_1, rho_1b)
    V0_2 = k_1b * Vstall
    T1_2 = isa.T_ISA(h1_2)
    V1_2 = M1_2 * isa.compute_cs(T1_2)
    V_range = np.linspace(V0_2, V1_2, n_split_2 + 1)
    CD_2 = aero.drag_polar(CL_2, K1, K2, CD0)
    beta_current = beta_before
    for i in range(n_split_2):
        h0 = h_range[i]
        h1 = h_range[i + 1]
        V1 = V_range[i + 1]
        alpha = cruise_alpha_model(h1, V1)
        TSFC = cruise_tsfc_model(h0, V1)
        beta_current *= mission.constant_speed_climb(
            TW_cruise, h0, h1, alpha, beta_current, CD_2, CL_2, V1, TSFC
        )
    beta_instant = beta_current / beta_before
    WF = phase_footer(WS, S, beta_instant, beta_before)
    return WF / Wref


def phase_3(x: np.ndarray) -> float:
    TW, WS, S, WTO, beta_before = phase_header(x, phase_2)
    TW_cruise = TW[0]
    h_range = np.linspace(h0_3, h1_3, n_split_3 + 1)
    T1_3 = isa.T_ISA(h1_3)
    V1_3 = M1_3 * isa.compute_cs(T1_3)
    T0_3 = isa.T_ISA(h0_3)
    V0_3 = M0_3 * isa.compute_cs(T0_3)
    V_range = np.linspace(V0_3, V1_3, n_split_3 + 1)
    beta_current = beta_before
    for i in range(n_split_3):
        h0 = h_range[i]
        h1 = h_range[i + 1]
        V1 = V_range[i + 1]
        alpha = cruise_alpha_model(h1, V1)
        TSFC = cruise_tsfc_model(h0, V1)
        beta_current *= mission.constant_speed_climb(
            TW_cruise, h0, h1, alpha, beta_current, CD_3, CL_3, V1, TSFC
        )
    beta_instant = beta_current / beta_before
    WF = phase_footer(WS, S, beta_instant, beta_before)
    return WF / Wref


def phase_4(x: np.ndarray) -> float:
    TW, WS, S, WTO, beta_before = phase_header(x, phase_3)
    TW_cruise = TW[0]
    CD = aero.drag_polar(CL_4, K1, K2, CD0)
    T = isa.T_ISA(h_4)
    V = M_4 * isa.compute_cs(T)
    TSFC = cruise_tsfc_model(h_4, V)
    beta_instant = mission.constant_altitude_speed_cruise(s_4, CD, CL_4, V, TSFC)
    WF = phase_footer(WS, S, beta_instant, beta_before)
    return WF / Wref


def phase_5(x: np.ndarray) -> float:
    TW, WS, S, WTO, beta_before = phase_header(x, phase_4)
    TW_cruise = TW[0]
    h_range = np.linspace(h0_5, h1_5, n_split_5 + 1)
    CD_5 = aero.drag_polar(CL_5, K1, K2, CD0)
    beta_current = beta_before
    for i in range(n_split_5):
        h0 = h_range[i]
        h1 = h_range[i + 1]
        T1 = isa.T_ISA(h1)
        V1 = M_5 * isa.compute_cs(T1)
        alpha = cruise_alpha_model(h1, V1)
        TSFC = cruise_tsfc_model(h0, V1)
        beta_current *= mission.constant_speed_climb(
            TW_cruise, h0, h1, alpha, beta_current, CD_5, CL_5, V1, TSFC
        )
    beta_instant = beta_current / beta_before
    WF = phase_footer(WS, S, beta_instant, beta_before)
    return WF / Wref


def phase_6(x: np.ndarray) -> float:
    TW, WS, S, WTO, beta_before = phase_header(x, phase_5)
    TW_cruise = TW[0]
    T = isa.T_ISA(h_6)
    V = M_6 * isa.compute_cs(T)
    TSFC = cruise_tsfc_model(h_6, V)
    beta_instant = mission.constant_altitude_speed_cruise(s_6, CD_6, CL_6, V, TSFC)
    WF = phase_footer(WS, S, beta_instant, beta_before)
    return WF / Wref


def phase_8(x: np.ndarray) -> float:
    TW, WS, S, WTO, beta_before = phase_header(x, phase_6)
    TW_cruise = TW[0]
    T = isa.T_ISA(h_8)
    V = M_8 * isa.compute_cs(T)
    TSFC = cruise_tsfc_model(h_8, V)
    beta_instant = mission.loiter(K1, K2, CD0, TSFC, t_8)
    WF = phase_footer(WS, S, beta_instant, beta_before)
    return WF / Wref


## Energy analysis constraints ##


def constraint_1(x: np.ndarray) -> float:
    TW, WS, S, beta_before = constraint_header(x, phase_1a)
    TW_cruise = TW[0]
    CD = aero.drag_polar(CLmax, K1, K2, CD0) * drag_penalty_1
    xi = rw.compute_xi(CD, CDR, mu_1b, CL_1)
    TWmin = cst.takeoff_ground_roll_low_thrust(
        WS, alpha_1b, beta_before, xi, mu_1b, CL_1, s_1, k_1b, rho_1b, g0
    )
    return (TW_cruise - TWmin) / TWref_cruise


def constraint_2(x: np.ndarray) -> float:
    TW, WS, S, beta_before = constraint_header(x, phase_1c)
    TW_cruise = TW[0]
    T1 = isa.T_ISA(h1_2)
    V1 = M1_2 * isa.compute_cs(T1)
    alpha = cruise_alpha_model(h1_2, V1)
    rho = isa.rho_ISA(h1_2)
    q = aero.compute_q(V1, rho)
    TWmin = cst.constant_speed_climb(WS, alpha, beta_before, K1, K2, CD0, dhdt_2, V1, q)
    return (TW_cruise - TWmin) / TWref_cruise


def constraint_3(x: np.ndarray) -> float:
    TW, WS, S, beta_before = constraint_header(x, phase_2)
    TW_cruise = TW[0]
    T1 = isa.T_ISA(h1_3)
    V1 = M1_3 * isa.compute_cs(T1)
    alpha = cruise_alpha_model(h1_3, V1)
    rho = isa.rho_ISA(h1_3)
    q = aero.compute_q(V1, rho)
    TWmin = cst.constant_speed_climb(WS, alpha, beta_before, K1, K2, CD0, dhdt_3, V1, q)
    return (TW_cruise - TWmin) / TWref_cruise


def constraint_4(x: np.ndarray) -> float:
    TW, WS, S, beta_before = constraint_header(x, phase_3)
    TW_cruise = TW[0]
    T = isa.T_ISA(h_4)
    V = M_4 * isa.compute_cs(T)
    alpha = cruise_alpha_model(h_4, V)
    rho = isa.rho_ISA(h_4)
    q = aero.compute_q(V, rho)
    TWmin = cst.constant_altitude_speed_cruise(WS, alpha, beta_before, K1, K2, CD0, q)
    return (TW_cruise - TWmin) / TWref_cruise


def constraint_5(x: np.ndarray) -> float:
    TW, WS, S, beta_before = constraint_header(x, phase_4)
    TW_cruise = TW[0]
    T1 = isa.T_ISA(h1_5)
    V1 = M_5 * isa.compute_cs(T1)
    alpha = cruise_alpha_model(h1_5, V1)
    rho = isa.rho_ISA(h1_5)
    q = aero.compute_q(V1, rho)
    TWmin = cst.constant_speed_climb(WS, alpha, beta_before, K1, K2, CD0, dhdt_5, V1, q)
    return (TW_cruise - TWmin) / TWref_cruise


def constraint_6(x: np.ndarray) -> float:
    TW, WS, S, beta_before = constraint_header(x, phase_5)
    TW_cruise = TW[0]
    T = isa.T_ISA(h_6)
    V = M_6 * isa.compute_cs(T)
    alpha = cruise_alpha_model(h_6, V)
    rho = isa.rho_ISA(h_6)
    q = aero.compute_q(V, rho)
    TWmin = cst.constant_altitude_speed_cruise(WS, alpha, beta_before, K1, K2, CD0, q)
    return (TW_cruise - TWmin) / TWref_cruise


def constraint_8(x: np.ndarray) -> float:
    TW, WS, S, beta_before = constraint_header(x, phase_6)
    TW_cruise = TW[0]
    T = isa.T_ISA(h_8)
    V = M_8 * isa.compute_cs(T)
    alpha = cruise_alpha_model(h_8, V)
    rho = isa.rho_ISA(h_8)
    q = aero.compute_q(V, rho)
    TWmin = cst.constant_altitude_speed_turn(
        WS, alpha, beta_before, K1, K2, CD0, q, n_8
    )
    return (TW_cruise - TWmin) / TWref_cruise


def constraint_9a(x: np.ndarray) -> float:  # Approach
    TW, WS, S, beta_before = constraint_header(x, phase_8)
    TW_cruise = TW[0]
    V = Vapp_9 * k_9
    WSmax = cst.non_stall_landing(beta_before, CLmax, V, rho_9)
    return (WSmax - WS) / WSref


def constraint_9b(x: np.ndarray) -> float:  # Landing field
    TW, WS, S, beta_before = constraint_header(x, phase_8)
    TW_cruise = TW[0]
    CD = aero.drag_polar(CLmax, K1, K2, CD0) * drag_penalty_9
    xi = rw.compute_xi(CD, CDR, mu_9, CL_9)
    TWmin = cst.braking_roll(
        WS, alpha_9, beta_before, xi, mu_9, CLmax, s_9, k_9, rho_9, g0
    )
    return (TW_cruise - TWmin) / TWref_cruise


def main():
    ## Optimization ##

    # Execution
    history = optim.minimum_fuel_design(
        TW0=TW0,
        WS0=WS0,
        S0=S0,
        WP=WP,
        empty_weight_fun=weights.wE_cargo_table,
        constraints=[
            constraint_1,
            constraint_2,
            constraint_3,
            constraint_4,
            constraint_5,
            constraint_6,
            constraint_8,
            constraint_9a,
            constraint_9b,
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
            phase_8,
        ],
        min_fuel=min_fuel,
        TWref=TWref,
        WSref=WSref,
        Sref=Sref,
        n_epochs=n_epochs,
        lr=lr,
        fd_step=fd_step,
        mission_penalty_fun=mission_penalty_fun,
        constraint_penalty_fun=constraints_penalty_fun,
        sanity_penalty_fun=sanity_penalty_fun,
        return_path=True,
        verbose=True,
        print_path=True,
        normalize_print=True,
        _gui=True,
    )
    TW_cruise_history = []
    WS_history = []
    S_history = []
    for TW, WS, S in history:
        TW_cruise_history.append(TW[0])
        WS_history.append(WS)
        S_history.append(S)
    results_dict = {
        "TW_cruise": TW_cruise_history,
        "WS": WS_history,
        "S": S_history,
    }
    TW_cruise_history = np.array(TW_cruise_history)
    WS_history = np.array(WS_history)
    S_history = np.array(S_history)

    # Diagnostics

    WS_plot_min = np.maximum(np.minimum(WS_history.min(), WSref), 0.0)
    WS_plot_max = np.maximum(np.maximum(WS_history.max(), WSref), 0.0)
    WS_span = WS_plot_max - WS_plot_min
    WS_range = np.linspace(
        np.maximum(WS_plot_min - 0.1 * WS_span, 10.0), WS_plot_max + 0.1 * WS_span
    )
    TW_cruise = TW_cruise_history[-1]  # We pick the optimal value
    WS = WS_history[-1]  # We pick the optimal value
    S = S_history[-1]  # We pick the optimal value
    x = np.array([TW_cruise, WS, S]) / np.concatenate([TWref, np.array([WSref, Sref])])

    st.subheader("Constraints plot")
    """
    * Phase 9a: Non-stall condition during approach
    * Phase 9b: Braking roll
    """
    fig = plt.figure(1)
    plt.plot(WS_history, TW_cruise_history, label="optim. path", color="gray")
    plt.scatter([WSref], [TWref[0]], label="ref.", color="blue")
    plt.scatter([WS], [TW_cruise], label="end", color="red")
    WF_1a = Wref * phase_1a(x)
    beta_1a = comp_beta(WF_1a, WS, S)
    CD_1 = aero.drag_polar(CLmax, K1, K2, CD0) * drag_penalty_1
    xi_1 = rw.compute_xi(CD_1, CDR, mu_1b, CL_1)
    TW_cruise_1 = cst.takeoff_ground_roll_low_thrust(
        WS_range, alpha_1b, beta_1a, xi_1, mu_1b, CL_1, s_1, k_1b, rho_1b, g0
    )
    plt.fill_between(WS_range, TW_cruise_1, 0.0, label="phase 1", alpha=0.3)
    WF_1c = Wref * phase_1c(x)
    beta_1 = comp_beta(WF_1c, WS, S)
    T1_2 = isa.T_ISA(h1_2)
    V1_2 = M1_2 * isa.compute_cs(T1_2)
    alpha_2 = cruise_alpha_model(h1_2, V1_2)
    rho1_2 = isa.rho_ISA(h1_2)
    q1_2 = aero.compute_q(V1_2, rho1_2)
    TW_cruise_2 = cst.constant_speed_climb(
        WS_range, alpha_2, beta_1, K1, K2, CD0, dhdt_2, V1_2, q1_2
    )
    plt.fill_between(WS_range, TW_cruise_2, 0.0, label="phase 2", alpha=0.3)
    WF_2 = Wref * phase_2(x)
    beta_2 = comp_beta(WF_2, WS, S)
    T1_3 = isa.T_ISA(h1_3)
    V1_3 = M1_3 * isa.compute_cs(T1_3)
    alpha_3 = cruise_alpha_model(h1_3, V1_3)
    rho1_3 = isa.rho_ISA(h1_3)
    q1_3 = aero.compute_q(V1_3, rho1_3)
    TW_cruise_3 = cst.constant_speed_climb(
        WS_range, alpha_3, beta_2, K1, K2, CD0, dhdt_3, V1_3, q1_3
    )
    plt.fill_between(WS_range, TW_cruise_3, 0.0, label="phase 3", alpha=0.3)
    WF_3 = Wref * phase_3(x)
    beta_3 = comp_beta(WF_3, WS, S)
    T_4 = isa.T_ISA(h_4)
    V_4 = M_4 * isa.compute_cs(T_4)
    alpha_4 = cruise_alpha_model(h_4, V_4)
    rho_4 = isa.rho_ISA(h_4)
    q_4 = aero.compute_q(V_4, rho_4)
    TW_cruise_4 = cst.constant_altitude_speed_cruise(
        WS_range, alpha_4, beta_3, K1, K2, CD0, q_4
    )
    plt.fill_between(WS_range, TW_cruise_4, 0.0, label="phase 4", alpha=0.3)
    WF_4 = Wref * phase_4(x)
    beta_4 = comp_beta(WF_4, WS, S)
    T1_5 = isa.T_ISA(h1_5)
    V1_5 = M_5 * isa.compute_cs(T1_5)
    alpha_5 = cruise_alpha_model(h1_5, V1_5)
    rho1_5 = isa.rho_ISA(h1_5)
    q1_5 = aero.compute_q(V1_5, rho1_5)
    TW_cruise_5 = cst.constant_speed_climb(
        WS_range, alpha_5, beta_4, K1, K2, CD0, dhdt_5, V1_5, q1_5
    )
    plt.fill_between(WS_range, TW_cruise_5, 0.0, label="phase 5", alpha=0.3)
    WF_5 = Wref * phase_5(x)
    beta_5 = comp_beta(WF_5, WS, S)
    T_6 = isa.T_ISA(h_6)
    V_6 = M_6 * isa.compute_cs(T_6)
    alpha_6 = cruise_alpha_model(h_6, V_6)
    rho_6 = isa.rho_ISA(h_6)
    q_6 = aero.compute_q(V_6, rho_6)
    TW_cruise_6 = cst.constant_altitude_speed_cruise(
        WS_range, alpha_6, beta_5, K1, K2, CD0, q_6
    )
    plt.fill_between(WS_range, TW_cruise_6, 0.0, label="phase 6", alpha=0.3)
    WF_6 = Wref * phase_6(x)
    beta_6 = comp_beta(WF_6, WS, S)
    T_8 = isa.T_ISA(h_8)
    V_8 = M_8 * isa.compute_cs(T_8)
    alpha_8 = cruise_alpha_model(h_8, V_8)
    rho_8 = isa.rho_ISA(h_8)
    q_8 = aero.compute_q(V_8, rho_8)
    TW_cruise_8 = cst.constant_altitude_speed_turn(
        WS_range, alpha_8, beta_6, K1, K2, CD0, q_8, n_8
    )
    plt.fill_between(WS_range, TW_cruise_8, 0.0, label="phase 8", alpha=0.3)
    WF_8 = Wref * phase_8(x)
    beta_8 = comp_beta(WF_8, WS, S)
    V_9 = Vapp_9 * k_9
    WSmax = cst.non_stall_landing(beta_8, CLmax, V_9, rho_9)
    plt.axvspan(WSmax, WS_range[-1], label="phase 9a", alpha=0.3)
    CD_9 = aero.drag_polar(CLmax, K1, K2, CD0) * drag_penalty_9
    xi_9 = rw.compute_xi(CD_9, CDR, mu_9, CL_9)
    TW_cruise_9 = cst.braking_roll(
        WS_range, alpha_9, beta_8, xi_9, mu_9, CLmax, s_9, k_9, rho_9, g0
    )
    plt.fill_between(WS_range, TW_cruise_9, 0.0, label="phase 9b", alpha=0.3)
    plt.legend()
    plt.grid()
    plt.ylabel("$T_{SL}/W_{TO}$ [-]")
    plt.xlabel("$W_{TO}/S$ [Pa]")
    plt.title("Constraints plot")
    plt.ylim(-0.1, None)
    plt.legend(loc="upper left", bbox_to_anchor=(1.02, 1))
    plt.tight_layout()
    st.pyplot(fig)

    st.subheader("Weight fraction evolution")
    fig = plt.figure(2)
    betas = [
        beta_1,
        beta_2,
        beta_3,
        beta_4,
        beta_5,
        beta_6,
        beta_6,
        beta_8,
        beta_8,
    ]
    plt.bar([f"phase {i}" for i in range(1, 9 + 1)], betas)
    plt.ylabel(r"$\beta$")
    plt.grid(axis="y")
    plt.title("Total weight fraction evolution during design mission")
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

    st.subheader("Initial weight breakdown")
    fig = plt.figure(3)
    WTO = WS * S
    WF0_total = comp_WF0(WS, S) / (1 - min_fuel)
    WE = WTO * weights.wE_cargo_table(WTO)
    plt.pie(
        [WE, WP, WF0_total],
        labels=["Empty weight", "Total payload weight", "Initial fuel weight"],
        autopct="%1.1f%%",
    )
    plt.title(f"Initial weight breakdown (total {(WTO / g0 / 1000):.2f} t)")
    plt.tight_layout()
    st.pyplot(fig)


with right_col:
    st.header("Results")
    if run_optim:
        main()
