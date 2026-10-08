# 3rd party
from typing import Callable
import numpy as np
import json
import matplotlib.pyplot as plt
import streamlit as st
from stqdm import stqdm

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

st.title("Benchmark - F86 redesign")

left_col, right_col = st.columns(2)

with left_col:
    st.header("Design parameters")

    st.subheader("Aircraft")

    # Global: requirements
    with st.expander("General requirements"):
        WP = units.lbf2N(
            st.number_input(
                "Total payload weight (crew and gear) in lb", value=(210 + 432) * 1.1
            )
        )  # N, ASSUMPTION (10% margin)
        min_fuel = (
            st.number_input("Min. fuel reserve in %", value=15) / 100
        )  # ASSUMPTION (5% margin)
        sTO_max = units.ft2m(
            st.number_input("Max. takeoff distance in ft", value=4400)
        )  # m
        sL_max = units.ft2m(
            st.number_input("Max. landing distance in ft", value=5000)
        )  # m

        # Global: aerodynamics
    with st.expander("Aerodynamics"):

        CLmax = st.number_input("$C_{L}^{max}$", value=1.3)
        AR = st.number_input("Aspect ratio $AR$", value=4.883)
        e = st.number_input("Oswald efficiency factor $e$", value=0.75)
        CD0 = st.number_input("$1000 C_{D}^{0}$", value=20.0) / 1000
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
            "$C_{D}^{R}$ (additional drag from landing gear)", value=0.0
        )  # ASSUMPTION (drag from landing gear is negligible)

    # Global: physics
    g0 = isa.g_ISA  # m.s-2

    # Global: propulsion

    with st.expander("Propulsion"):
        cruise_lapse_decay = (
            st.number_input("Cruise lapse decay in $10^{-5}$ m$^{-1}$", value=5.657)
            * 1e-5
        )
        mil_lapse_decay = (
            st.number_input("Military lapse decay in $10^{-5}$ m$^{-1}$", value=5.500)
            * 1e-5
        )
        wet_lapse_decay = (
            st.number_input("Wet lapse decay in $10^{-5}$ m$^{-1}$", value=4.805) * 1e-5
        )
        cruise_alpha_model = lambda x: np.maximum(0.0, 1.0 - cruise_lapse_decay * x)
        mil_alpha_model = lambda x: np.maximum(0.0, 1.0 - mil_lapse_decay * x)
        wet_alpha_model = lambda x: np.maximum(0.0, 1.0 - wet_lapse_decay * x)
        fig = plt.figure(0)
        h_range_alpha_plot = np.linspace(0, 16000, 100)
        plt.plot(
            h_range_alpha_plot / 1000,
            cruise_alpha_model(h_range_alpha_plot),
            label="Cruise",
        )
        plt.plot(
            h_range_alpha_plot / 1000,
            mil_alpha_model(h_range_alpha_plot),
            label="Military",
        )
        plt.plot(
            h_range_alpha_plot / 1000, wet_alpha_model(h_range_alpha_plot), label="Wet"
        )
        plt.grid()
        plt.xlabel("Altitude above sea level [km]")
        plt.ylabel(r"$\alpha$")
        plt.legend()
        plt.title("Lapse rate decay model")
        plt.ylim(-0.1, 1.1)
        st.pyplot(fig)
        cruise_avg_tsfc = (
            st.number_input("Cruise average TSFC in $10^{-4}$ s$^{-1}$", value=4.23)
            * 1e-4
        )  # s-1
        mil_avg_tsfc = (
            st.number_input("Military average TSFC in $10^{-4}$ s$^{-1}$", value=3.98)
            * 1e-4
        )  # s-1
        wet_avg_tsfc = (
            st.number_input("Wet average TSFC in $10^{-4}$ s$^{-1}$", value=7.83) * 1e-4
        )  # s-1
        TSFC_1a = (
            st.number_input(
                "Warm-up fuel consumption in $10^{-4}$ s$^{-1}$", value=3.14
            )
            * 1e-4
        )  # s-1

    ## Phases ##

    st.subheader("Mission")

    with st.expander("Phase 1: Takeoff"):
        """
        Set parameters:
        * Propulsion: wet
        """

        # Phase 1a: warm up
        alpha_1a = 1.0
        beta_1a = 1.0
        t_1a = st.number_input("Warm up time in s", value=80.0)  # s

        # Phase 1b: takeoff ground roll
        rho_1b = isa.rho0_ISA  # kg.m-3
        k_1b = st.number_input(
            r"$k = V_{TO}/V_{stall}$", value=1.1, key="k_1b"
        )  # ASSUMPTION (10% margin over Vstall for takeoff)
        alpha_1b = alpha_1a
        TSFC_1b = wet_avg_tsfc  # s-1
        HL_mult_1 = st.number_input(
            "High lift $C_L^{max}$ multiplier", value=1.3, key="HL_mult_1"
        )
        CL_1 = (
            CLmax * HL_mult_1
        )  # ASSUMPTION (high lift devices, increasing lift by 30%)
        drag_penalty_1 = st.number_input(
            "High lift drag multiplier", value=1.35, key="drag_penalty_1"
        )  # ASSUMPTION (drag is increased by 35% using high lift devices)
        mu_1b = st.number_input(
            r"$\mu$ (Coulomb drag coefficient)", value=0.02, key="mu_1b"
        )  # ASSUMPTION (from lecture slides)
        s_1 = (
            st.number_input("$s_{GR}/s_{TO}^{max}$", value=0.9, key="s_1") * sTO_max
        )  # ASSUMPTION (10% margin)

        # Phase 1c: takeoff rotation
        rho_1c = isa.rho0_ISA  # kg.m-3
        alpha_1c = alpha_1b
        TSFC_1c = TSFC_1b  # s-1
        tR_1c = st.number_input(
            "$t_R$ in s", value=3.0
        )  # s, ASSUMPTION (from lecture slides)
        hOBS_1c = units.ft2m(st.number_input(r"$h_{obs}$ in ft", value=50))  # m

    # Phase 2: horizontal acceleration
    with st.expander("Phase 2: Acceleration at sea level"):
        """
        Set parameters:
        * Propulsion: wet
        """
        h2_2 = 0.0  # m
        rho_2 = isa.rho0_ISA  # kg.m-3
        CL_2 = CLmax * HL_mult_1
        V1_2 = units.ft2m(
            st.number_input("Target speed in ft/s", value=1020 * 1.1, key="V1_2")
        )  # m/s, ASSUMPTION (10% more than target speed)
        TSFC_2 = wet_avg_tsfc  # s-1
        alpha_2 = 1.0
        dVdt_2 = (
            st.number_input(r"Acceleration in $g$", value=0.2) * g0
        )  # m.s-2 ASSUMPTION (0.2g acceleration)

    # Phase 3: initial climb
    with st.expander("Phase 3: Initial climb"):
        """
        Set parameters:
        * Propulsion: military
        * Speed: same than phase 2
        * Lift: $C_L^{max}$ without high lift devices
        """
        dhdt_3 = units.ft2m(st.number_input("Climb rate in ft/s", value=90 * 0.8))
        # m/s, ASSUMTION (20% margin wrt requirement)
        h0_3 = 0.0  # m/s
        h1_3 = units.ft2m(st.number_input("$h_{max}$ in ft", value=35400))  # m
        n_split_3 = st.number_input(
            "$n_{split}$ (number of subdivisions of the climb for more accurate integration)",
            value=10,
            key="n_split_3",
        )  # ASSUMPTION (alpha relatively constant on 3540 ft intervals)
        CL_3 = CLmax  # ASSUMPTION (climbing with CLmax and no high lift devices)
        V_3 = V1_2  # ASSUMPTION (continuity)
        TSFC_3 = mil_avg_tsfc  # m.s-1

    # Phase 4: cruise-climb
    with st.expander("Phase 4: Cruise climb"):
        """
        Set parameters:
        * Propulsion: cruise
        * Aerodynamics: max. $L/D$
        """
        h0_4 = h1_3  # m
        h1_4 = units.ft2m(st.number_input("$h_{max}$ in ft", value=38700))  # m
        CD_4 = CD_max_LD  # ASSUMPTION (economy)
        CL_4 = CL_max_LD  # ASSUMPTION (economy)
        V_4 = units.kt2ms(st.number_input("$V$ in kt", value=458))  # m.s-1
        TSFC_4 = cruise_avg_tsfc  # s-1
        s_4 = (
            units.nm2km(st.number_input("$s$ in nmi", value=550, key="s_4")) * 1000
        )  # m

    # Phase 5: loiter
    with st.expander("Phase 5: Loiter"):
        """
        Set parameters:
        * Propulsion: cruise
        * Altitude: same than end of phase 4
        * Speed: cruise speed (same than phase 4)
        """
        TSFC_5 = cruise_avg_tsfc  # s-1
        t_5 = st.number_input(f"$t$ in min", value=10, key="t_5") * 60  # s
        h_5 = h1_4  # m
        V_5 = V_4  # ASSUMPTION (speed continuity)

    # Phase 6: climb
    with st.expander("Phase 6: Climb to combat"):
        """
        Set parameters:
        * Propulsion: cruise
        * Speed: cruise speed (same than phase 5)
        * Lift: $C_L^{max}$ without high lift devices
        """
        h0_6 = h_5  # m
        h1_6 = units.ft2m(st.number_input("$h_{max}$ in ft", value=47550))  # m
        n_split_6 = st.number_input(
            "$n_{split}$ (number of subdivisions of the climb for more accurate integration)",
            value=3,
        )  # ASSUMPTION (alpha relatively constant on 2950 ft intervals)
        CL_6 = CLmax  # ASSUMPTION (climbing with CLmax and no high lift devices)
        V_6 = V_4  # ASSUMPTION (climb at cruise speed)
        TSFC_6 = wet_avg_tsfc  # ASSUMPTION (climb using afterburners)
        dhdt_6 = dhdt_3  # ASSUMPTION (climb rate is the same than initial climb)

    # Phase 7: combat
    with st.expander("Phase 7: Combat"):
        """
        The combat phase was designed as a series of turn of radius $R_C$ for a duration $t$ at speed $V$ under maximum (wet) power.

        Set parameters:
        * Propulsion: wet
        * Lift: $C_L^{max}$ without high lift devices (to enable shorter turn radii)
        * Altitude: same than end of phase 6
        """
        V_7 = units.kt2ms(
            st.number_input("$V$ in kt", value=500)
        )  # m.s-1 (from unclassified data)
        RC_7 = units.ft2m(
            st.number_input("$R_C$ in ft", 17000)
        )  # m, ASSUMPTION (worst case 45k ft unclassified data)
        t_7 = st.number_input("$t$ in min", value=5, key="t_7") * 60  # s
        CL_7 = CLmax  # ASSUMPTION (lift necessary to shorten turn radius without requiring immmense S)
        TSFC_7 = wet_avg_tsfc  # s-1
        h_7 = h1_6  # m

    # Phase 8: cruise
    with st.expander("Phase 8: Cruise"):
        """
        Set parameters:
        * Propulsion: cruise
        * Aerodynamics: max. $L/D$
        """
        h_8 = units.ft2m(st.number_input("$h$ in ft", value=37000, key="h_8"))  # m
        V_8 = units.kt2ms(st.number_input("$V$ in kt", value=536))  # m.s-1
        CD_8 = CD_max_LD  # ASSUMPTION (economy)
        CL_8 = CL_max_LD  # ASSUMPTION (economy)
        TSFC_8 = cruise_avg_tsfc  # s-1
        s_8 = (
            units.nm2km(st.number_input("$s$ in nmi", value=550, key="s_8")) * 1000
        )  # m

    # Phase 9: loiter
    with st.expander("Phase 9: Loiter before landing"):
        """
        Set parameters:
        * Propulsion: cruise
        * Speed: same than phase 8
        """
        h_9 = units.ft2m(st.number_input("$h$ in ft", value=35000, key="h_9"))  # m
        t_9 = st.number_input("$t$ in min", value=10) * 60  # s
        TSFC_9 = cruise_avg_tsfc  # s-1
        V_9 = V_8  # m.s-1  # ASSUMPTION (maximum endurance speed close to 458 kt)

    # Phase 10: landing
    with st.expander("Phase 10: Landing at sea level"):
        """
        Set parameters:
        * Propulsion: cruise
        * Speed: same than phase 8
        * Aerodynamics: $C_L^{max}$ without high lift devices (in case of damage during combat)
        """
        h_10 = 0.0
        alpha_10 = st.number_input(
            r"$\alpha$ (reversed)", value=0.65, key="alpha_10"
        )  # ASSUMPTION (fraction of thrust that can be reversed)
        CL_10 = CLmax
        mu_10 = st.number_input(
            r"$\mu$", value=10 * 0.05, key="mu_10"
        )  # ASSUMPTION (brakes have the same effect than 100 times the Coulomb drag coefficient during takeoff)
        k_10 = st.number_input(
            "$k = V_{TO} / V_{stall}$", value=1.1, key="k_10"
        )  # ASSUMPTION (10% margin wrt Vstall)
        rho_10 = isa.rho0_ISA
        s_10 = (
            st.number_input("$s_B / s_L^{max}$", value=0.8, key="s_10") * sL_max
        )  # ASSUMPTION (20% margin)

    ## Optimization settings ##

    st.subheader("Optimization")

    with st.expander("Hyperparameters"):
        lr_val = st.number_input("Learning rate value", value=5)
        lr_mag = st.number_input("Learning rate magnitude (in power of 10)", value=-2)
        lr = lr_val * 10**lr_mag
        n_epochs = st.number_input("Epochs", value=200)

    mission_penalty_fun = lambda x: 100 * np.maximum(-x + 0.001, 0.0) ** 2
    constraints_penalty_fun = lambda x: 3 * np.maximum(-x + 0.05, 0.0) ** 2
    sanity_penalty_fun = lambda x: 0.0  # no sanity penalty (for positivity)

    with st.expander("Initialization"):
        Wref = units.lbf2N(st.number_input("$W_{ref}$ in lbf", value=16252))  # N
        Tref_wet = units.lbf2N(
            st.number_input("$T_{ref}^{wet}$ in lbf", value=7650)
        )  # N
        Tref_mil = units.lbf2N(
            st.number_input("$T_{ref}^{mil}$ in lbf", value=5550)
        )  # N
        Tref_cruise = units.lbf2N(
            st.number_input("$T_{ref}^{cruise}$ in lbf", value=5100)
        )  # N
        Sref = units.ft2m(
            units.ft2m(st.number_input("$S_{ref}$ in ft$^2$", value=313.37))
        )  # m2

    TWref_cruise = Tref_cruise / Wref
    TWref_mil = Tref_mil / Wref
    TWref_wet = Tref_wet / Wref
    TWref = np.array([TWref_cruise, TWref_mil, TWref_wet])
    WSref = Wref / Sref  # Pa
    TW0 = (np.array([1, 1, 1]) * TWref).tolist()
    WS0 = WSref
    S0 = Sref

    run_optim = st.button(label="Run", width="stretch")

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
    beta_current = beta_before
    CD_3 = aero.drag_polar(CL_3, K1, K2, CD0)
    for i in range(n_split_3):
        h0 = h_range[i]
        h1 = h_range[i + 1]
        alpha = mil_alpha_model(h1)
        beta_current *= mission.constant_speed_climb(
            TW_mil, h0, h1, alpha, beta_current, CD_3, CL_3, V_3, TSFC_3
        )
    beta_instant = beta_current / beta_before
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
    beta_current = beta_before
    CD_6 = aero.drag_polar(CL_6, K1, K2, CD0)
    for i in range(n_split_6):
        h0 = h_range[i]
        h1 = h_range[i + 1]
        alpha = wet_alpha_model(h1)
        beta_current *= mission.constant_speed_climb(
            TW_wet, h0, h1, alpha, beta_current, CD_6, CL_6, V_6, TSFC_6
        )
    beta_instant = beta_current / beta_before
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
    alpha = cruise_alpha_model(h1_4)
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
    alpha = wet_alpha_model(h1_6)
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


def main():
    ## Optimization ##

    if __name__ == "__main__":
        # Execution
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
            verbose=True,
            print_path=True,
            normalize_print=True,
            _gui=True,
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

        # Plots
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

        st.subheader("Constraint plots")

        fig = plt.figure(1)
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
        plt.fill_between(WS_range, TW_cruise_9, 0.0, label="phase 9", alpha=0.3)
        plt.legend()
        plt.grid()
        plt.ylabel("$T_{SL}/W_{TO}$ [-]")
        plt.xlabel("$W_{TO}/S$ [Pa]")
        plt.title("Constraints plot - Cruise")
        plt.ylim(-0.1, None)
        plt.legend(loc="upper left", bbox_to_anchor=(1.02, 1))
        plt.tight_layout()
        st.pyplot(fig)

        fig = plt.figure(2)
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
        st.pyplot(fig)

        fig = plt.figure(3)
        plt.plot(WS_history, TW_wet_history, label="optim. path", color="gray")
        plt.scatter([WSref], [TWref[2]], label="ref.", color="blue")
        plt.scatter([WS], [TW_wet], label="end", color="red")
        CD_1 = aero.drag_polar(CL_1, K1, K2, CD0) * drag_penalty_1
        xi_1b = rw.compute_xi(CD_1, CDR, mu_1b, CL_1)
        TW_wet_1b = cst.takeoff_ground_roll_low_thrust(
            WS_range, alpha_1b, beta_1a, xi_1b, mu_1b, CL_1, s_1, k_1b, rho_1b, g0
        )
        plt.fill_between(WS_range, TW_wet_1b, 0.0, label="phase 1", alpha=0.3)
        WF_1 = Wref * phase_1c(x)
        beta_1 = comp_beta(WF_1, WS, S)
        V_1c = k_1b * aero.compute_Vstall(WS, beta_1, CL_2, rho_1c)
        q_2 = aero.compute_q(V_1c, rho_2)
        TW_wet_2 = (
            cst.horizontal_acceleration(
                WS_range, alpha_2, beta_1, K1, K2, CD0, dVdt_2, q_2
            )
            * drag_penalty_1
        )
        plt.fill_between(WS_range, TW_wet_2, 0.0, label="phase 2", alpha=0.3)
        WF_5 = Wref * phase_5(x)
        beta_5 = comp_beta(WF_5, WS, S)
        alpha_6 = wet_alpha_model(h1_6)
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
            WS_range, alpha_10, beta_9, xi_10, mu_10, CLmax, s_10, k_10, rho_10, g0
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
        st.pyplot(fig)

        st.subheader("Weight fraction evolution")

        fig = plt.figure(4)
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
        st.pyplot(fig)

        st.subheader("initial weight breakdown")

        fig = plt.figure(5)
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
        st.pyplot(fig)

        st.subheader("Convereged design point")
        f"""
* $T_{{SL}} / W_{{TO}}$ (cruise): {TW_cruise:.3f}
* $T_{{SL}} / W_{{TO}}$ (military): {TW_mil:.3f}
* $T_{{SL}} / W_{{TO}}$ (wet): {TW_wet:.3f}
* $W_{{TO}} / S$: {WS:.0f} Pa
* $S$: {S:.1f} m$^2$
"""


with right_col:
    st.header("Results")
    if run_optim:
        main()
