import numpy as np

from airsize.units import lb2kg

# Tables


def wE_cargo_table(WTO):
    return np.clip(0.55 - 0.05 / lb2kg(800 * 1000) * WTO, a_min=0.1, a_max=0.9)


def wE_fighter_table(WTO):
    return np.clip(0.65 - 0.15 / lb2kg(100 * 1000) * WTO, a_min=0.1, a_max=0.9)


def compute_WP(n_passengers: int, long_flight: bool):
    return lb2kg(n_passengers * (175 + long_flight * 40 + (not long_flight) * 30))
