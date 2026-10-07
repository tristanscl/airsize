import numpy as np
import numdifftools as nd
from typing import Callable
from tqdm import tqdm
from stqdm import stqdm

Constraint = Callable[[np.ndarray], float]

DEFAULT_PENALTY_FUN = lambda x: np.maximum(x, 0.0)


def minimum_fuel_design(
    TW0,
    WS0,
    S0,
    WP,
    empty_weight_fun: Callable,
    mission: list[Constraint],
    constraints: list[Constraint],
    min_fuel: float = 0.0,
    TWref: np.ndarray = None,
    WSref: float = None,
    Sref: float = None,
    n_epochs: int = 100,
    lr: float = 1e-3,
    fd_step: float = None,
    constraint_penalty_fun: Callable = DEFAULT_PENALTY_FUN,
    mission_penalty_fun: Callable = DEFAULT_PENALTY_FUN,
    sanity_penalty_fun: Callable = DEFAULT_PENALTY_FUN,
    return_path: bool = False,
    verbose: bool = False,
    print_path: bool = False,
    normalize_print: bool = True,
    _gui: bool = False,
) -> (
    tuple[np.ndarray, np.ndarray, np.ndarray]
    | list[tuple[np.ndarray, np.ndarray, np.ndarray]]
):
    def loss(x: np.ndarray):
        TW = x[:-2] * TWref
        WS = x[-2] * WSref
        S = x[-1] * Sref
        WTO = WS * S
        WE = WTO * empty_weight_fun(WTO)
        WF0_total = WTO - WE - WP
        WF0 = WF0_total * (1 - min_fuel)
        Wref = WSref * Sref
        objective = WF0 / Wref
        mission_penalty = 0.0
        for phase in mission:
            mission_penalty += mission_penalty_fun(phase(x))
        constraints_penalty = 0.0
        for cst in constraints:
            constraints_penalty += constraint_penalty_fun(cst(x))
        sanity_penalty = 0.0
        for i in range(len(TW) + 2):
            sanity_penalty += sanity_penalty_fun(x[i])
        loss_value = objective + mission_penalty + constraints_penalty + sanity_penalty
        return loss_value

    grad_loss = nd.Gradient(loss, step=fd_step)
    if TWref is None:
        TWref = np.ones(len(TW0))
    xref = np.concatenate([TWref, np.array([WSref, Sref])])
    x = np.array(TW0 + [WS0, S0]) / xref
    if return_path:
        x_path = [x * xref]
    if print_path:
        print(f"Epoch 0/{n_epochs}:", x)
    load_bar = stqdm if _gui else tqdm
    for i in load_bar(range(n_epochs), disable=not verbose, desc="Minimizing fuel"):
        x -= lr * grad_loss(x)
        if print_path:
            print(f"Epoch {i+1}/{n_epochs}:", x if normalize_print else x * xref)
        if return_path:
            x_path.append(x * xref)
    x *= xref
    if return_path:
        return [(xi[:-2], xi[-2], xi[-1]) for xi in x_path]
    else:
        return x[:-2], x[-2], x[-1]
