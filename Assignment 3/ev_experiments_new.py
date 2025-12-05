"""
Experiment utilities – ALL RATIO LOGIC REMOVED.
Sweeps now use I0 instead of ratio.
"""

from __future__ import annotations
import os
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd

from ev_core_new import (
    EVStagHuntModel,
    set_initial_adopters,
    final_mean_adoption_vs_I0,
    phase_sweep_X0_vs_I0,
)

####################################
# Timeseries runner (unchanged conceptually)
####################################

def run_timeseries_trial(
    T: int = 200,
    scenario_kwargs: Optional[Dict] = None,
    seed: Optional[int] = None,
    policy=None,
    strategy_choice_func="imitate",
    tau=1.0,
):

    scenario = {
        "a0": 2.0,
        "beta_I": 3.0,
        "b": 1.0,
        "g_I": 0.1,
        "I0": 0.05,
        "network_type": "random",
        "n_nodes": 100,
        "p": 0.05,
        "m": 2,
        "collect": True,
        "X0_frac": 0.0,
        "init_method": "random",
    }
    if scenario_kwargs:
        scenario.update(scenario_kwargs)

    model = EVStagHuntModel(
        a0=scenario["a0"],
        beta_I=scenario["beta_I"],
        b=scenario["b"],
        g_I=scenario["g_I"],
        I0=scenario["I0"],
        seed=seed,
        network_type=scenario["network_type"],
        n_nodes=scenario["n_nodes"],
        p=scenario["p"],
        m=scenario["m"],
        collect=True,
        strategy_choice_func=strategy_choice_func,
        tau=tau,
    )

    if scenario.get("X0_frac", 0.0) > 0.0:
        set_initial_adopters(
            model,
            scenario["X0_frac"],
            method=scenario.get("init_method", "random"),
            seed=seed,
        )

    for t in range(T):
        if policy is not None:
            policy(model, t)
        model.step()

    df = model.datacollector.get_model_vars_dataframe().copy()
    return df["X"].to_numpy(), df["I"].to_numpy(), df


####################################
# NEW — sweep X* vs I0
####################################

def I0_sweep_df(
    X0_frac: float = 0.40,
    I0_values: Optional[np.ndarray] = None,
    scenario_kwargs: Optional[Dict] = None,
    T: int = 250,
    batch_size: int = 16,
    strategy_choice_func="logit",
    tau=1.0,
) -> pd.DataFrame:

    if I0_values is None:
        I0_values = np.linspace(0.0, 1.0, 41)

    scenario = {
        "a0": 2.0,
        "beta_I": 2.0,
        "b": 1.0,
        "g_I": 0.05,
        "network_type": "BA",
        "n_nodes": 120,
        "p": 0.05,
        "m": 2,
    }
    if scenario_kwargs:
        scenario.update(scenario_kwargs)

    X_means = final_mean_adoption_vs_I0(
        X0_frac,
        I0_values,
        a0=scenario["a0"],
        beta_I=scenario["beta_I"],
        b=scenario["b"],
        g_I=scenario["g_I"],
        T=T,
        network_type=scenario["network_type"],
        n_nodes=scenario["n_nodes"],
        p=scenario["p"],
        m=scenario["m"],
        batch_size=batch_size,
        strategy_choice_func=strategy_choice_func,
        tau=tau,
    )

    return pd.DataFrame({"I0": I0_values, "X_mean": X_means})


####################################
# NEW — 2D phase sweep X0 vs I0
####################################

def phase_sweep_df_I0(
    X0_values: Optional[np.ndarray] = None,
    I0_values: Optional[np.ndarray] = None,
    scenario_kwargs: Optional[Dict] = None,
    batch_size: int = 16,
    T: int = 250,
    strategy_choice_func="logit",
    tau=1.0,
) -> pd.DataFrame:

    if X0_values is None:
        X0_values = np.linspace(0.0, 1.0, 21)
    if I0_values is None:
        I0_values = np.linspace(0.0, 1.0, 41)

    scenario = {
        "a0": 2.0,
        "beta_I": 2.0,
        "b": 1.0,
        "g_I": 0.05,
        "network_type": "BA",
        "n_nodes": 120,
        "p": 0.05,
        "m": 2,
    }
    if scenario_kwargs:
        scenario.update(scenario_kwargs)

    X_final = phase_sweep_X0_vs_I0(
        X0_values,
        I0_values,
        a0=scenario["a0"],
        beta_I=scenario["beta_I"],
        b=scenario["b"],
        g_I=scenario["g_I"],
        T=T,
        network_type=scenario["network_type"],
        n_nodes=scenario["n_nodes"],
        p=scenario["p"],
        m=scenario["m"],
        batch_size=batch_size,
        strategy_choice_func=strategy_choice_func,
        tau=tau,
    )

    rows = []
    for i, I0 in enumerate(I0_values):
        for j, X0 in enumerate(X0_values):
            rows.append((float(X0), float(I0), float(X_final[i, j])))

    return pd.DataFrame(rows, columns=["X0", "I0", "X_final"])