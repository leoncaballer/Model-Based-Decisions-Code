"""
Core EV Stag Hunt model components – revised for X0 vs I0 sweeps only.

NO RATIO-BASED CODE REMAINS.
"""

from mesa import Agent, Model
from mesa.time import SimultaneousActivation
from mesa.space import NetworkGrid
from mesa.datacollection import DataCollector
import networkx as nx
import math
import numpy as np
import os
import random
from typing import Iterable, List


####################################
# Strategy selection helpers
####################################

def choose_strategy_imitate(agent, neighbors):
    candidates = neighbors + [agent]
    best = max(candidates, key=lambda a: a.payoff)
    return best.strategy


def choose_strategy_logit(agent, neighbors, a_I, b, tau):
    pi_C = 0.0
    pi_D = 0.0
    for other in neighbors:
        s_j = other.strategy
        if s_j == "C":
            pi_C += a_I
            pi_D += b
        else:
            pi_C += 0.0
            pi_D += b

    denom = np.exp(pi_C / tau) + np.exp(pi_D / tau)
    P_C = np.exp(pi_C / tau) / denom if denom > 0 else 0.5
    return "C" if random.random() < P_C else "D"


####################################
# Agent
####################################

class EVAgent(Agent):
    def __init__(self, uid, model, init_strategy="D"):
        super().__init__(uid, model)
        self.strategy = init_strategy
        self.payoff = 0.0
        self.next_strategy = init_strategy

    def step(self):
        I = self.model.infrastructure
        a0 = self.model.a0
        beta_I = self.model.beta_I
        b = self.model.b
        a_I = a0 + beta_I * I

        neighbor_agents = []
        for nbr in self.model.G.neighbors(self.pos):
            neighbor_agents.extend(self.model.grid.get_cell_list_contents([nbr]))
        if not neighbor_agents:
            self.payoff = 0.0
            return

        payoff = 0.0
        for other in neighbor_agents:
            s_i = self.strategy
            s_j = other.strategy
            if s_i == "C" and s_j == "C":
                payoff += a_I
            elif s_i == "C" and s_j == "D":
                payoff += 0.0
            elif s_i == "D" and s_j == "C":
                payoff += b
            else:
                payoff += b
        self.payoff = payoff

    def advance(self, strategy_choice_func="imitate"):
        func = strategy_choice_func if strategy_choice_func else getattr(self.model, "strategy_choice_func", "imitate")

        neighbor_agents = []
        for nbr in self.model.G.neighbors(self.pos):
            neighbor_agents.extend(self.model.grid.get_cell_list_contents([nbr]))

        if func == "imitate":
            self.next_strategy = choose_strategy_imitate(self, neighbor_agents)
        elif func == "logit":
            a_I = self.model.a0 + self.model.beta_I * self.model.infrastructure
            self.next_strategy = choose_strategy_logit(self, neighbor_agents, a_I, self.model.b, getattr(self.model, "tau", 1.0))
        else:
            raise ValueError(f"Unknown strategy choice: {func}")

        self.strategy = self.next_strategy


####################################
# Model
####################################

class EVStagHuntModel(Model):
    def __init__(
        self,
        initial_ev=10,
        a0=2.0,
        beta_I=3.0,
        b=1.0,
        g_I=0.1,
        I0=0.05,
        seed=None,
        network_type="random",
        n_nodes=100,
        p=0.05,
        m=2,
        collect=True,
        strategy_choice_func="imitate",
        tau=1.0,
    ):
        super().__init__(seed=seed)

        # Build graph
        if network_type == "BA":
            # Scale-free Barabási–Albert
            G = nx.barabasi_albert_graph(n_nodes, m, seed=seed)

        elif network_type == "grid":
            # 2D L×L grid, L chosen to give ~n_nodes
            L = int(round(math.sqrt(n_nodes)))
            G = nx.grid_2d_graph(L, L)

            # Relabel (i, j) → 0..N-1 so the rest of the code still works
            mapping = {node: i for i, node in enumerate(G.nodes())}
            G = nx.relabel_nodes(G, mapping)

        elif network_type == "WS":
            # Watts–Strogatz small-world
            k = 4          # each node initially connected to k neighbours
            G = nx.watts_strogatz_graph(n_nodes, k, p, seed=seed)

        else:
            # Default: Erdős–Rényi random graph (“random” / “ER”)
            G = nx.erdos_renyi_graph(n_nodes, p=p, seed=seed)

        self.G = G
        self.grid = NetworkGrid(G)
        self.schedule = SimultaneousActivation(self)

        self.a0 = a0
        self.beta_I = beta_I
        self.b = b
        self.g_I = g_I
        self.infrastructure = I0
        self.strategy_choice_func = strategy_choice_func
        self.tau = tau

        total_nodes = self.G.number_of_nodes()
        k_ev = max(0, min(initial_ev, total_nodes))
        ev_nodes = set(self.random.sample(list(self.G.nodes), k_ev))

        uid = 0
        for node in self.G.nodes:
            init_strategy = "C" if node in ev_nodes else "D"
            agent = EVAgent(uid, self, init_strategy)
            uid += 1
            self.schedule.add(agent)
            self.grid.place_agent(agent, node)

        self.datacollector = None
        if collect:
            self.datacollector = DataCollector(
                model_reporters={"X": self.get_adoption_fraction,
                                 "I": lambda m: m.infrastructure},
                agent_reporters={"strategy": "strategy", "payoff": "payoff"},
            )

    def get_adoption_fraction(self):
        agents = self.schedule.agents
        if not agents:
            return 0.0
        return sum(1 for a in agents if a.strategy == "C") / len(agents)

    def step(self):
        self.schedule.step()
        X = self.get_adoption_fraction()
        I = self.infrastructure
        dI = self.g_I * (X - I)
        self.infrastructure = float(min(1.0, max(0.0, I + dI)))
        if self.datacollector is not None:
            self.datacollector.collect(self)


####################################
# Initial adopters
####################################

def set_initial_adopters(model, X0_frac, method="random", seed=None, high=True):
    rng = np.random.default_rng(seed)
    agents = model.schedule.agents
    n = len(agents)
    k = int(round(X0_frac * n))

    for a in agents:
        a.strategy = "D"

    if k <= 0:
        return

    if method == "random":
        idx = rng.choice(n, size=k, replace=False)
        for i in idx:
            agents[i].strategy = "C"
        return

    if method == "degree":
        deg = dict(model.G.degree())
        ordered = sorted(deg.keys(), key=lambda u: deg[u], reverse=high)
        chosen = set(ordered[:k])
        for a in agents:
            if a.unique_id in chosen:
                a.strategy = "C"
        return

    raise ValueError("Unknown init method")


####################################
# NEW — no ratio logic
# run_network_trial is parameterized by (X0, I0)
####################################

def run_network_trial(
    X0_frac: float,
    I0: float,
    *,
    a0: float = 2.0,
    beta_I: float = 2.0,
    b: float = 1.0,
    g_I: float = 0.05,
    T: int = 200,
    network_type="random",
    n_nodes=120,
    p=0.05,
    m=2,
    seed=None,
    collect=False,
    strategy_choice_func="imitate",
    tau=1.0,
) -> float:

    initial_ev = int(round(X0_frac * n_nodes))

    model = EVStagHuntModel(
        initial_ev=initial_ev,
        a0=a0,
        beta_I=beta_I,
        b=b,
        g_I=g_I,
        I0=I0,
        seed=seed,
        network_type=network_type,
        n_nodes=n_nodes,
        p=p,
        m=m,
        collect=collect,
        strategy_choice_func=strategy_choice_func,
        tau=tau,
    )

    prev_X = None
    prev_I = None
    stable = 0
    tol = 1e-3
    patience = 30

    for _ in range(T):
        model.step()
        X = model.get_adoption_fraction()
        I = model.infrastructure

        if prev_X is not None and prev_I is not None:
            if abs(X - prev_X) < tol and abs(I - prev_I) < tol:
                stable += 1
            else:
                stable = 0
        prev_X, prev_I = X, I

        if stable >= patience or X in (0.0, 1.0):
            break

    return model.get_adoption_fraction()


####################################
# NEW — mean final adoption vs I0
####################################

def final_mean_adoption_vs_I0(
    X0_frac: float,
    I0_values: Iterable[float],
    *,
    a0: float = 2.0,
    beta_I: float = 2.0,
    b: float = 1.0,
    g_I: float = 0.05,
    T: int = 200,
    network_type="random",
    n_nodes=120,
    p=0.05,
    m=2,
    batch_size=16,
    strategy_choice_func="imitate",
    tau=1.0,
) -> np.ndarray:

    means = []
    for I0 in I0_values:
        finals = []
        for _ in range(batch_size):
            seed_j = np.random.randint(0, 2**31 - 1)
            x_star = run_network_trial(
                X0_frac=X0_frac,
                I0=I0,
                a0=a0,
                beta_I=beta_I,
                b=b,
                g_I=g_I,
                T=T,
                network_type=network_type,
                n_nodes=n_nodes,
                p=p,
                m=m,
                seed=seed_j,
                collect=False,
                strategy_choice_func=strategy_choice_func,
                tau=tau,
            )
            finals.append(x_star)
        means.append(float(np.mean(finals)))

    return np.asarray(means, dtype=float)


####################################
# NEW — 2D phase sweep over (X0, I0)
####################################

def phase_sweep_X0_vs_I0(
    X0_values: Iterable[float],
    I0_values: Iterable[float],
    *,
    a0: float = 2.0,
    beta_I: float = 2.0,
    b: float = 1.0,
    g_I: float = 0.05,
    T: int = 250,
    network_type="BA",
    n_nodes=120,
    p=0.05,
    m=2,
    batch_size=16,
    strategy_choice_func="logit",
    tau=1.0,
) -> np.ndarray:

    X0_values = list(X0_values)
    I0_values = list(I0_values)

    X_final = np.zeros((len(I0_values), len(X0_values)), dtype=float)

    for i, I0 in enumerate(I0_values):
        for j, X0 in enumerate(X0_values):
            finals = []
            for _ in range(batch_size):
                seed_j = np.random.randint(0, 2**31 - 1)
                x_star = run_network_trial(
                    X0_frac=X0,
                    I0=I0,
                    a0=a0,
                    beta_I=beta_I,
                    b=b,
                    g_I=g_I,
                    T=T,
                    network_type=network_type,
                    n_nodes=n_nodes,
                    p=p,
                    m=m,
                    seed=seed_j,
                    collect=False,
                    strategy_choice_func=strategy_choice_func,
                    tau=tau,
                )
                finals.append(x_star)

            X_final[i, j] = float(np.mean(finals))

    return X_final