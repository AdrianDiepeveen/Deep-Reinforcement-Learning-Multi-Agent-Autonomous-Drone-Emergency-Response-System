"""
Q-learning Model
Comparative Pipeline
Implemented Without High Level Libraries
"""
from __future__ import annotations
import os
import pickle
import math
from collections import defaultdict
import numpy as np

"""
Q-table Class
"""
class QTable:
    # QTable initialiser
    # Initialise Q-table parameters and storage
    def __init__(self,
                 alpha: float = 0.5,
                 gamma: float = 0.90,
                 n_actions: int = 3,
                 optimism: float = 1.0,
                 beta: float = 0.5) -> None:
        
        # Set base learning rate (alpha)
        self.base_alpha = alpha

        # Set base discount factor (gamma)
        self.gamma = gamma

        # Store number of possible actions
        self.n_act = n_actions

        # Set initial optimistic Q-value
        self.opt_init = optimism

        # Set exploration bonus coefficient
        self.beta = beta

        # Create Q-table
        self.table: dict[tuple, np.ndarray] = defaultdict(lambda: np.full(self.n_act, self.opt_init, dtype=np.float32))
        
        # Create count of state-action visits
        self.counts: dict[tuple, int] = defaultdict(int)

    # Get Q-values for a specific state
    def _get(self, key: tuple) -> np.ndarray:
        return self.table[key]       

    # Get Q-values for action selection
    def predict(self, key: tuple) -> np.ndarray:
        """Vector of Q-values for ε-greedy / softmax action selection."""
        return self._get(key)

    # Update Q-table entry using Temporal Difference with exploration bonus
    def update(self,
               s_key: tuple,
               a_idx: int,
               reward: float,
               nxt_key: tuple,
               done: bool) -> None:
        
        # Key combining state and chosen action
        sa_key = (s_key, a_idx)
        self.counts[sa_key] += 1
        visits = self.counts[sa_key]

        # Retrieve current qvalues for state
        qs = self._get(s_key)
        q_sa = qs[a_idx]

        # Compute maximum Q-value for next state
        q_next = 0.0 if done else np.max(self._get(nxt_key))

        # Compute exploration bonus scaling with visit count
        bonus = self.beta / math.sqrt(visits)         

        # BELLMAN EQUATION
        # Bellman Equation backup (Temporal Difference target)
        target = reward + bonus + self.gamma * q_next

        # Decay learning rate as visits increase
        alpha = self.base_alpha / (1.0 + 0.01 * visits)

        # TEMPORAL DIFFERENCE
        # Tabular update towards Temporal Difference target
        qs[a_idx] += alpha * (target - q_sa)

    # Save Q-table and visit counts
    def save(self, fname: str = "qtable.pkl") -> None:
        os.makedirs("model", exist_ok=True)
        with open(os.path.join("model", fname), "wb") as f:
            pickle.dump({"table":  dict(self.table), "counts": dict(self.counts)}, f)

    # Load Q-table and visit counts
    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            payload = pickle.load(f)
            self.table = defaultdict(lambda: np.full(self.n_act, self.opt_init, dtype=np.float32), payload.get("table", {}))
            self.counts = defaultdict(int, payload.get("counts", {}))