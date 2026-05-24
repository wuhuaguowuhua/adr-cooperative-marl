import math
import torch
from typing import List, Optional, Dict
from collections import defaultdict


class CountBasedExplorer:
    """
    Count-based intrinsic motivation for multi-agent exploration.

    Each agent maintains its own visitation count table using SimHash
    (random projection to low-dim binary space). Intrinsic reward for
    agent i visiting state s:

        bonus_i(s) = η / sqrt(N_i(s))

    Parameter derivation:

    1) Exploration coefficient η (coef):
       η = β · R̂ / T
       where R̂ = estimated achievable episodic return,
             T  = episode time horizon,
             β  = exploration-exploitation ratio (hyperparameter).
       Example (RWARE-small-4ag): β=1.25, R̂=20, T=500 → η=0.05

    2) Hash dimensionality d_h (hash_dim):
       d_h = ⌈log₂(S_eff · γ)⌉
       where S_eff = N_cells × k  (effective reachable state count),
             γ     = collision factor controlling bucket fill ratio.
       Example (RWARE-small-4ag): 144×32≈4608, γ=0.9 → d_h=12

    3) Count decay (λ, T_d):
       Periodic count decay prevents bonus from vanishing over long training.
       Steady-state count: c_ss = λ · T_d · n_proc / ((1-λ) · 2^d_h)
       Steady-state bonus: η_ss = η / √c_ss
       Target: η_ss = δ · R̂ / T  (δ = sustained exploration ratio)
       Example: λ=0.5, T_d=50000, n_proc=4, d_h=12
                c_ss = 0.5×200000/(0.5×4096) ≈ 49 → η_ss ≈ 0.007
    """

    def __init__(
        self,
        n_agents: int,
        coef: float = 0.01,
        hash_dim: int = 16,
        decay_factor: float = 1.0,
        decay_interval: int = 50000,
        warmup_ratio: float = 0.0,
        anneal_end_ratio: float = 1.0,
    ):
        self.n_agents = n_agents
        self.coef = coef
        self.hash_dim = hash_dim
        self.decay_factor = decay_factor
        self.decay_interval = decay_interval
        self.warmup_ratio = warmup_ratio
        self.anneal_end_ratio = anneal_end_ratio

        self._projection: Optional[torch.Tensor] = None
        self._counts: List[Dict[int, float]] = [
            defaultdict(float) for _ in range(n_agents)
        ]
        self._step = 0

        self._last_bonus_mean = 0.0
        self._last_avg_count = 0.0
        self._last_anneal_mult = 0.0
        self._unique_states = [0] * n_agents

        print(
            f"[CountBasedExplorer] n_agents={n_agents}, coef={coef}, "
            f"hash_dim={hash_dim}, decay={decay_factor}/{decay_interval}, "
            f"warmup={warmup_ratio}, anneal_end={anneal_end_ratio}"
        )

    def _init_projection(self, obs_dim: int, device: torch.device):
        gen = torch.Generator(device="cpu")
        gen.manual_seed(42)
        proj = torch.randn(obs_dim, self.hash_dim, generator=gen)
        self._projection = (proj / proj.norm(dim=0, keepdim=True)).to(device)

    def _batch_hash(self, obs_batch: torch.Tensor) -> List[int]:
        """SimHash: obs [n_proc, obs_dim] → list of int keys."""
        projected = obs_batch.float() @ self._projection
        binary = (projected > 0).cpu().numpy()
        keys = []
        for row in binary:
            k = 0
            for bit in row:
                k = (k << 1) | int(bit)
            keys.append(k)
        return keys

    def _anneal_multiplier(self, progress: float) -> float:
        """Warmup → linear decay → off."""
        if progress < self.warmup_ratio:
            return 0.0
        if self.anneal_end_ratio <= self.warmup_ratio:
            return 0.0
        if progress >= self.anneal_end_ratio:
            return 0.0
        return 1.0 - (progress - self.warmup_ratio) / (
            self.anneal_end_ratio - self.warmup_ratio
        )

    def compute_per_agent_bonus(
        self,
        obs_list: List[torch.Tensor],
        progress: float = 0.0,
    ) -> Optional[List[torch.Tensor]]:
        """
        Per-agent count-based intrinsic reward with warmup + annealing.

        Schedule (controlled by warmup_ratio and anneal_end_ratio):
          - [0, warmup_ratio): no bonus, no count updates (warmup)
          - [warmup_ratio, anneal_end_ratio): bonus * linear_decay(1→0),
            counts updated normally
          - [anneal_end_ratio, 1.0]: no bonus, no count updates (done)

        Returns None when bonus is inactive (warmup / done phases).
        """
        mult = self._anneal_multiplier(progress)
        self._last_anneal_mult = mult

        if mult <= 0.0:
            self._last_bonus_mean = 0.0
            return None

        n_agents = len(obs_list)
        n_proc = obs_list[0].size(0)
        device = obs_list[0].device

        if self._projection is None:
            self._init_projection(obs_list[0].size(-1), device)

        bonuses = []
        total_bonus = 0.0
        total_count = 0.0
        count_n = 0

        for i in range(n_agents):
            keys = self._batch_hash(obs_list[i])
            agent_bonus = torch.zeros(n_proc, 1, device=device)
            for p, key in enumerate(keys):
                self._counts[i][key] += 1.0
                c = self._counts[i][key]
                agent_bonus[p, 0] = self.coef / math.sqrt(c) * mult
                total_count += c
                count_n += 1
            bonuses.append(agent_bonus)
            total_bonus += agent_bonus.sum().item()
            self._unique_states[i] = len(self._counts[i])

        self._last_bonus_mean = total_bonus / max(n_agents * n_proc, 1)
        self._last_avg_count = total_count / max(count_n, 1)

        self._step += 1
        if self.decay_factor < 1.0 and self._step % self.decay_interval == 0:
            for i in range(n_agents):
                to_del = []
                for k in self._counts[i]:
                    self._counts[i][k] *= self.decay_factor
                    if self._counts[i][k] < 0.01:
                        to_del.append(k)
                for k in to_del:
                    del self._counts[i][k]

        return bonuses

    def log_stats(self) -> dict:
        stats = {
            "count_explore/bonus_mean": self._last_bonus_mean,
            "count_explore/avg_count": self._last_avg_count,
            "count_explore/total_unique": sum(self._unique_states),
            "count_explore/anneal_mult": self._last_anneal_mult,
        }
        for i, n in enumerate(self._unique_states):
            stats[f"count_explore/unique_agent{i}"] = n
        return stats
