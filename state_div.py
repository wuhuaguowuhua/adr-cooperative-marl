"""
State-Visitation Diversity (StateDiv) loss.

Dual to SNDMonitor's RampDiv (action-policy diversity):
RampDiv encourages inter-agent action distribution divergence on the same obs;
StateDiv encourages inter-agent state visitation divergence via a
policy-gradient-style auxiliary loss.

Crucially, StateDiv DOES NOT modify the environment reward, value function,
or advantage. Its signal is injected purely through the loss:

    L_state_i = -eta * coef * mean( log pi_i(a|s) * d_centered(s, i) )

where d(s, i) is the inter-agent distinguishability of hashed state h(s):

    d(s, i) = p_i(h(s)) / sum_j p_j(h(s)) - 1/N

Two scheduling modes:
  - "time":         triangular ramp (warmup -> peak -> down -> off). Good for MPE
                    where rewards are always negative and a reward-based gate is unnatural.
  - "reward_gated": ramp up, then permanently shut off after a sustained
                    positive-reward streak. Same as SNDMonitor's proactive trigger.
"""

import math
from collections import defaultdict
from typing import Dict, List, Optional

import torch


class StateDivMonitor:
    def __init__(
        self,
        n_agents: int,
        hash_dim: int = 12,
        coef: float = 0.1,
        # Schedule
        schedule_mode: str = "time",          # "time" or "reward_gated"
        warmup_ratio: float = 0.05,
        ramp_peak_ratio: float = 0.15,        # only used in "time" mode
        ramp_end_ratio: float = 0.30,         # only used in "time" mode
        reward_threshold: float = 2.0,        # only used in "reward_gated" mode
        shutoff_streak: int = 10,             # only used in "reward_gated" mode
        eta_max: float = 0.10,
        # Loss-shaping
        center_baseline: bool = True,
        eps: float = 1e-6,
        seed: int = 42,
    ):
        self.n_agents = n_agents
        self.hash_dim = hash_dim
        self.coef = coef
        self.schedule_mode = schedule_mode.lower()
        self.warmup_ratio = warmup_ratio
        self.ramp_peak_ratio = ramp_peak_ratio
        self.ramp_end_ratio = ramp_end_ratio
        self.reward_threshold = reward_threshold
        self.shutoff_streak = shutoff_streak
        self.eta_max = eta_max
        self.center_baseline = center_baseline
        self.eps = eps
        self.seed = seed

        # Visitation counts: per-agent hash -> count
        self._counts: List[Dict[int, float]] = [
            defaultdict(float) for _ in range(n_agents)
        ]
        self._totals: List[float] = [0.0 for _ in range(n_agents)]

        # SimHash projection (lazy init when first obs arrives)
        self._projection: Optional[torch.Tensor] = None

        # Reward-gated state
        self._reward_positive_streak: int = 0
        self._shutoff_done: bool = False
        self._warmup_exited: bool = False

        # Telemetry
        self._eta: float = 0.0
        self._last_d_mean_abs: float = 0.0
        self._last_loss_val: float = 0.0
        self._last_d_per_agent: List[float] = [0.0 for _ in range(n_agents)]
        self._last_unique: List[int] = [0 for _ in range(n_agents)]

        print(
            f"[StateDivMonitor] n_agents={n_agents}, hash_dim={hash_dim}, "
            f"coef={coef}, schedule={self.schedule_mode}, "
            f"warmup={warmup_ratio}, ramp_peak={ramp_peak_ratio}, "
            f"ramp_end={ramp_end_ratio}, reward_thresh={reward_threshold}, "
            f"shutoff_streak={shutoff_streak}, eta_max={eta_max}, "
            f"center_baseline={center_baseline}"
        )

    # ==================== SimHash ====================

    def _init_projection(self, obs_dim: int, device: torch.device) -> None:
        gen = torch.Generator(device="cpu")
        gen.manual_seed(self.seed)
        proj = torch.randn(obs_dim, self.hash_dim, generator=gen)
        self._projection = (proj / proj.norm(dim=0, keepdim=True)).to(device)

    def _hash_batch(self, obs_batch: torch.Tensor) -> List[int]:
        """obs [B, obs_dim] -> list of int hash keys."""
        if self._projection is None:
            self._init_projection(obs_batch.size(-1), obs_batch.device)
        elif self._projection.device != obs_batch.device:
            self._projection = self._projection.to(obs_batch.device)
        projected = obs_batch.float() @ self._projection
        binary = (projected > 0).cpu().numpy()
        keys = []
        for row in binary:
            k = 0
            for bit in row:
                k = (k << 1) | int(bit)
            keys.append(k)
        return keys

    # ==================== Visitation tracking ====================

    def update_visits(self, obs_list: List[torch.Tensor]) -> None:
        """Called from training loop after each env.step(). Updates per-agent counts.

        obs_list: list of length N, each [num_processes, obs_dim].
        """
        for i, obs_i in enumerate(obs_list):
            keys = self._hash_batch(obs_i)
            for k in keys:
                self._counts[i][k] += 1.0
            self._totals[i] += len(keys)
            self._last_unique[i] = len(self._counts[i])

    # ==================== Distinguishability ====================

    def compute_distinguishability(
        self, agent_id: int, obs_batch: torch.Tensor
    ) -> torch.Tensor:
        """Compute d(s, i) for each obs in batch.

        Returns: [B, 1] tensor (no grad), values in approx [-1/N, (N-1)/N].
        """
        keys = self._hash_batch(obs_batch)
        N = self.n_agents
        device = obs_batch.device

        d = torch.zeros(len(keys), 1, device=device)
        # Pre-compute per-agent normalization
        totals = [max(t, self.eps) for t in self._totals]

        for b, k in enumerate(keys):
            p = [self._counts[j].get(k, 0.0) / totals[j] for j in range(N)]
            denom = sum(p) + self.eps
            d[b, 0] = p[agent_id] / denom - 1.0 / N

        return d

    # ==================== Schedule (eta computation) ====================

    def compute_eta(self, progress: float) -> float:
        if self.schedule_mode == "time":
            return self._compute_eta_time(progress)
        elif self.schedule_mode == "reward_gated":
            return self._compute_eta_reward_gated(progress)
        else:
            return 0.0

    def _compute_eta_time(self, progress: float) -> float:
        """Triangular ramp: 0 -> eta_max -> 0 over [warmup, ramp_end]."""
        if progress < self.warmup_ratio or progress >= self.ramp_end_ratio:
            self._eta = 0.0
            return 0.0
        peak = max(self.ramp_peak_ratio, self.warmup_ratio + 1e-6)
        peak = min(peak, self.ramp_end_ratio - 1e-6)
        if progress < peak:
            frac = (progress - self.warmup_ratio) / (peak - self.warmup_ratio)
        else:
            frac = 1.0 - (progress - peak) / (self.ramp_end_ratio - peak)
        eta = max(0.0, frac) * self.eta_max
        eta = min(eta, self.eta_max)
        self._eta = eta
        return eta

    def _compute_eta_reward_gated(self, progress: float) -> float:
        if progress < self.warmup_ratio:
            self._eta = 0.0
            return 0.0
        if not self._warmup_exited:
            self._warmup_exited = True
            self._reward_positive_streak = 0
        if self._shutoff_done:
            self._eta = 0.0
            return 0.0
        if self._reward_positive_streak >= self.shutoff_streak:
            self._shutoff_done = True
            self._eta = 0.0
            return 0.0
        ramp_start = self.warmup_ratio
        ramp_end = self.ramp_end_ratio
        if ramp_end <= ramp_start:
            ramp_frac = 1.0
        else:
            ramp_frac = min(1.0, (progress - ramp_start) / (ramp_end - ramp_start))
        eta = (ramp_frac ** 2) * self.eta_max
        eta = min(eta, self.eta_max)
        self._eta = eta
        return eta

    def update_reward(self, mean_reward: float) -> None:
        """Track positive-reward streak (only meaningful in reward_gated mode)."""
        if mean_reward > self.reward_threshold:
            self._reward_positive_streak += 1
        else:
            self._reward_positive_streak = 0

    def needs_grad(self, progress: float) -> bool:
        return self.compute_eta(progress) >= 1e-4

    # ==================== Loss ====================

    def compute_state_div_loss(
        self,
        agent_id: int,
        obs_batch: torch.Tensor,
        action_log_probs: torch.Tensor,
        progress: float,
    ) -> torch.Tensor:
        """Compute the state-diversity loss for a single agent.

        obs_batch:         [B, obs_dim]      (used for d only, no grad path through obs)
        action_log_probs:  [B, 1]            (must carry grad)

        Loss = -eta * coef * mean( log_prob * d_centered )
        """
        eta = self.compute_eta(progress)
        if eta < 1e-4:
            self._last_loss_val = 0.0
            return obs_batch.new_zeros(())

        d = self.compute_distinguishability(agent_id, obs_batch)  # [B, 1], no grad
        d = d.detach()

        if self.center_baseline:
            d = d - d.mean()

        # Telemetry
        self._last_d_mean_abs = d.abs().mean().item()
        self._last_d_per_agent[agent_id] = d.mean().item()

        loss = -eta * self.coef * (action_log_probs * d).mean()
        self._last_loss_val = float(loss.item())
        return loss

    # ==================== Logging ====================

    def log_stats(self) -> dict:
        stats = {
            "state_div/eta": self._eta,
            "state_div/effective_coef": self._eta * self.coef,
            "state_div/d_mean_abs": self._last_d_mean_abs,
            "state_div/loss_val": self._last_loss_val,
            "state_div/reward_positive_streak": self._reward_positive_streak,
            "state_div/shutoff_done": float(self._shutoff_done),
            "state_div/total_unique": sum(self._last_unique),
        }
        for i in range(self.n_agents):
            stats[f"state_div/unique_agent{i}"] = self._last_unique[i]
            stats[f"state_div/d_mean_agent{i}"] = self._last_d_per_agent[i]
        return stats
