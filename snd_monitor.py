import math
import torch
import torch.nn.functional as F
from typing import List, Optional


class SNDMonitor:
    """
    Reward-Coupled Diversity Control (RCDC).

    Uses dual-rate reward EMA to compute a continuous diversity coefficient eta(t).

    Four trigger modes:
      - "rising":     eta ∝ positive reward velocity (active during rapid improvement)
      - "stagnation": eta ∝ reward flatness (active during plateaus)
      - "proactive":  eta = 1.0 before rewards appear, 0 after (early exploration only)
      - "feedback":   closed-loop reward-coupled direction. magnitude ramps as in
                      proactive, but direction ∈ [-1, +1] flips sign when reward
                      drops. Starts at +1 (trust encourage); if encourage is
                      falsified (reward drops while eta active) direction slides
                      toward 0 and can cross into suppress territory. Turns the
                      hand-set sign of global_coef into an online adaptive signal.

    Gated by reward_level_threshold to protect early learning (rising/stagnation modes).
    """

    def __init__(
        self,
        metric: str = "l2",
        tau: float = 0.01,
        global_coef: float = 0.1,
        loss_mode: str = "rcdc",
        trigger: str = "rising",
        warmup_ratio: float = 0.15,
        tau_fast: float = 0.1,
        tau_slow: float = 0.02,
        sensitivity: float = 5.0,
        momentum_beta: float = 0.95,
        activation_threshold: float = 0.2,
        reward_level_threshold: float = 0.5,
        diversity_ceiling: float = 0.5,
        min_maturity: int = 50,
        proactive_shutoff: int = 10,
        proactive_ramp_end: float = 0.5,
        proactive_reward_threshold: float = 0.01,
        proactive_ramp_power: float = 1.0,
        proactive_time_shutoff: float = 1.0,
        eta_max: float = 1.0,
        feedback_gain: float = 0.05,
        feedback_trend_threshold: float = 0.001,
        feedback_init_direction: float = 1.0,
        feedback_decel_threshold: float = 0.5,
        feedback_min_peak: float = 0.02,
        feedback_min_interval: int = 50,
        feedback_min_progress: float = 0.05,
    ):
        self.metric = metric.lower()
        self.tau = tau
        self.global_coef = global_coef
        self.loss_mode = loss_mode.lower()
        self.trigger = trigger.lower()

        self.warmup_ratio = warmup_ratio
        self.tau_fast = tau_fast
        self.tau_slow = tau_slow
        self.sensitivity = sensitivity
        self.momentum_beta = momentum_beta
        self.activation_threshold = activation_threshold
        self.reward_level_threshold = reward_level_threshold
        self.diversity_ceiling = diversity_ceiling
        self.min_maturity = min_maturity
        self.proactive_shutoff = proactive_shutoff
        self.proactive_ramp_end = proactive_ramp_end
        self.proactive_reward_threshold = proactive_reward_threshold
        self.proactive_ramp_power = proactive_ramp_power
        self.proactive_time_shutoff = proactive_time_shutoff
        self.eta_max = eta_max
        self.feedback_gain = feedback_gain
        self.feedback_trend_threshold = feedback_trend_threshold
        self.feedback_init_direction = max(-1.0, min(1.0, feedback_init_direction))
        self.feedback_decel_threshold = feedback_decel_threshold
        self.feedback_min_peak = feedback_min_peak
        self.feedback_min_interval = feedback_min_interval
        self.feedback_min_progress = feedback_min_progress

        self.ema: Optional[torch.Tensor] = None
        self._last_diversity: float = 0.0

        self._reward_ema_fast: Optional[float] = None
        self._reward_ema_slow: Optional[float] = None
        self._ever_positive: bool = False
        self._max_reward: float = 0.0
        self._reward_positive_streak: int = 0
        self._proactive_shutoff_done: bool = False
        self._warmup_exited: bool = False

        self._momentum: float = 0.0
        self._eta: float = 0.0
        self._reward_level: float = 0.0
        self._signal: float = 0.0
        self._direction: float = self.feedback_init_direction
        self._feedback_norm_v: float = 0.0
        self._feedback_norm_v_peak: float = 0.0
        self._feedback_update_count: int = 0
        self._feedback_last_trigger: int = -10**9
        self._feedback_trigger_count: int = 0

        print(
            f"[SNDMonitor-RCDC] metric='{self.metric}', max_coef={self.global_coef}, "
            f"mode='{self.loss_mode}', trigger='{self.trigger}', "
            f"warmup={self.warmup_ratio}, tau_fast={self.tau_fast}, "
            f"tau_slow={self.tau_slow}, sensitivity={self.sensitivity}, "
            f"momentum_beta={self.momentum_beta}, "
            f"activation_thresh={self.activation_threshold}, "
            f"reward_level_thresh={self.reward_level_threshold}, "
            f"diversity_ceiling={self.diversity_ceiling}, "
            f"min_maturity={self.min_maturity}, "
            f"proactive_shutoff={self.proactive_shutoff}, "
            f"proactive_ramp_end={self.proactive_ramp_end}, "
            f"proactive_reward_thresh={self.proactive_reward_threshold}, "
            f"proactive_ramp_power={self.proactive_ramp_power}, "
            f"proactive_time_shutoff={self.proactive_time_shutoff}, "
            f"eta_max={self.eta_max}, "
            f"feedback_gain={self.feedback_gain}, "
            f"feedback_trend_thresh={self.feedback_trend_threshold}, "
            f"feedback_init_direction={self.feedback_init_direction}, "
            f"feedback_decel_thresh={self.feedback_decel_threshold}, "
            f"feedback_min_peak={self.feedback_min_peak}, "
            f"feedback_min_interval={self.feedback_min_interval}, "
            f"feedback_min_progress={self.feedback_min_progress}"
        )

    # ==================== reward tracking ====================

    def update_reward(self, mean_reward: float):
        if mean_reward > self.proactive_reward_threshold:
            self._ever_positive = True
            self._reward_positive_streak += 1
        else:
            self._reward_positive_streak = 0

        if mean_reward > self._max_reward:
            self._max_reward = mean_reward

        if self._reward_ema_fast is None:
            self._reward_ema_fast = mean_reward
            self._reward_ema_slow = mean_reward
        else:
            self._reward_ema_fast = (
                self.tau_fast * mean_reward
                + (1 - self.tau_fast) * self._reward_ema_fast
            )
            self._reward_ema_slow = (
                self.tau_slow * mean_reward
                + (1 - self.tau_slow) * self._reward_ema_slow
            )

    # ==================== eta computation ====================

    def compute_eta(self, progress: float) -> float:
        """
        Continuous diversity coefficient in [0, 1].

        trigger="rising":     high when reward is actively improving
        trigger="stagnation": high when reward is flat
        trigger="proactive":  1.0 before rewards appear, 0 after

        Gated by reward_level_threshold to protect early learning (rising/stagnation).
        """
        if progress < self.warmup_ratio:
            self._eta = 0.0
            return 0.0

        if self.trigger == "feedback":
            if progress >= self.proactive_time_shutoff:
                self._eta = 0.0
                return 0.0

            ramp_start = self.warmup_ratio
            ramp_end = self.proactive_ramp_end
            if ramp_end <= ramp_start:
                ramp_frac = 1.0
            else:
                ramp_frac = min(1.0, (progress - ramp_start) / (ramp_end - ramp_start))
            ramp_value = ramp_frac ** self.proactive_ramp_power
            magnitude = ramp_value * self.eta_max

            if self.diversity_ceiling > 0:
                ratio = self._last_diversity / (self.diversity_ceiling + 1e-6)
                magnitude = magnitude * max(0.0, 1.0 - ratio)
            magnitude = min(magnitude, self.eta_max)

            self._feedback_update_count += 1
            if self._reward_ema_fast is not None:
                velocity = self._reward_ema_fast - self._reward_ema_slow
                norm_v = velocity / (self._max_reward + 1e-6)
                self._feedback_norm_v = norm_v

                if (
                    progress > self.feedback_min_progress
                    and norm_v > self._feedback_norm_v_peak
                ):
                    self._feedback_norm_v_peak = norm_v

                signed_eta = magnitude * self._direction
                ready = (
                    abs(signed_eta) > 0.01
                    and progress > self.feedback_min_progress
                )
                if ready:
                    has_meaningful_history = (
                        self._feedback_norm_v_peak > self.feedback_min_peak
                    )
                    cond_drop = (
                        has_meaningful_history
                        and norm_v < -self.feedback_trend_threshold
                    )
                    cond_decel = (
                        has_meaningful_history
                        and norm_v
                        < self._feedback_norm_v_peak
                        * (1.0 - self.feedback_decel_threshold)
                    )
                    interval = (
                        self._feedback_update_count - self._feedback_last_trigger
                    )
                    if (cond_drop or cond_decel) and interval >= self.feedback_min_interval:
                        self._direction -= self.feedback_gain
                        self._direction = max(-1.0, min(1.0, self._direction))
                        self._feedback_last_trigger = self._feedback_update_count
                        self._feedback_trigger_count += 1

            self._eta = magnitude
            return magnitude

        if self.trigger == "proactive":
            if not self._warmup_exited:
                self._warmup_exited = True
                self._reward_positive_streak = 0
            if self._proactive_shutoff_done:
                self._eta = 0.0
                return 0.0
            if self._reward_positive_streak >= self.proactive_shutoff:
                self._proactive_shutoff_done = True
                self._eta = 0.0
                return 0.0
            if progress >= self.proactive_time_shutoff:
                self._proactive_shutoff_done = True
                self._eta = 0.0
                return 0.0
            ramp_start = self.warmup_ratio
            ramp_end = self.proactive_ramp_end
            if ramp_end <= ramp_start:
                ramp_frac = 1.0
            else:
                ramp_frac = min(1.0, (progress - ramp_start) / (ramp_end - ramp_start))
            ramp_value = ramp_frac ** self.proactive_ramp_power
            # When eta_max > 1.0, ramp_value (capped at 1.0) cannot reach eta_max,
            # so we scale the ramp directly to eta_max. For eta_max <= 1.0 we keep
            # the legacy "ramp to 1.0 then post-cap" behaviour to preserve exact
            # reproducibility of all historical RWARE/MPE experiments.
            if self.eta_max > 1.0:
                eta = ramp_value * self.eta_max
            else:
                eta = ramp_value
            if self.diversity_ceiling > 0:
                ratio = self._last_diversity / (self.diversity_ceiling + 1e-6)
                eta = eta * max(0.0, 1.0 - ratio)
            eta = min(eta, self.eta_max)
            self._eta = eta
            return eta

        if self._reward_ema_fast is None or not self._ever_positive:
            self._eta = 0.0
            return 0.0

        reward_level = max(0.0, self._reward_ema_fast) / (self._max_reward + 1e-6)
        reward_level = min(1.0, reward_level)
        self._reward_level = reward_level

        if reward_level < self.reward_level_threshold:
            self._momentum = self.momentum_beta * self._momentum
            self._eta = 0.0
            return 0.0

        if self._reward_positive_streak < self.min_maturity:
            self._momentum = self.momentum_beta * self._momentum
            self._eta = 0.0
            return 0.0

        if self.trigger == "rising":
            velocity = (self._reward_ema_fast - self._reward_ema_slow)
            rising_vel = max(0.0, velocity) / (self._max_reward + 1e-6)
            signal = 1.0 - math.exp(-self.sensitivity * rising_vel)
        else:
            abs_velocity = abs(self._reward_ema_fast - self._reward_ema_slow)
            norm_velocity = abs_velocity / (self._max_reward + 1e-6)
            signal = math.exp(-self.sensitivity * norm_velocity)

        self._signal = signal
        self._momentum = (
            self.momentum_beta * self._momentum
            + (1 - self.momentum_beta) * signal
        )

        if self._momentum <= self.activation_threshold:
            eta = 0.0
        else:
            eta = min(
                1.0,
                (self._momentum - self.activation_threshold)
                / (1.0 - self.activation_threshold),
            )

        if self.diversity_ceiling > 0 and eta > 0:
            ratio = self._last_diversity / (self.diversity_ceiling + 1e-6)
            eta = eta * max(0.0, 1.0 - ratio)

        eta = min(eta, self.eta_max)
        self._eta = eta
        return eta

    def needs_grad(self, progress: float) -> bool:
        """Pre-check: call compute_eta and return whether gradient is needed."""
        if self.loss_mode == "rcdc":
            return self.compute_eta(progress) >= 0.01
        elif self.loss_mode == "encourage":
            return self._phase_coef(progress) > 0
        return False

    # ==================== diversity metrics (differentiable) ====================

    @staticmethod
    def _compute_l2(probs_list: List[torch.Tensor]) -> torch.Tensor:
        n = len(probs_list)
        if n < 2:
            return probs_list[0].new_tensor(0.0)
        acc = probs_list[0].new_tensor(0.0)
        cnt = 0
        for i in range(n):
            for j in range(i + 1, n):
                d = (probs_list[i] - probs_list[j]).pow(2).sum(-1).mean()
                acc = acc + torch.sqrt(torch.clamp(d, min=1e-8))
                cnt += 1
        return acc / max(1, cnt)

    @staticmethod
    def _compute_action_var(probs_list: List[torch.Tensor]) -> torch.Tensor:
        P = torch.stack(probs_list, dim=0)
        return P.var(dim=0).mean()

    @staticmethod
    def _compute_action_std(probs_list: List[torch.Tensor]) -> torch.Tensor:
        if len(probs_list) < 2:
            return probs_list[0].new_tensor(0.0)
        P = torch.stack(probs_list, dim=0)
        return P.std(dim=0).mean()

    def compute_diversity(self, probs_list: List[torch.Tensor]) -> torch.Tensor:
        if self.metric == "action_var":
            return self._compute_action_var(probs_list)
        elif self.metric == "l2":
            return self._compute_l2(probs_list)
        else:
            return self._compute_action_std(probs_list)

    # ==================== EMA tracking ====================

    def update_ema(self, diversity: torch.Tensor):
        d = diversity.detach()
        if self.ema is None:
            self.ema = d
        else:
            self.ema = self.tau * d + (1.0 - self.tau) * self.ema
        self._last_diversity = d.item()

    # ==================== loss computation ====================

    def diversity_loss(
        self,
        probs_list: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute diversity loss using the pre-computed eta (from needs_grad / compute_eta).
        When eta ~ 0, the probs_list should already be no-grad tensors.
        Strength is controlled solely by eta * global_coef (no per-agent weight).
        """
        diversity = self.compute_diversity(probs_list)
        self.update_ema(diversity)

        if self.loss_mode == "rcdc":
            if self._eta < 0.01:
                return probs_list[0].new_tensor(0.0)
            if self.trigger == "feedback":
                return -self._eta * self._direction * self.global_coef * diversity
            return -self._eta * self.global_coef * diversity

        elif self.loss_mode == "encourage":
            coef = getattr(self, "_cached_phase_coef", 0.0)
            if coef <= 0:
                return probs_list[0].new_tensor(0.0)
            return -coef * diversity

        return probs_list[0].new_tensor(0.0)

    def _phase_coef(self, progress: float) -> float:
        if progress < 0.3:
            c = self.global_coef
        elif progress < 0.7:
            c = self.global_coef * (0.7 - progress) / 0.4
        else:
            c = 0.0
        self._cached_phase_coef = c
        return c

    # ==================== logging ====================

    def log_stats(self) -> dict:
        direction = self._direction if self.trigger == "feedback" else 1.0
        effective_coef = self._eta * direction * self.global_coef
        return {
            "snd/diversity": self._last_diversity,
            "snd/ema": float(self.ema) if self.ema is not None else 0.0,
            "snd/eta": self._eta,
            "snd/effective_coef": effective_coef,
            "snd/momentum": self._momentum,
            "snd/signal": self._signal,
            "snd/reward_level": self._reward_level,
            "snd/max_reward": self._max_reward,
            "snd/reward_positive_streak": self._reward_positive_streak,
            "snd/proactive_shutoff_done": float(self._proactive_shutoff_done),
            "snd/reward_ema_fast": (
                self._reward_ema_fast if self._reward_ema_fast is not None else 0.0
            ),
            "snd/reward_ema_slow": (
                self._reward_ema_slow if self._reward_ema_slow is not None else 0.0
            ),
            "snd/direction": direction,
            "snd/feedback_norm_v": self._feedback_norm_v,
            "snd/feedback_norm_v_peak": self._feedback_norm_v_peak,
            "snd/feedback_trigger_count": self._feedback_trigger_count,
            "snd/signed_eta": self._eta * direction,
        }
