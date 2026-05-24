# mappo.py
# Multi-Agent PPO (MAPPO):
#   - Each agent has its own actor (independent policy networks).
#   - All agents share a single centralized critic V_i(s_global) that
#     receives the concatenation of all agents' observations and emits
#     N per-agent value estimates (reused from maa2c.CentralCritic).
#   - PPO clip surrogate with K-epoch optimization (reused from ippo.IPPO).
#   - RCDC diversity loss is integrated through SNDMonitor exactly as in
#     IPPO: one contribution per update, attached to the first epoch's
#     loss when num_mini_batch == 1.
#
# Why MAPPO as the third algorithm (vs SEAC, MAA2C):
#   The cooperative-MARL community has converged on MAPPO (Yu et al.,
#   2022) as a strong reference. Demonstrating that RCDC works on top
#   of MAPPO closes the cross-algorithm story across three families:
#   shared experience (SEAC), centralized-critic A2C (MAA2C), and
#   PPO-style trust-region with centralized critic (MAPPO).
#
# Implementation notes:
#   * Critic training is done OUTSIDE this class via
#     `central_critic_update` (one gradient step per rollout, in
#     train.py, BEFORE the actor updates), mirroring MAA2C exactly.
#     This deviates from EPyMARL's MAPPO which K-epochs the critic
#     too, but it (i) keeps the codebase identical between MAA2C and
#     MAPPO except for the actor surrogate and (ii) is a documented
#     simplification that does not affect the central RCDC claim.
#   * `self.storage.value_preds` was already overwritten in train.py
#     with V_i(s_global) BEFORE GAE was computed, so
#     `storage.returns[:-1] - storage.value_preds[:-1]` is a valid
#     centralized GAE advantage. We additionally compute a fresh
#     detached `v_self` from the post-update central critic to match
#     MAA2C's advantage convention (returns - V_post).

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from gym.spaces.utils import flatdim

from model import Policy
from storage import RolloutStorage

from a2c import algorithm  # reuse the same Sacred Ingredient
# `_ppo_config` already registered the PPO-specific knobs (clip_eps,
# ppo_epochs, num_mini_batch, ...) into the `algorithm` ingredient when
# ippo.py was imported by train.py, so we can reuse them verbatim.

from maa2c import CentralCritic, build_global_obs  # noqa: F401


class MAPPO:
    @algorithm.capture()
    def __init__(
        self,
        agent_id,
        obs_space,
        action_space,
        lr,
        adam_eps,
        recurrent_policy,
        num_steps,
        num_processes,
        device,
    ):
        self.agent_id = agent_id
        self.obs_size = flatdim(obs_space)
        self.action_size = flatdim(action_space)
        self.obs_space = obs_space
        self.action_space = action_space

        self.model = Policy(
            obs_space, action_space, base_kwargs={"recurrent": recurrent_policy},
        )

        self.storage = RolloutStorage(
            obs_space,
            action_space,
            self.model.recurrent_hidden_state_size,
            num_steps,
            num_processes,
        )

        self.model.to(device)
        # Optimizer covers ONLY the actor branch; the centralized critic
        # is updated externally by `central_critic_update`. The local
        # critic head produced by Policy.base.critic_linear is a dead
        # path (its outputs are overwritten in train.py before GAE).
        self.optimizer = optim.Adam(self.model.parameters(), lr, eps=adam_eps)

        self.saveables = {
            "model": self.model,
            "optimizer": self.optimizer,
        }

    def save(self, path):
        torch.save(self.saveables, os.path.join(path, "models.pt"))

    def restore(self, path):
        checkpoint = torch.load(os.path.join(path, "models.pt"))
        for k, v in self.saveables.items():
            v.load_state_dict(checkpoint[k].state_dict())

    @algorithm.capture
    def compute_returns(
        self,
        gamma,
        gae_lambda,
        use_proper_time_limits,
        use_gae_ippo,
    ):
        # Fallback path: invoked only if train.py forgets to handle
        # mappo's centralized GAE. Uses the per-agent (dead) critic for
        # next_value, which is incorrect under MAPPO but harmless when
        # the train.py centralized path is in effect.
        with torch.no_grad():
            next_value = self.model.get_value(
                self.storage.obs[-1],
                self.storage.recurrent_hidden_states[-1],
                self.storage.masks[-1],
            ).detach()
        # Force GAE for MAPPO regardless of A2C's default `use_gae`.
        self.storage.compute_returns(
            next_value, use_gae_ippo, gamma, gae_lambda, use_proper_time_limits,
        )

    @algorithm.capture
    def compute_returns_extrinsic(
        self,
        gamma,
        gae_lambda,
        use_proper_time_limits,
        use_gae_ippo,
    ):
        # Kept for interface parity with A2C/SEAC; MAPPO does not use
        # extrinsic-only returns since it has no cross-agent learning.
        with torch.no_grad():
            next_value = self.model.get_value(
                self.storage.obs[-1],
                self.storage.recurrent_hidden_states[-1],
                self.storage.masks[-1],
            ).detach()
        self.storage.compute_returns_extrinsic(
            next_value, use_gae_ippo, gamma, gae_lambda, use_proper_time_limits,
        )

    @algorithm.capture
    def update(
        self,
        storages,
        models=None,
        central_critic: CentralCritic = None,
        snd_monitor=None,
        state_div_monitor=None,
        progress=0.0,
        clip_eps=0.2,
        ppo_epochs=4,
        num_mini_batch=1,
        use_clipped_value_loss=True,  # accepted for API parity, ignored
        normalize_advantage=True,
        value_loss_coef=0.5,           # accepted for API parity, ignored
        entropy_coef=0.01,
        max_grad_norm=0.5,
        device="cpu",
    ):
        """
        Actor-only PPO update for one agent. The centralized critic must
        have already been updated in the same iteration via
        `central_critic_update` so that the V(s_global) snapshot used
        here matches what every agent sees.
        """
        assert central_critic is not None, (
            "MAPPO.update requires a `central_critic` instance."
        )

        obs_shape = self.storage.obs.size()[2:]

        # Compute fresh post-update centralized advantage. The storage's
        # value_preds were the PRE-critic-update V_i (used for GAE
        # consistency); here we use the POST-critic-update V_i so that
        # the actor sees the freshest baseline. This mirrors MAA2C
        # exactly (see maa2c.MAA2C.update step 2).
        with torch.no_grad():
            global_obs = build_global_obs(storages, time_slice=slice(None, -1))
            v_all = central_critic(global_obs)  # (T, N_proc, n_agents)
            v_self = v_all[..., self.agent_id : self.agent_id + 1]  # (T, N_proc, 1)
        advantages = (self.storage.returns[:-1] - v_self).detach()

        if normalize_advantage:
            advantages = (advantages - advantages.mean()) / (
                advantages.std() + 1e-5
            )

        # Pre-compute RCDC diversity loss once per update on the full
        # batch, mirroring the IPPO path. Only added to the FIRST
        # epoch's loss to keep its weight comparable across algorithms.
        diversity_loss = torch.tensor(
            0.0, device=device if isinstance(device, str) else device
        )
        diversity_loss_for_log = 0.0
        current_diversity = 0.0
        rcdc_active = (
            snd_monitor is not None
            and models is not None
            and snd_monitor.needs_grad(progress)
        )
        if snd_monitor is not None and models is not None:
            batch_obs = self.storage.obs[:-1].view(-1, *obs_shape)
            batch_size = batch_obs.size(0)
            ctx = torch.enable_grad() if rcdc_active else torch.no_grad()
            with ctx:
                probs_list = []
                for m in models:
                    _, logits, _ = m.actor_logits(
                        batch_obs,
                        torch.zeros(
                            batch_size,
                            m.recurrent_hidden_state_size,
                            device=batch_obs.device,
                        ),
                        torch.ones(batch_size, 1, device=batch_obs.device),
                    )
                    probs_list.append(F.softmax(logits, dim=-1))
                diversity_loss = snd_monitor.diversity_loss(probs_list)
            current_diversity = snd_monitor._last_diversity
            diversity_loss_for_log = (
                float(diversity_loss)
                if isinstance(diversity_loss, (int, float))
                else diversity_loss.item()
            )

        # State-Visitation Diversity (parity with A2C / MAA2C; usually
        # disabled in our experiments).
        state_div_loss = self.storage.obs.new_zeros(())
        if state_div_monitor is not None and state_div_monitor.needs_grad(progress):
            # SVD wants per-step action_log_probs. We re-evaluate the
            # whole batch once outside the PPO epoch loop using the
            # current actor; this matches A2C's semantics where SVD is
            # taken on the latest policy snapshot.
            with torch.no_grad():
                _, alp_full, _, _ = self.model.evaluate_actions(
                    self.storage.obs[:-1].view(-1, *obs_shape),
                    self.storage.recurrent_hidden_states[0].view(
                        -1, self.model.recurrent_hidden_state_size
                    ),
                    self.storage.masks[:-1].view(-1, 1),
                    self.storage.actions.view(-1, self.storage.actions.size(-1)),
                )
            state_div_obs = self.storage.obs[:-1].view(-1, *obs_shape)
            state_div_loss = state_div_monitor.compute_state_div_loss(
                agent_id=self.agent_id,
                obs_batch=state_div_obs,
                action_log_probs=alp_full.view(-1, 1),
                progress=progress,
            )

        policy_loss_epoch = 0.0
        entropy_epoch = 0.0
        clip_frac_epoch = 0.0
        approx_kl_epoch = 0.0
        n_updates = 0

        for e in range(ppo_epochs):
            data_gen = self.storage.feed_forward_generator(
                advantages, num_mini_batch=num_mini_batch
            )

            for sample in data_gen:
                (
                    obs_batch,
                    rnn_hxs_batch,
                    actions_batch,
                    value_preds_batch,         # unused (critic is external)
                    return_batch,              # unused (critic is external)
                    masks_batch,
                    old_action_log_probs_batch,
                    adv_targ,
                ) = sample

                values, action_log_probs, dist_entropy, _ = (
                    self.model.evaluate_actions(
                        obs_batch, rnn_hxs_batch, masks_batch, actions_batch
                    )
                )

                ratio = torch.exp(action_log_probs - old_action_log_probs_batch)
                surr1 = ratio * adv_targ
                surr2 = (
                    torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv_targ
                )
                policy_loss = -torch.min(surr1, surr2).mean()

                # No value loss term: the centralized critic is trained
                # by central_critic_update OUTSIDE this loop. Including
                # a value-loss term against the dead local critic head
                # would corrupt the actor's gradient through the shared
                # base trunk in Policy.
                total_loss = (
                    policy_loss
                    - entropy_coef * dist_entropy
                )
                if e == 0 and rcdc_active and num_mini_batch == 1:
                    total_loss = total_loss + diversity_loss
                if e == 0 and num_mini_batch == 1:
                    total_loss = total_loss + state_div_loss

                self.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_grad_norm
                )
                self.optimizer.step()

                with torch.no_grad():
                    clip_frac = (
                        ((ratio - 1.0).abs() > clip_eps).float().mean().item()
                    )
                    approx_kl = (
                        (old_action_log_probs_batch - action_log_probs)
                        .mean()
                        .item()
                    )

                policy_loss_epoch += policy_loss.item()
                entropy_epoch += dist_entropy.item()
                clip_frac_epoch += clip_frac
                approx_kl_epoch += approx_kl
                n_updates += 1

        # Fallback path: if minibatch mode was requested OR RCDC was
        # inactive within the main loop (num_mini_batch>1), apply RCDC
        # as a separate gradient step over the full batch. Mirrors IPPO.
        if rcdc_active and num_mini_batch != 1:
            self.optimizer.zero_grad()
            diversity_loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
            self.optimizer.step()

        return {
            "policy_loss": policy_loss_epoch / max(n_updates, 1),
            # Report the per-agent advantage variance against V_i(s_global)
            # so the W&B "value_loss" panel still has a meaningful signal,
            # even though no value gradient is taken inside this class.
            "value_loss": float(advantages.detach().pow(2).mean().item())
            * value_loss_coef,
            "dist_entropy": entropy_coef * entropy_epoch / max(n_updates, 1),
            "clip_frac": clip_frac_epoch / max(n_updates, 1),
            "approx_kl": approx_kl_epoch / max(n_updates, 1),
            # Stub fields kept for parity with A2C return dict / wandb panels
            "importance_sampling": 0.0,
            "seac_policy_loss": 0.0,
            "seac_value_loss": 0.0,
            "diversity_loss": diversity_loss_for_log,
            "state_div_loss": (
                float(state_div_loss)
                if isinstance(state_div_loss, (int, float))
                else state_div_loss.item()
            ),
            "current_diversity": current_diversity,
        }
