# ippo.py
# Independent PPO (IPPO): each agent learns its own actor-critic via PPO clip
# surrogate, with no cross-agent experience sharing (unlike SEAC).
#
# Compatible with the existing train.py loop:
#   - same Policy / RolloutStorage / Sacred algorithm config as a2c.py
#   - same compute_returns / update / save / restore interface as A2C
#   - same RCDC diversity loss integration via SNDMonitor
#
# Differences from A2C / SEAC:
#   1. PPO clip surrogate (with optional value clipping)
#   2. K-epoch update with optional minibatch (num_mini_batch=1 -> full batch)
#   3. NO importance-weighted cross-agent learning (no `seac_*` terms)
#   4. RCDC diversity loss is added to the FIRST epoch's loss (single
#      contribution per update), keeping the relative diversity weight
#      comparable to the A2C setup.

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from gym.spaces.utils import flatdim

from model import Policy
from storage import RolloutStorage

from a2c import algorithm  # reuse the same Sacred Ingredient


# Extend the existing algorithm config with PPO-specific knobs. Sacred allows
# multiple `@ingredient.config` blocks; values declared here are merged into
# the same algorithm namespace as the A2C config.
@algorithm.config
def _ppo_config():
    # PPO clip range
    clip_eps = 0.2
    # Number of full passes over the rollout per update
    ppo_epochs = 4
    # Number of minibatches per epoch. With num_processes=4 and num_steps=5
    # the per-update batch is only 20 transitions, so num_mini_batch=1 (full
    # batch) is the only sensible choice; left configurable for future scaling.
    num_mini_batch = 1
    # Whether to clip the value-function loss (PPO2 style)
    use_clipped_value_loss = True
    # Whether to normalise advantages within an update
    normalize_advantage = True
    # Force GAE on by default for IPPO (PPO standard); A2C default is False
    use_gae_ippo = True


class IPPO:
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
        with torch.no_grad():
            next_value = self.model.get_value(
                self.storage.obs[-1],
                self.storage.recurrent_hidden_states[-1],
                self.storage.masks[-1],
            ).detach()

        # Force GAE for IPPO regardless of A2C's default `use_gae`.
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
        # Kept for interface parity with A2C/SEAC; IPPO does not use the
        # extrinsic-only returns since it has no cross-agent learning, but
        # train.py still calls this every iteration.
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
        snd_monitor=None,
        state_div_monitor=None,
        progress=0.0,
        clip_eps=0.2,
        ppo_epochs=4,
        num_mini_batch=1,
        use_clipped_value_loss=True,
        normalize_advantage=True,
        value_loss_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        device="cpu",
    ):
        obs_shape = self.storage.obs.size()[2:]

        advantages = (
            self.storage.returns[:-1] - self.storage.value_preds[:-1]
        )
        if normalize_advantage:
            advantages = (advantages - advantages.mean()) / (
                advantages.std() + 1e-5
            )

        # Pre-compute RCDC diversity loss once per update on the full batch,
        # mirroring the A2C path. Only added to the FIRST epoch's loss to
        # keep its weight comparable.
        diversity_loss = torch.tensor(0.0, device=device if isinstance(device, str) else device)
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

        policy_loss_epoch = 0.0
        value_loss_epoch = 0.0
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
                    value_preds_batch,
                    return_batch,
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

                if use_clipped_value_loss:
                    value_pred_clipped = value_preds_batch + (
                        values - value_preds_batch
                    ).clamp(-clip_eps, clip_eps)
                    value_loss_unclipped = (values - return_batch).pow(2)
                    value_loss_clipped = (value_pred_clipped - return_batch).pow(2)
                    value_loss = (
                        0.5
                        * torch.max(value_loss_unclipped, value_loss_clipped).mean()
                    )
                else:
                    value_loss = 0.5 * (return_batch - values).pow(2).mean()

                total_loss = (
                    policy_loss
                    + value_loss_coef * value_loss
                    - entropy_coef * dist_entropy
                )
                if e == 0 and rcdc_active and num_mini_batch == 1:
                    # Single contribution per update, only when full-batch
                    # mode is used (to keep RCDC weight comparable to A2C).
                    total_loss = total_loss + diversity_loss

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
                value_loss_epoch += value_loss.item()
                entropy_epoch += dist_entropy.item()
                clip_frac_epoch += clip_frac
                approx_kl_epoch += approx_kl
                n_updates += 1

        # Fallback path: if minibatch mode was requested OR RCDC was inactive
        # within the main loop (e.g. num_mini_batch>1), apply RCDC as a
        # separate gradient step over the full batch. Keeps behaviour
        # consistent with the A2C implementation when num_mini_batch==1.
        if rcdc_active and num_mini_batch != 1:
            self.optimizer.zero_grad()
            diversity_loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
            self.optimizer.step()

        return {
            "policy_loss": policy_loss_epoch / max(n_updates, 1),
            "value_loss": value_loss_coef * value_loss_epoch / max(n_updates, 1),
            "dist_entropy": entropy_coef * entropy_epoch / max(n_updates, 1),
            "clip_frac": clip_frac_epoch / max(n_updates, 1),
            "approx_kl": approx_kl_epoch / max(n_updates, 1),
            # Stub fields kept for parity with A2C return dict / wandb panels
            "importance_sampling": 0.0,
            "seac_policy_loss": 0.0,
            "seac_value_loss": 0.0,
            "diversity_loss": diversity_loss_for_log,
            "state_div_loss": 0.0,
            "current_diversity": current_diversity,
        }
