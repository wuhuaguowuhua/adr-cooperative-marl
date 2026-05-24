# maa2c.py
# Multi-Agent Advantage Actor-Critic (MAA2C):
#   - Each agent has its own actor (independent policy networks).
#   - All agents share a single centralized critic V(s_global) that
#     receives the concatenation of all agents' observations.
#   - No SEAC-style cross-agent importance-weighted learning.
#   - RCDC diversity loss is integrated through SNDMonitor exactly as in A2C.
#
# Why MAA2C as the second algorithm (vs SEAC):
#   - SEAC and MAA2C share the A2C base but use *different* mechanisms to
#     exploit multi-agent structure: SEAC uses cross-agent experience
#     sharing; MAA2C uses a centralized value function. Demonstrating that
#     RCDC works on both yields a stronger cross-algorithm generalisation
#     claim than IPPO (which had no multi-agent buff at all and could not
#     learn the 4-agent task baseline).

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from gym.spaces.utils import flatdim

from model import Policy
from storage import RolloutStorage
from a2c import algorithm  # reuse the same Sacred Ingredient


class CentralCritic(nn.Module):
    """
    Centralized state-value critic V(s_global, agent_id) where s_global is
    the concatenation of every agent's local observation. The same network
    is shared across agents but produces N separate scalar values, one per
    agent, by emitting an N-dimensional output. This matches the EPyMARL
    CentralVCritic design (input=state+agent_id_one_hot -> per-agent V) and
    is essential for MARL tasks where the per-agent reward signals differ
    (e.g. RWARE, where only one agent typically receives r=1 on a delivery
    while the others receive 0). A single team-mean V cannot serve as a
    correct baseline in such per-agent-reward settings: it would punish
    agents that did nothing wrong simply because another agent collected a
    reward.
    """

    def __init__(self, total_obs_dim: int, n_agents: int, hidden_size: int = 64):
        super().__init__()
        self.total_obs_dim = total_obs_dim
        self.n_agents = n_agents
        self.net = nn.Sequential(
            nn.Linear(total_obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_agents),
        )
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, global_obs: torch.Tensor) -> torch.Tensor:
        """
        global_obs: (..., total_obs_dim) input tensor.
        returns:    (..., n_agents) per-agent V estimates.
        """
        return self.net(global_obs)


def build_global_obs(storages, time_slice=None) -> torch.Tensor:
    """
    Build the global observation tensor by concatenating each agent's
    storage.obs along the feature dim.

    Args:
        storages: list of RolloutStorage, one per agent.
        time_slice: slice or int. If None, use storages[i].obs (full
            length T+1). If slice, apply to the time dim.

    Returns:
        global_obs of shape (T_sliced, num_processes, sum_obs_dim) or
        (num_processes, sum_obs_dim) if time_slice is an int.
    """
    parts = []
    for s in storages:
        x = s.obs if time_slice is None else s.obs[time_slice]
        parts.append(x.float())
    return torch.cat(parts, dim=-1)


def central_critic_update(
    storages,
    central_critic: CentralCritic,
    central_optim: optim.Optimizer,
    value_loss_coef: float = 0.5,
    max_grad_norm: float = 0.5,
):
    """
    One MAA2C critic gradient step. The critic is trained to predict the
    *per-agent* discounted return R_i at each (state, time): the i-th
    output head V_i(s_global) regresses against agent i's own return,
    sharing all hidden parameters across agents. This is the standard
    EPyMARL/MAVEN central-V parameterisation and the only formulation
    that yields a correct advantage A_i = R_i - V_i in environments with
    per-agent rewards (e.g. RWARE delivery).

    Must be called BEFORE the actor updates so that all actors see the
    same V_i(s_global) snapshot when computing advantages.
    """
    global_obs = build_global_obs(storages, time_slice=slice(None, -1))
    v_all = central_critic(global_obs)  # (T, num_processes, n_agents)

    # Stack each agent's discounted returns along the last axis to produce
    # a per-agent target tensor of shape (T, num_processes, n_agents).
    per_agent_returns = torch.stack(
        [s.returns[:-1].squeeze(-1) for s in storages], dim=-1
    )

    value_loss = (per_agent_returns.detach() - v_all).pow(2).mean()

    central_optim.zero_grad()
    (value_loss_coef * value_loss).backward()
    nn.utils.clip_grad_norm_(central_critic.parameters(), max_grad_norm)
    central_optim.step()

    return float(value_loss.item())


class MAA2C:
    """
    Each MAA2C instance owns one agent's actor (and an unused per-agent
    critic head, kept only for `Policy.act()` API compatibility). The
    centralized critic is shared externally.
    """

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
        # is updated externally by `central_critic_update`. We still
        # update the entire `self.model` because Policy.base.critic_linear
        # produces dummy values consumed by storage.insert (those values
        # are overwritten downstream when MAA2C re-computes advantages
        # against the centralized critic, so this branch is effectively a
        # dead path; keeping it avoids invasive changes to Policy/Storage).
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
    def compute_returns(self, use_gae, gamma, gae_lambda, use_proper_time_limits):
        # Fallback path: invoked only if train.py forgets the maa2c
        # bootstrap. Uses per-agent critic for next_value, which is
        # technically inconsistent with MAA2C but harmless when
        # use_gae=False (returns ignore value_preds entirely).
        with torch.no_grad():
            next_value = self.model.get_value(
                self.storage.obs[-1],
                self.storage.recurrent_hidden_states[-1],
                self.storage.masks[-1],
            ).detach()
        self.storage.compute_returns(
            next_value, use_gae, gamma, gae_lambda, use_proper_time_limits,
        )

    @algorithm.capture
    def compute_returns_extrinsic(
        self, use_gae, gamma, gae_lambda, use_proper_time_limits
    ):
        # Kept for interface parity with A2C/SEAC. MAA2C does not use
        # extrinsic-only returns (no cross-agent learning).
        with torch.no_grad():
            next_value = self.model.get_value(
                self.storage.obs[-1],
                self.storage.recurrent_hidden_states[-1],
                self.storage.masks[-1],
            ).detach()
        self.storage.compute_returns_extrinsic(
            next_value, use_gae, gamma, gae_lambda, use_proper_time_limits,
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
        value_loss_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        device="cpu",
    ):
        """
        Actor-only update for one agent. The centralized critic must
        have already been updated in the same iteration via
        `central_critic_update` so that the V(s_global) snapshot used
        here matches what every agent sees.
        """
        assert central_critic is not None, (
            "MAA2C.update requires a `central_critic` instance."
        )

        obs_shape = self.storage.obs.size()[2:]
        action_shape = self.storage.actions.size()[-1]
        num_steps, num_processes, _ = self.storage.rewards.size()

        # 1. Re-evaluate per-agent actor on its own trajectory.
        _, action_log_probs, dist_entropy, _ = self.model.evaluate_actions(
            self.storage.obs[:-1].view(-1, *obs_shape),
            self.storage.recurrent_hidden_states[0].view(
                -1, self.model.recurrent_hidden_state_size
            ),
            self.storage.masks[:-1].view(-1, 1),
            self.storage.actions.view(-1, action_shape),
        )
        action_log_probs = action_log_probs.view(num_steps, num_processes, 1)

        # 2. Centralized advantage: A_i = R_i - V_i(s_global). The critic
        # is detached because it was already updated this iteration; the
        # actor only consumes its scalar advantage signal. We slice out
        # the i-th head to get this agent's own baseline; using the team
        # mean here would be wrong on per-agent-reward tasks.
        # Note: storage.returns[:-1] is the GAE-augmented per-agent return
        # which was itself computed against V_i(s_global) by train.py
        # (see the rollout-boundary block that overwrites every storage's
        # value_preds with the central V), so the baseline used here and
        # the baseline used inside GAE are guaranteed to be the same V_i.
        with torch.no_grad():
            global_obs = build_global_obs(storages, time_slice=slice(None, -1))
            v_all = central_critic(global_obs)  # (T, N_proc, n_agents)
            v_self = v_all[..., self.agent_id : self.agent_id + 1]  # (T, N_proc, 1)
        advantages = (self.storage.returns[:-1] - v_self).detach()

        policy_loss = -(advantages * action_log_probs).mean()

        # 3. RCDC diversity loss (identical to A2C path).
        diversity_loss = torch.tensor(
            0.0, device=device if isinstance(device, str) else device
        )
        current_diversity = 0.0
        if snd_monitor is not None and models is not None:
            use_grad = snd_monitor.needs_grad(progress)
            batch_obs = self.storage.obs[:-1].view(-1, *obs_shape)
            batch_size = batch_obs.size(0)
            ctx = torch.enable_grad() if use_grad else torch.no_grad()
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

        # 4. State-Visitation Diversity (parity with A2C; usually
        # disabled in our experiments).
        state_div_loss = self.storage.obs.new_zeros(())
        if state_div_monitor is not None and state_div_monitor.needs_grad(progress):
            state_div_obs = self.storage.obs[:-1].view(-1, *obs_shape)
            state_div_loss = state_div_monitor.compute_state_div_loss(
                agent_id=self.agent_id,
                obs_batch=state_div_obs,
                action_log_probs=action_log_probs.view(-1, 1),
                progress=progress,
            )

        # 5. Actor backward + step.
        self.optimizer.zero_grad()
        actor_loss = (
            policy_loss
            - entropy_coef * dist_entropy
            + diversity_loss
            + state_div_loss
        )
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
        self.optimizer.step()

        return {
            "policy_loss": policy_loss.item(),
            # Report the per-agent TD/MC value loss against V_i(s_global)
            # so the W&B "value_loss" panel reflects how well the central
            # critic predicts agent i's own return. The actual gradient
            # step on the critic is taken in central_critic_update, this
            # is a logging-only quantity.
            "value_loss": float(advantages.pow(2).mean().item()) * value_loss_coef,
            "dist_entropy": entropy_coef * dist_entropy.item(),
            # MAA2C has no SEAC terms; keep stub fields for log parity.
            "importance_sampling": 0.0,
            "seac_policy_loss": 0.0,
            "seac_value_loss": 0.0,
            "diversity_loss": (
                float(diversity_loss)
                if isinstance(diversity_loss, (int, float))
                else diversity_loss.item()
            ),
            "state_div_loss": (
                float(state_div_loss)
                if isinstance(state_div_loss, (int, float))
                else state_div_loss.item()
            ),
            "current_diversity": current_diversity,
        }
