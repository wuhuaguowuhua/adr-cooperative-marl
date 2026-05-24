# train.py
import glob
import logging
import os
import shutil
import time
from collections import deque
from os import path
from pathlib import Path

import numpy as np
import torch
from sacred import Experiment
from sacred.observers import (  # noqa
    FileStorageObserver,
    MongoObserver,
    QueuedMongoObserver,
    QueueObserver,
)
from torch.utils.tensorboard import SummaryWriter

import utils
from a2c import A2C, algorithm
from ippo import IPPO
from maa2c import MAA2C, CentralCritic, central_critic_update, build_global_obs
from mappo import MAPPO
from envs import make_vec_envs
from wrappers import RecordEpisodeStatistics, SquashDones, PressurePlateRewardShaper
from model import Policy
from snd_monitor import SNDMonitor
from state_diversity import CountBasedExplorer
from state_div import StateDivMonitor

try:
    import robotic_warehouse  # noqa: F401
except ImportError:
    import rware as robotic_warehouse  # noqa: F401
import lbforaging  # noqa
try:
    import pressureplate  # noqa: F401  # registers pressureplate-linear-{4,5,6}p-v0
except ImportError:
    pass
from typing import Optional
import torch.nn.functional as F

# --- W&B x Sacred ---
try:
    from wandb.integration.sacred import WandbObserver
    _WANDB_OK = True
except Exception:
    try:
        from wandb.sacred import WandbObserver
        _WANDB_OK = True
    except Exception:
        _WANDB_OK = False


def wb_log(data: dict, step: Optional[int] = None) -> None:
    try:
        import wandb
        if wandb.run is not None:
            if step is None:
                wandb.log(data)
            else:
                wandb.log(data, step=step)
    except Exception:
        pass


def _headless() -> bool:
    no_display = os.environ.get("DISPLAY", "") == ""
    pyglet_headless = os.environ.get("PYGLET_HEADLESS", "").lower() in ("1", "true")
    return no_display and not pyglet_headless


ex = Experiment(ingredients=[algorithm])
ex.captured_out_filter = lambda captured_output: "Output capturing turned off."
ex.observers.append(FileStorageObserver("./results/sacred"))

if _WANDB_OK and os.getenv("WANDB_DISABLED", "false").lower() not in ("true", "1"):
    ex.observers.append(
        WandbObserver(
            project=os.getenv("WANDB_PROJECT", "seac-rware"),
            entity=os.getenv("WANDB_ENTITY") or None,
        )
    )

logging.basicConfig(
    level=logging.INFO,
    format="(%(process)d) [%(levelname).1s] - (%(asctime)s) - %(name)s >> %(message)s",
    datefmt="%m/%d %H:%M:%S",
)


@ex.config
def config():
    env_name = None
    time_limit = None
    # NOTE: PressurePlateRewardShaper is inserted BEFORE RecordEpisodeStatistics
    # (so the shaped, positive reward is what we log per-episode) but AFTER
    # TimeLimit (so info['TimeLimit.truncated'] is set when the wrapper
    # decides whether to grant the terminal bonus). The TimeLimit wrapper is
    # applied separately in envs.py:make_env() and runs *before* this list.
    wrappers = (PressurePlateRewardShaper, RecordEpisodeStatistics, SquashDones)
    dummy_vecenv = False

    num_env_steps = 100e6

    eval_dir = None
    loss_dir = None
    save_dir = "./results/trained_models/{id}"

    log_interval = 2000
    save_interval = int(5e6)
    eval_interval = int(5e6)
    episodes_per_eval = 8

    use_wandb = True
    wandb_run_name = os.getenv("WANDB_RUN_NAME", None)
    wandb_tags = []
    wandb_project = None


for conf in glob.glob("configs/*.yaml"):
    name = f"{Path(conf).stem}"
    ex.add_named_config(name, conf)


def _squash_info(info):
    info = [i for i in info if i]
    new_info = {}
    keys = set([k for i in info for k in i.keys()])
    keys.discard("TimeLimit.truncated")
    for key in keys:
        mean = np.mean([np.array(d[key]).sum() for d in info if key in d])
        new_info[key] = mean
    return new_info


@ex.capture
def evaluate(
    agents,
    monitor_dir,
    episodes_per_eval,
    env_name,
    seed,
    wrappers,
    dummy_vecenv,
    time_limit,
    algorithm,
    _log,
):
    device = algorithm["device"]

    if monitor_dir and (
        _headless() or os.getenv("SEAC_DISABLE_VIDEO", "").lower() in ("1", "true")
    ):
        _log.warning("Headless/SEAC_DISABLE_VIDEO=1 -> skip video recording.")
        monitor_dir = None

    eval_envs = make_vec_envs(
        env_name,
        seed,
        dummy_vecenv,
        episodes_per_eval,
        time_limit,
        wrappers,
        device,
        monitor_dir=monitor_dir,
    )

    n_obs = eval_envs.reset()
    n_recurrent_hidden_states = [
        torch.zeros(
            episodes_per_eval, agent.model.recurrent_hidden_state_size, device=device
        )
        for agent in agents
    ]
    masks = torch.zeros(episodes_per_eval, 1, device=device)

    all_infos = []

    while len(all_infos) < episodes_per_eval:
        with torch.no_grad():
            _, n_action, _, n_recurrent_hidden_states = zip(
                *[
                    agent.model.act(
                        n_obs[agent.agent_id].float(), rhs, masks
                    )
                    for agent, rhs in zip(agents, n_recurrent_hidden_states)
                ]
            )
            actions_env = list(n_action)

        n_obs, _, done, infos = eval_envs.step(actions_env)

        masks = torch.tensor(
            [[0.0] if done_ else [1.0] for done_ in done],
            dtype=torch.float32,
            device=device,
        )
        all_infos.extend([i for i in infos if i])

    eval_envs.close()
    info = _squash_info(all_infos)
    _log.info(
        f"Evaluation using {len(all_infos)} episodes: mean reward {info.get('episode_reward', float('nan')):.5f}\n"
    )
    return info


@ex.automain
def main(
    _run,
    _log,
    num_env_steps,
    env_name,
    seed,
    algorithm,
    dummy_vecenv,
    time_limit,
    wrappers,
    save_dir,
    eval_dir,
    loss_dir,
    log_interval,
    save_interval,
    eval_interval,
    use_wandb,
    wandb_run_name,
    wandb_tags,
    wandb_project,
):

    if loss_dir:
        loss_dir = path.expanduser(loss_dir.format(id=str(_run._id)))
        utils.cleanup_log_dir(loss_dir)
        writer = SummaryWriter(loss_dir)
    else:
        writer = None

    # --- SNDMonitor: Reward-Coupled Diversity Control (RCDC) ---
    snd_enable = os.getenv("SND_ENABLE", "0").lower() in ("1", "true")
    if snd_enable:
        snd_monitor = SNDMonitor(
            metric=os.getenv("SND_METRIC", "l2"),
            tau=float(os.getenv("SND_TAU", "0.01")),
            global_coef=float(os.getenv("SND_DIVERSITY_COEF", "0.1")),
            loss_mode=os.getenv("SND_LOSS_MODE", "rcdc"),
            trigger=os.getenv("SND_TRIGGER", "rising"),
            warmup_ratio=float(os.getenv("SND_WARMUP_RATIO", "0.15")),
            tau_fast=float(os.getenv("SND_TAU_FAST", "0.1")),
            tau_slow=float(os.getenv("SND_TAU_SLOW", "0.02")),
            sensitivity=float(os.getenv("SND_SENSITIVITY", "5.0")),
            momentum_beta=float(os.getenv("SND_MOMENTUM_BETA", "0.95")),
            activation_threshold=float(os.getenv("SND_ACTIVATION_THRESHOLD", "0.2")),
            reward_level_threshold=float(os.getenv("SND_REWARD_LEVEL_THRESHOLD", "0.5")),
            diversity_ceiling=float(os.getenv("SND_DIVERSITY_CEILING", "0.5")),
            min_maturity=int(os.getenv("SND_MIN_MATURITY", "50")),
            proactive_shutoff=int(os.getenv("SND_PROACTIVE_SHUTOFF", "10")),
            proactive_ramp_end=float(os.getenv("SND_PROACTIVE_RAMP_END", "0.5")),
            proactive_reward_threshold=float(os.getenv("SND_PROACTIVE_REWARD_THRESH", "0.01")),
            proactive_ramp_power=float(os.getenv("SND_PROACTIVE_RAMP_POWER", "1.0")),
            proactive_time_shutoff=float(os.getenv("SND_PROACTIVE_TIME_SHUTOFF", "1.0")),
            eta_max=float(os.getenv("SND_ETA_MAX", "1.0")),
            feedback_gain=float(os.getenv("SND_FEEDBACK_GAIN", "0.05")),
            feedback_trend_threshold=float(os.getenv("SND_FEEDBACK_TREND_THRESH", "0.001")),
            feedback_init_direction=float(os.getenv("SND_FEEDBACK_INIT_DIRECTION", "1.0")),
            feedback_decel_threshold=float(os.getenv("SND_FEEDBACK_DECEL_THRESH", "0.5")),
            feedback_min_peak=float(os.getenv("SND_FEEDBACK_MIN_PEAK", "0.02")),
            feedback_min_interval=int(os.getenv("SND_FEEDBACK_MIN_INTERVAL", "50")),
            feedback_min_progress=float(os.getenv("SND_FEEDBACK_MIN_PROGRESS", "0.05")),
        )
    else:
        snd_monitor = SNDMonitor(
            metric=os.getenv("SND_METRIC", "l2"),
            loss_mode="monitor",
        )

    # --- Count-Based Exploration: deferred until n_agents known ---
    count_explore_enable = os.getenv("COUNT_EXPLORE_ENABLE", "0").lower() in ("1", "true")
    count_explorer = None  # created after agents

    # --- State-Visitation Diversity: deferred until n_agents known ---
    state_div_enable = os.getenv("STATE_DIV_ENABLE", "0").lower() in ("1", "true")
    state_div_monitor = None  # created after agents

    if eval_dir:
        eval_dir = path.expanduser(eval_dir.format(id=str(_run._id)))
    if save_dir:
        save_dir = path.expanduser(save_dir.format(id=str(_run._id)))

    if eval_dir:
        utils.cleanup_log_dir(eval_dir)
    if save_dir:
        utils.cleanup_log_dir(save_dir)

    torch.set_num_threads(1)
    envs = make_vec_envs(
        env_name,
        seed,
        dummy_vecenv,
        algorithm["num_processes"],
        time_limit,
        wrappers,
        algorithm["device"],
    )

    # ---------- W&B runtime info ----------
    _W_ON = (
        _WANDB_OK
        and use_wandb
        and os.getenv("WANDB_DISABLED", "false").lower() not in ("true", "1")
    )
    if _W_ON:
        try:
            import wandb
            if wandb.run is not None:
                if wandb_run_name:
                    wandb.run.name = wandb_run_name
                    try:
                        wandb.run.save()
                    except Exception:
                        pass
                wandb_group = os.getenv("WANDB_RUN_GROUP")
                if wandb_group:
                    wandb.run.config.update(
                        {"experiment_group": wandb_group}, allow_val_change=True
                    )
                all_tags = list(wandb.run.tags or [])
                if wandb_group:
                    all_tags.append(wandb_group)
                if wandb_tags:
                    all_tags.extend(list(wandb_tags))
                if all_tags:
                    wandb.run.tags = list(set(all_tags))
        except Exception as e:
            _log.warning(f"[W&B] runtime decoration skipped: {e}")

    algo_name = os.getenv("ALGO", "a2c").lower()
    if algo_name == "ippo":
        AgentCls = IPPO
        _log.info(f"[ALGO] Using IPPO (Independent PPO)")
    elif algo_name == "maa2c":
        AgentCls = MAA2C
        _log.info(f"[ALGO] Using MAA2C (Multi-Agent A2C with centralized critic)")
    elif algo_name == "mappo":
        AgentCls = MAPPO
        _log.info(f"[ALGO] Using MAPPO (Multi-Agent PPO with centralized critic)")
    else:
        AgentCls = A2C
        _log.info(f"[ALGO] Using A2C/SEAC (default)")

    # MAPPO and MAA2C share the same centralized-critic plumbing in the
    # rollout / GAE / critic-update path; treat them uniformly below.
    uses_central_critic = algo_name in ("maa2c", "mappo")

    agents = [
        AgentCls(i, osp, asp)
        for i, (osp, asp) in enumerate(zip(envs.observation_space, envs.action_space))
    ]

    # --- MAA2C / MAPPO: build the shared centralized critic over
    #     concatenated obs. Both algorithms reuse the same critic
    #     architecture and external `central_critic_update` (one
    #     gradient step per rollout, taken BEFORE the actor updates).
    central_critic = None
    central_optim = None
    if uses_central_critic:
        total_obs_dim = sum(
            int(np.prod(osp.shape)) for osp in envs.observation_space
        )
        hidden_size = int(os.getenv("MAA2C_CRITIC_HIDDEN", "64"))
        n_agents_central = len(agents)
        central_critic = CentralCritic(
            total_obs_dim,
            n_agents=n_agents_central,
            hidden_size=hidden_size,
        )
        central_critic.to(algorithm["device"])
        central_optim = torch.optim.Adam(
            central_critic.parameters(),
            lr=float(os.getenv("MAA2C_CRITIC_LR", str(algorithm["lr"]))),
            eps=algorithm["adam_eps"],
        )
        _log.info(
            f"[{algo_name.upper()}] CentralCritic built: "
            f"total_obs_dim={total_obs_dim}, "
            f"n_agents={n_agents_central}, hidden={hidden_size}"
        )

    if count_explore_enable:
        count_explorer = CountBasedExplorer(
            n_agents=len(agents),
            coef=float(os.getenv("COUNT_EXPLORE_COEF", "0.01")),
            hash_dim=int(os.getenv("COUNT_EXPLORE_HASH_DIM", "16")),
            decay_factor=float(os.getenv("COUNT_EXPLORE_DECAY", "1.0")),
            decay_interval=int(os.getenv("COUNT_EXPLORE_DECAY_INTERVAL", "50000")),
            warmup_ratio=float(os.getenv("COUNT_EXPLORE_WARMUP", "0.0")),
            anneal_end_ratio=float(os.getenv("COUNT_EXPLORE_ANNEAL_END", "1.0")),
        )

    if state_div_enable:
        state_div_monitor = StateDivMonitor(
            n_agents=len(agents),
            hash_dim=int(os.getenv("STATE_DIV_HASH_DIM", "12")),
            coef=float(os.getenv("STATE_DIV_COEF", "0.1")),
            schedule_mode=os.getenv("STATE_DIV_SCHEDULE", "reward_gated"),
            warmup_ratio=float(os.getenv("STATE_DIV_WARMUP", "0.10")),
            ramp_peak_ratio=float(os.getenv("STATE_DIV_RAMP_PEAK", "0.30")),
            ramp_end_ratio=float(os.getenv("STATE_DIV_RAMP_END", "0.60")),
            reward_threshold=float(os.getenv("STATE_DIV_REWARD_THRESH", "2.0")),
            shutoff_streak=int(os.getenv("STATE_DIV_SHUTOFF", "10")),
            eta_max=float(os.getenv("STATE_DIV_ETA_MAX", "0.10")),
            center_baseline=os.getenv("STATE_DIV_CENTER", "1") == "1",
            seed=seed,
        )

    obs = envs.reset()

    for i in range(len(obs)):
        agents[i].storage.obs[0].copy_(obs[i])
        agents[i].storage.to(algorithm["device"])

    start = time.time()
    num_updates = (
        int(num_env_steps) // algorithm["num_steps"] // algorithm["num_processes"]
    )

    all_infos = deque(maxlen=10)
    steps_per_update = algorithm["num_steps"] * algorithm["num_processes"]

    for j in range(1, num_updates + 1):

        progress = j / num_updates

        for step in range(algorithm["num_steps"]):
            with torch.no_grad():
                n_value, n_action, n_action_log_prob, n_recurrent_hidden_states = zip(
                    *[
                        agent.model.act(
                            agent.storage.obs[step].float(),
                            agent.storage.recurrent_hidden_states[step],
                            agent.storage.masks[step],
                        )
                        for agent in agents
                    ]
                )
                actions_env = list(n_action)

            n_obs, reward, done, infos = envs.step(actions_env)
            obs = n_obs

            masks = torch.tensor(
                [[0.0] if done_ else [1.0] for done_ in done],
                dtype=torch.float32, device=algorithm["device"]
            )
            bad_masks = torch.tensor(
                [[0.0] if info.get("TimeLimit.truncated", False) else [1.0] for info in infos],
                dtype=torch.float32, device=algorithm["device"]
            )

            # Compute count-based exploration intrinsic reward
            if count_explorer is not None:
                explore_bonuses = count_explorer.compute_per_agent_bonus(
                    [obs[i] for i in range(len(agents))],
                    progress=progress,
                )
            else:
                explore_bonuses = None

            # Update StateDiv visitation counts
            if state_div_monitor is not None:
                state_div_monitor.update_visits([obs[i] for i in range(len(agents))])

            for i in range(len(agents)):
                buf = agents[i].storage.actions[step]
                K = buf.size(-1)
                if K == 1:
                    act_to_store = actions_env[i]
                    if act_to_store.dtype != buf.dtype:
                        act_to_store = act_to_store.to(buf.dtype)
                else:
                    act_val = actions_env[i]
                    if act_val.dim() > 0:
                        act_val = act_val.squeeze()
                    act_to_store = F.one_hot(act_val.long(), num_classes=K).to(buf.dtype)

                agent_reward_ext = reward[:, i].unsqueeze(1)
                agent_reward = agent_reward_ext
                if explore_bonuses is not None:
                    agent_reward = agent_reward_ext + explore_bonuses[i]

                agents[i].storage.insert(
                    obs[i],
                    n_recurrent_hidden_states[i],
                    act_to_store,
                    n_action_log_prob[i],
                    n_value[i],
                    agent_reward,
                    masks,
                    bad_masks,
                    rewards_extrinsic=agent_reward_ext,
                )

            for info in infos:
                if info:
                    all_infos.append(info)

        if uses_central_critic:
            # Overwrite EVERY agent's storage.value_preds with the
            # centralized critic's V_i(s_global, t). This is essential
            # because storage.compute_returns uses self.value_preds at
            # every intermediate timestep when use_gae=True (delta_t =
            # r_t + gamma * V_{t+1} - V_t). Without this overwrite the
            # GAE delta would use each agent's local critic head, which
            # in MAA2C is never trained (the actor-only update touches
            # only the policy branch). That would turn returns into a
            # high-variance MC estimate against an untrained baseline,
            # and the central critic would be regressed against those
            # corrupted targets, producing the reward=0 collapse we saw
            # in the v1 sanity. After this overwrite, GAE uses a
            # consistent V_i across t = 0..T and the central critic is
            # the only baseline anywhere in the pipeline.
            with torch.no_grad():
                global_obs_full = build_global_obs(
                    [a.storage for a in agents]
                )  # (T+1, N_proc, sum_obs_dim)
                v_all_full = central_critic(global_obs_full).detach()
                # (T+1, N_proc, n_agents)
                for i, a in enumerate(agents):
                    a.storage.value_preds.copy_(v_all_full[..., i : i + 1])
            for i, agent in enumerate(agents):
                next_value_i = v_all_full[-1, ..., i : i + 1]
                agent.storage.compute_returns(
                    next_value_i,
                    algorithm["use_gae"],
                    algorithm["gamma"],
                    algorithm["gae_lambda"],
                    algorithm["use_proper_time_limits"],
                )
        else:
            for agent in agents:
                agent.compute_returns()
                agent.compute_returns_extrinsic()

        # MAA2C / MAPPO critic update must precede actor updates so all
        # actors see the same (post-update) V(s_global) snapshot when
        # computing advantages. The critic gradient step is the same
        # across both algorithms; only the actor update differs.
        if uses_central_critic:
            critic_loss_val = central_critic_update(
                storages=[a.storage for a in agents],
                central_critic=central_critic,
                central_optim=central_optim,
                value_loss_coef=algorithm["value_loss_coef"],
                max_grad_norm=algorithm["max_grad_norm"],
            )
            if writer:
                writer.add_scalar(
                    f"{algo_name}/critic_loss", critic_loss_val, j,
                )

        for agent in agents:
            update_kwargs = dict(
                snd_monitor=snd_monitor,
                state_div_monitor=state_div_monitor,
                progress=progress,
            )
            if uses_central_critic:
                update_kwargs["central_critic"] = central_critic
            loss = agent.update(
                [a.storage for a in agents],
                [a.model for a in agents],
                **update_kwargs,
            )
            total_num_steps = j * steps_per_update
            for k, v in loss.items():
                if writer:
                    writer.add_scalar(f"agent{agent.agent_id}/{k}", v, j)

        if snd_monitor is not None:
            stats = snd_monitor.log_stats()
            wb_log(stats, step=j * steps_per_update)

        if count_explorer is not None:
            ce_stats = count_explorer.log_stats()
            wb_log(ce_stats, step=j * steps_per_update)

        if state_div_monitor is not None:
            sd_stats = state_div_monitor.log_stats()
            wb_log(sd_stats, step=j * steps_per_update)

        for agent in agents:
            agent.storage.after_update()

        if j % log_interval == 0 and len(all_infos) > 1:
            squashed = _squash_info(all_infos)
            total_num_steps = j * steps_per_update
            end = time.time()
            logging.getLogger(__name__).info(
                f"Updates {j}, num timesteps {total_num_steps}, FPS {int(total_num_steps / (end - start))}"
            )
            logging.getLogger(__name__).info(
                f"Last {len(all_infos)} training episodes mean reward {float(squashed.get('episode_reward', 0)):.3f}"
            )
            if "episode_length" in squashed:
                logging.getLogger(__name__).info(
                    f"Mean episode length: {float(squashed['episode_length']):.1f}"
                )
            wb_log({"returns": float(squashed.get("episode_reward", float("nan")))}, step=total_num_steps)

            if snd_monitor is not None:
                snd_monitor.update_reward(float(squashed.get("episode_reward", 0)))

            if state_div_monitor is not None:
                state_div_monitor.update_reward(float(squashed.get("episode_reward", 0)))

            for k, v in squashed.items():
                _run.log_scalar(k, v, j)
                if writer:
                    writer.add_scalar(f"train/{k}", float(v), total_num_steps)
                if k == "episode_length":
                    wb_log({"episode_length": float(v)}, step=total_num_steps)

            if writer:
                writer.add_scalar("train/fps", int(total_num_steps / (end - start)), total_num_steps)
                writer.flush()
            all_infos.clear()

        if save_interval is not None and save_dir and (j % save_interval == 0 or j == num_updates):
            cur_save_dir = path.join(save_dir, f"u{j}")
            for agent in agents:
                save_at = path.join(cur_save_dir, f"agent{agent.agent_id}")
                os.makedirs(save_at, exist_ok=True)
                agent.save(save_at)
            archive_name = shutil.make_archive(cur_save_dir, "xztar", save_dir, f"u{j}")
            shutil.rmtree(cur_save_dir)
            _run.add_artifact(archive_name)

        if eval_interval is not None and (j % eval_interval == 0 or j == num_updates):
            monitor_dir = None if not eval_dir else os.path.join(eval_dir, f"u{j}")
            info = evaluate(agents, monitor_dir)
            total_num_steps = j * steps_per_update
            if info and "episode_reward" in info:
                wb_log({"eval/returns": float(info["episode_reward"])})
            if writer and info and "episode_reward" in info:
                writer.add_scalar("eval/mean_reward", float(info["episode_reward"]), total_num_steps)
            if eval_dir:
                videos = glob.glob(os.path.join(eval_dir, f"u{j}") + "/*.mp4")
                for i, v in enumerate(videos):
                    _run.add_artifact(v, f"u{j}.{i}.mp4")

    envs.close()
    if writer:
        writer.close()
