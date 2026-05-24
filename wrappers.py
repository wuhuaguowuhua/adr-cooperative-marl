
import math
from collections import deque
from time import perf_counter

import gym
import numpy as np
from gym import ObservationWrapper, spaces
from gym.wrappers import TimeLimit as GymTimeLimit
from gym.wrappers import Monitor as GymMonitor


class RecordEpisodeStatistics(gym.Wrapper):
    """ Multi-agent version of RecordEpisodeStatistics gym wrapper"""

    def __init__(self, env, deque_size=100):
        super().__init__(env)
        self.t0 = perf_counter()
        self.episode_reward = np.zeros(self.n_agents)
        self.episode_length = 0
        self.reward_queue = deque(maxlen=deque_size)
        self.length_queue = deque(maxlen=deque_size)

    def reset(self, **kwargs):
        observation = super().reset(**kwargs)
        self.episode_reward = 0
        self.episode_length = 0
        self.t0 = perf_counter()

        return observation

    def step(self, action):
        observation, reward, done, info = super().step(action)
        self.episode_reward += np.array(reward, dtype=np.float64)
        self.episode_length += 1
        if all(done):
            info["episode_reward"] = self.episode_reward
            for i, agent_reward in enumerate(self.episode_reward):
                info[f"agent{i}/episode_reward"] = agent_reward
            info["episode_length"] = self.episode_length
            info["episode_time"] = perf_counter() - self.t0

            self.reward_queue.append(self.episode_reward)
            self.length_queue.append(self.episode_length)
        return observation, reward, done, info


class FlattenObservation(ObservationWrapper):
    r"""Observation wrapper that flattens the observation of individual agents."""

    def __init__(self, env):
        super(FlattenObservation, self).__init__(env)

        ma_spaces = []

        for sa_obs in env.observation_space:
            flatdim = spaces.flatdim(sa_obs)
            ma_spaces += [
                spaces.Box(
                    low=-float("inf"),
                    high=float("inf"),
                    shape=(flatdim,),
                    dtype=np.float32,
                )
            ]

        self.observation_space = spaces.Tuple(tuple(ma_spaces))

    def observation(self, observation):
        return tuple([
            spaces.flatten(obs_space, obs)
            for obs_space, obs in zip(self.env.observation_space, observation)
        ])


class SquashDones(gym.Wrapper):
    r"""Wrapper that squashes multiple dones to a single one using all(dones)"""

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        return observation, reward, all(done), info


class GlobalizeReward(gym.RewardWrapper):
    def reward(self, reward):
        return self.n_agents * [sum(reward)]


class TimeLimit(GymTimeLimit):
    def __init__(self, env, max_episode_steps=None):
        super().__init__(env)
        if max_episode_steps is None and self.env.spec is not None:
            max_episode_steps = env.spec.max_episode_steps
        # if self.env.spec is not None:
        #     self.env.spec.max_episode_steps = max_episode_steps
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = None

    def step(self, action):
        assert self._elapsed_steps is not None, "Cannot call env.step() before calling reset()"
        observation, reward, done, info = self.env.step(action)
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            info['TimeLimit.truncated'] = not all(done)
            done = len(observation) * [True]
        return observation, reward, done, info

class ClearInfo(gym.Wrapper):
    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        return observation, reward, done, {}


class PressurePlateRewardShaper(gym.Wrapper):
    """v4: potential-based distance + chain-stage entry bonuses.

    DIAGNOSIS OF v3 FAILURE (stuck at 50.000):
      Only agent 0 starts in its correct room: all four agents spawn in
      room 0, but agents 1/2/3 are assigned to rooms 1/2/3 respectively.
      The v3 in_room_reward (+0.1/step) was therefore visible only to
      agent 0, which converged on the trivial policy "stand still in
      starting room"  ==>  0.1 * 500 = 50.000 per episode, deterministic,
      no exploration.  Agents 1/2/3 had identically zero learning signal,
      and RCDC's diversity gradient further destabilised the only agent
      (agent 0) that had any signal to learn from.

    v4 strategy:
      (i)   remove the in_room_reward (eliminates the 50.000 attractor),
      (ii)  install a potential-based Manhattan-distance shaping per agent
            (telescoping --> preserves the cooperative optimum),
      (iii) add a chain_entry_bonus the first time *any* agent enters
            room k (for k=1,2,3) --- this rewards downstream unlocking
            so agents 1/2/3 get a strong, attributable upstream-progress
            signal even before they reach their own plates.

    Per-agent reward components (v4):
      - shaped(t) = k_dist * (d_prev - d_curr)         (k_dist = 0.5)
            where d = Manhattan distance from agent to its target cell
            (plate i for i < N-1, goal for i = N-1).  Telescoping.
      - on_plate_reward   = +1.0 / step on target
      - entry_bonus       = +30.0 one-shot, first time agent reaches plate
      - chain_entry_bonus = +30.0 one-shot (TEAM reward), first time any
                            agent enters room k for k=1,2,3
      - terminal_bonus    = +100.0 per agent on goal achievement
                            (NOT on TimeLimit truncation)

    Predicted scale (smoke test target):
      - random episode total      ~ 0..50
      - solved episode total      ~ 1200..1500
      - signal-to-noise ratio     ~ 30x   (clean signal vs. v3's degenerate
                                           50.000 attractor)

    No-op for non-PressurePlate envs.
    """

    def __init__(self, env,
                 on_plate_reward=1.0,
                 entry_bonus=30.0,
                 terminal_bonus=100.0,
                 chain_entry_bonus=30.0,
                 distance_coef=0.5):
        super().__init__(env)
        self.on_plate_reward = float(on_plate_reward)
        self.entry_bonus = float(entry_bonus)
        self.terminal_bonus = float(terminal_bonus)
        self.chain_entry_bonus = float(chain_entry_bonus)
        self.distance_coef = float(distance_coef)

        spec_id = ""
        try:
            spec_id = (env.spec.id if env.spec is not None else "") or ""
        except Exception:
            pass
        self._active = spec_id.lower().startswith("pressureplate")

        self._on_plate_seen = None
        self._prev_dist = None
        self._room_seen = None  # room k entry flag for k=1..N-1

    def _target_xy(self, i, n):
        u = self.unwrapped
        if i == n - 1:
            return u.goal.x, u.goal.y
        return u.plates[i].x, u.plates[i].y

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        if self._active:
            u = self.unwrapped
            n = u.n_agents
            self._on_plate_seen = [False] * n
            self._prev_dist = [
                abs(u.agents[i].x - self._target_xy(i, n)[0])
                + abs(u.agents[i].y - self._target_xy(i, n)[1])
                for i in range(n)
            ]
            # Chain-entry flag: index k corresponds to "any agent first in room k".
            self._room_seen = [False] * n
            # Mark the starting room (typically room 0) as already seen so we
            # never pay a chain bonus for staying put.
            start_room = u._get_curr_room_reward(u.agents[0].y)
            if 0 <= start_room < n:
                self._room_seen[start_room] = True
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if not self._active:
            return obs, reward, done, info

        u = self.unwrapped
        n = u.n_agents
        new_reward = [0.0] * n
        if self._on_plate_seen is None:
            self._on_plate_seen = [False] * n
        if self._prev_dist is None:
            self._prev_dist = [0.0] * n
        if self._room_seen is None:
            self._room_seen = [False] * n

        for i, agent in enumerate(u.agents):
            tx, ty = self._target_xy(i, n)

            d_curr = abs(agent.x - tx) + abs(agent.y - ty)
            # Potential-based Manhattan shaping: + as agent gets closer.
            new_reward[i] += self.distance_coef * (self._prev_dist[i] - d_curr)
            self._prev_dist[i] = d_curr

            on_target = (agent.x == tx and agent.y == ty)
            if on_target:
                new_reward[i] += self.on_plate_reward
                if not self._on_plate_seen[i]:
                    new_reward[i] += self.entry_bonus
                    self._on_plate_seen[i] = True

            # Chain-entry team bonus: any agent newly entering room k
            # (k=1..N-1) earns the bonus shared by all agents.
            curr_room = u._get_curr_room_reward(agent.y)
            if 0 < curr_room < n and not self._room_seen[curr_room]:
                self._room_seen[curr_room] = True
                for j in range(n):
                    new_reward[j] += self.chain_entry_bonus

        # Terminal success bonus on goal achievement (not timeout).
        if all(done) and not info.get("TimeLimit.truncated", False):
            new_reward = [r + self.terminal_bonus for r in new_reward]

        return obs, new_reward, done, info


class Monitor(GymMonitor):
    def _after_step(self, observation, reward, done, info):
        if not self.enabled: return done

        if all(done) and self.env_semantics_autoreset:
            # For envs with BlockingReset wrapping VNCEnv, this observation will be the first one of the new episode
            self.reset_video_recorder()
            self.episode_id += 1
            self._flush()

        # Record stats
        self.stats_recorder.after_step(observation, sum(reward), all(done), info)
        # Record video
        self.video_recorder.capture_frame()

        return done
