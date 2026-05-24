
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
    """v3: sparse, event-driven, strictly positive rewards for PressurePlate.

    v1 (raw rewards in [-3, 0]): both baseline and RCDC stuck at -4640 for
       30M steps. Diagnosis: feedback gate breaks on negative reward
       normalisation; baseline can't escape "wrong-room" -3 noise floor.

    v2 (uniform +3 shift): random ep total = +1387, after 13M still +1370.
       Diagnosis: the shift introduces a HUGE constant baseline (1387) that
       only changes by ~10% when fully learned (1500). Signal-to-noise too
       low; SEAC's advantage estimate is dominated by the value-function
       baseline and yields a near-zero gradient.

    v3 (this) replaces the dense shaping with an event-driven sparse signal:
      - +on_plate_reward (default +1.0) per step ONLY if agent is ON their
        assigned plate. For the last agent the "plate" is the goal cell.
      - +in_room_reward (default +0.1) per step if agent is in their correct
        room but NOT on the plate. This gives wrong-room agents a weak but
        non-zero gradient the moment a door opens and they cross over.
      - +entry_bonus (default +10) the first time each agent reaches their
        plate in an episode (one-shot, encourages exploration that finds
        the plate at least once).
      - +terminal_bonus (default +50) per agent on goal achievement (NOT on
        TimeLimit truncation).

    Predicted scale:
      - random policy episode total ~ +110 (agent 0 occasionally on plate 0)
      - solved policy episode total ~ +2240 (all on plates + entries + term.)
      - signal-to-noise ratio ~ 20x, well above the ~1.1x in v2.

    No-op for non-PressurePlate envs (so the wrapper can stay in the global
    default wrapper list without affecting RWARE/LBF).
    """

    def __init__(self, env, on_plate_reward=1.0, in_room_reward=0.1,
                 entry_bonus=10.0, terminal_bonus=50.0):
        super().__init__(env)
        self.on_plate_reward = float(on_plate_reward)
        self.in_room_reward = float(in_room_reward)
        self.entry_bonus = float(entry_bonus)
        self.terminal_bonus = float(terminal_bonus)

        spec_id = ""
        try:
            spec_id = (env.spec.id if env.spec is not None else "") or ""
        except Exception:
            pass
        self._active = spec_id.lower().startswith("pressureplate")
        self._on_plate_seen = None  # populated in reset()

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        if self._active:
            self._on_plate_seen = [False] * self.unwrapped.n_agents
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if not self._active:
            return obs, reward, done, info

        u = self.unwrapped  # raw PressurePlate env (exposes agents/plates/goal)
        n = u.n_agents
        new_reward = [0.0] * n
        if self._on_plate_seen is None:
            self._on_plate_seen = [False] * n

        for i, agent in enumerate(u.agents):
            # Target cell: plate i for agents 0..N-2, the goal cell for agent N-1.
            if i == n - 1:
                tx, ty = u.goal.x, u.goal.y
            else:
                tx, ty = u.plates[i].x, u.plates[i].y

            on_target = (agent.x == tx and agent.y == ty)
            curr_room = u._get_curr_room_reward(agent.y)
            in_correct_room = (i == curr_room)

            if on_target:
                new_reward[i] += self.on_plate_reward
                if not self._on_plate_seen[i]:
                    new_reward[i] += self.entry_bonus
                    self._on_plate_seen[i] = True
            elif in_correct_room:
                new_reward[i] += self.in_room_reward
            # else: 0 (wrong room, no signal)

        # Terminal success bonus on goal achievement (not timeout)
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
