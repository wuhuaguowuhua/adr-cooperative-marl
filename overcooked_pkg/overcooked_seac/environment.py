"""Overcooked-AI cooperative env, exposed as a multi-agent gym.Env.

step() returns the 4-tuple (obs_tuple, reward_list, done_list, info)
that our SEAC / MAA2C / MAPPO training loop expects (matches
PressurePlate / BoxPushing / RWARE / LBF wrappers).

Per-agent reward = sparse_r_by_agent[i] + shaped_r_by_agent[i].
This gives each agent its own credit signal (specialization-friendly,
which is what RCDC is designed to amplify).
"""

import numpy as np
import gym
from gym import spaces

from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld, Action
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv


class OvercookedSEAC(gym.Env):
    metadata = {"render.modes": []}

    def __init__(self, layout_name="asymmetric_advantages", horizon=400):
        super().__init__()
        self.layout_name = layout_name
        self.horizon = int(horizon)

        self._mdp = OvercookedGridworld.from_layout_name(layout_name)
        self._env = OvercookedEnv.from_mdp(self._mdp, horizon=self.horizon)

        # Probe obs dim by running featurize once.
        self._env.reset()
        sample = self._env.featurize_state_mdp(self._env.state)
        obs_dim = int(np.asarray(sample[0]).shape[0])

        self.n_agents = 2
        self._actions = Action.ALL_ACTIONS  # [N, S, E, W, STAY, INTERACT]
        assert len(self._actions) == 6

        single_obs = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        single_act = spaces.Discrete(len(self._actions))

        self.observation_space = spaces.Tuple(
            tuple(single_obs for _ in range(self.n_agents))
        )
        self.action_space = spaces.Tuple(
            tuple(single_act for _ in range(self.n_agents))
        )

        self._np_random = np.random.RandomState()
        self._step_count = 0

    def seed(self, seed=None):
        if seed is not None:
            self._np_random = np.random.RandomState(int(seed))
            try:
                self._mdp.np_random = self._np_random
            except Exception:
                pass
        return [seed]

    def _featurize(self):
        feat = self._env.featurize_state_mdp(self._env.state)
        return tuple(np.asarray(f, dtype=np.float32) for f in feat)

    def reset(self):
        self._env.reset()
        self._step_count = 0
        return self._featurize()

    def step(self, actions):
        # actions: iterable of length n_agents, each a Python int or 0-d int tensor.
        joint = []
        for a in actions:
            if hasattr(a, "item"):
                a = a.item()
            joint.append(self._actions[int(a)])

        _next_state, _joint_reward, done, info = self._env.step(joint)
        self._step_count += 1

        sparse = info.get("sparse_r_by_agent", [0, 0])
        shaped = info.get("shaped_r_by_agent", [0, 0])
        reward_list = [float(sparse[i]) + float(shaped[i]) for i in range(self.n_agents)]
        done_list = [bool(done)] * self.n_agents

        info_out = {}
        if done:
            info_out["TimeLimit.truncated"] = bool(self._step_count >= self.horizon)

        return self._featurize(), reward_list, done_list, info_out

    def render(self, mode="human"):
        return None

    def close(self):
        return None
