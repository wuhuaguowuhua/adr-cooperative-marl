"""
PettingZoo to Gym-style wrapper for SEAC compatibility.
Converts PettingZoo ParallelEnv to the multi-agent gym format expected by SEAC.
"""
import numpy as np
import gym
from gym import spaces
from pettingzoo.sisl import pursuit_v4


class PursuitGymWrapper(gym.Env):
    """
    Wraps PettingZoo Pursuit environment to gym-style interface compatible with SEAC.
    """
    
    def __init__(self, n_pursuers=4, obs_range=7, n_evaders=30, tag_reward=0.01,
                 catch_reward=5.0, urgency_reward=0.0, max_cycles=500):
        super().__init__()
        
        # 创建PettingZoo环境
        self.env = pursuit_v4.parallel_env(
            n_pursuers=n_pursuers,
            n_evaders=n_evaders,
            obs_range=obs_range,
            tag_reward=tag_reward,
            catch_reward=catch_reward,
            urgency_reward=urgency_reward,
            max_cycles=max_cycles,
        )
        
        self.env.reset()
        self.agents = self.env.agents
        self.n_agents = len(self.agents)
        
        # 获取空间
        sample_agent = self.agents[0]
        obs_space = self.env.observation_space(sample_agent)
        act_space = self.env.action_space(sample_agent)
        
        # 扁平化观测空间 (7,7,3) -> (147,)
        self.obs_shape = obs_space.shape
        flat_obs_dim = np.prod(self.obs_shape)
        
        # 创建gym兼容的空间（列表形式，每个agent一个）
        self.observation_space = [
            spaces.Box(low=0.0, high=30.0, shape=(flat_obs_dim,), dtype=np.float32)
            for _ in range(self.n_agents)
        ]
        self.action_space = [
            spaces.Discrete(act_space.n) for _ in range(self.n_agents)
        ]
        
        self._max_episode_steps = max_cycles
        
    def reset(self):
        obs_dict, info = self.env.reset()
        # 转换为列表格式，扁平化观测
        obs_list = [obs_dict[agent].flatten().astype(np.float32) for agent in self.agents]
        return obs_list
    
    def step(self, actions):
        """
        actions: list of actions for each agent
        """
        # 转换为字典格式
        action_dict = {agent: int(actions[i]) for i, agent in enumerate(self.agents)}
        
        obs_dict, rew_dict, term_dict, trunc_dict, info_dict = self.env.step(action_dict)
        
        # 转换回列表格式
        obs_list = [obs_dict[agent].flatten().astype(np.float32) for agent in self.agents]
        rew_list = [rew_dict[agent] for agent in self.agents]
        
        # done: 任意agent结束则全部结束
        done = any(term_dict.values()) or any(trunc_dict.values())
        
        # 聚合info
        info = {}
        if done:
            total_reward = sum(rew_list)
            info['episode_reward'] = total_reward
            # 尝试获取episode长度
            try:
                info['episode_length'] = getattr(self.env.unwrapped, 'frames', 500)
            except:
                info['episode_length'] = 500
        
        return obs_list, rew_list, done, info
    
    def seed(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        return [seed]
    
    def render(self, mode='human'):
        return self.env.render()
    
    def close(self):
        self.env.close()


def make_pursuit_env(n_pursuers=4, max_cycles=500):
    """工厂函数，创建Pursuit环境"""
    def _thunk():
        return PursuitGymWrapper(n_pursuers=n_pursuers, max_cycles=max_cycles)
    return _thunk


# 注册为gym环境（可选）
if __name__ == "__main__":
    # 测试
    env = PursuitGymWrapper(n_pursuers=4)
    print(f"Agent数量: {env.n_agents}")
    print(f"观测空间: {env.observation_space[0]}")
    print(f"动作空间: {env.action_space[0]}")
    
    obs = env.reset()
    print(f"Reset观测形状: {[o.shape for o in obs]}")
    
    actions = [env.action_space[i].sample() for i in range(env.n_agents)]
    obs, rew, done, info = env.step(actions)
    print(f"Step观测形状: {[o.shape for o in obs]}")
    print(f"奖励: {rew}")
    print("测试成功!")
