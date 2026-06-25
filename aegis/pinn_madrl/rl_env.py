import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
from aegis.plasma_sim.simulator import TorchMHD1D

class FusionEnv(gym.Env):
    def __init__(self, agent_id=0):
        super().__init__()
        
        self.simulator = TorchMHD1D()
        self.agent_id = agent_id  # 0 or 1 for multi-agent narrative
        
        # State space: q(r) [32], p(r) [32], I_coil [1], time_step [1], stability_margin [1] -> 67
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(67,), dtype=np.float32)
        
        # Action space: a1, a2 in [-1, 1] representing two coil groups
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        
        self.max_steps = 100
        self.current_step = 0
        
        self.q = None
        self.p = None
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        
        # Random initial profile similar to dataset generation
        q_center = 1.1 + torch.rand(1) * 0.5
        q_edge = 2.5 + torch.rand(1) * 1.5
        r = self.simulator.r
        self.q = q_center + (q_edge - q_center) * (r**2)
        self.q = self.q.unsqueeze(0)
        
        self.p = 2.0 * (1.0 - r**2)
        self.p = self.p.unsqueeze(0)
        
        self.I_coil = torch.zeros(1, 1)
        
        return self._get_obs(), {}
        
    def _get_obs(self):
        # Calculate stability margin (distance to disruption)
        q_edge = self.q[0, -1].item()
        
        grad_p = torch.zeros_like(self.p)
        grad_p[:, 1:-1] = (self.p[:, 2:] - self.p[:, :-2]) / (2 * self.simulator.dr)
        grad_p[:, 0] = (self.p[:, 1] - self.p[:, 0]) / self.simulator.dr
        grad_p[:, -1] = (self.p[:, -1] - self.p[:, -2]) / self.simulator.dr
        max_grad_p = torch.max(torch.abs(grad_p), dim=1)[0].item()
        
        stability_margin = min(q_edge - 2.0, 15.0 - max_grad_p)
        
        obs = np.concatenate([
            self.q.squeeze(0).numpy(),
            self.p.squeeze(0).numpy(),
            self.I_coil.squeeze(0).numpy(),
            np.array([self.current_step], dtype=np.float32),
            np.array([stability_margin], dtype=np.float32)
        ])
        return obs
        
    def step(self, action):
        a1, a2 = action
        # Map to I_coil in [-2.0, 2.0] kA
        # Combine actions from both "coil groups"
        I_coil_val = (a1 + a2) # [-2, 2]
        self.I_coil = torch.tensor([[I_coil_val]], dtype=torch.float32)
        
        # Simulate step
        with torch.no_grad():
            self.q, self.p = self.simulator(self.q, self.p, self.I_coil)
            
        self.current_step += 1
        
        # Check disruption
        disrupted = self.simulator.check_major_disruption(self.q, self.p)[0].item()
        
        # Reward
        p_mean = torch.mean(self.p).item()
        action_penalty = 0.02 * (a1**2 + a2**2)
        
        reward = 1.0 + 0.5 * p_mean - action_penalty
        
        done = False
        if disrupted:
            reward -= 10.0
            done = True
        elif self.current_step >= self.max_steps:
            done = True
            
        return self._get_obs(), reward, done, False, {}

