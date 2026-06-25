import torch
import torch.nn as nn
import numpy as np
import os

class TorchMHD1D(nn.Module):
    def __init__(self, N=32, dt=0.01):
        super().__init__()
        self.N = N
        self.dt = dt
        self.r = torch.linspace(0, 1, N)
        self.dr = self.r[1] - self.r[0]
        
        # Physics constants
        self.eta = 0.1
        self.alpha = 0.5
        self.tau_p = 1.0
        self.p_source = 2.0
        self.Dp = 0.05
        self.beta = 5.0
        
        # Precompute spatial terms
        self.coil_profile = torch.exp(-(self.r - 0.8)**2 / 0.02)
        
    def forward(self, q, p, I_coil):
        # q: (batch_size, N)
        # p: (batch_size, N)
        # I_coil: (batch_size, 1)
        
        batch_size = q.shape[0]
        
        # Compute second derivatives using finite difference
        d2q = torch.zeros_like(q)
        d2q[:, 1:-1] = (q[:, 2:] - 2*q[:, 1:-1] + q[:, :-2]) / (self.dr**2)
        d2q[:, 0] = (2*q[:, 1] - 2*q[:, 0]) / (self.dr**2)
        # Let d2q[:, -1] = 0 or similar
        d2q[:, -1] = (2*q[:, -2] - 2*q[:, -1]) / (self.dr**2)

        d2p = torch.zeros_like(p)
        d2p[:, 1:-1] = (p[:, 2:] - 2*p[:, 1:-1] + p[:, :-2]) / (self.dr**2)
        d2p[:, 0] = (2*p[:, 1] - 2*p[:, 0]) / (self.dr**2)
        d2p[:, -1] = (2*p[:, -2] - 2*p[:, -1]) / (self.dr**2)

        # Disruption mask (sawtooth / minor disruption)
        q0 = q[:, 0].unsqueeze(1)
        sawtooth_mask = (q0 < 1.0).float()
        
        # PDEs
        # dq/dt = eta * d2q/dr2 + alpha * I_coil(t) * exp(-(r-0.8)^2 / 0.02)
        dq_dt = self.eta * d2q + self.alpha * I_coil * self.coil_profile.unsqueeze(0).to(q.device)
        
        # dp/dt = (p_source - p)/tau_p + Dp * d2p/dr2 - beta * 1_{q(0)<1} * p
        dp_dt = (self.p_source - p) / self.tau_p + self.Dp * d2p - self.beta * sawtooth_mask * p
        
        # Euler step
        q_new = q + self.dt * dq_dt
        p_new = p + self.dt * dp_dt
        
        # Apply Boundary Conditions
        q_new[:, 0] = q_new[:, 1]  # dq/dr = 0 at r=0
        p_new[:, -1] = 0.0         # p(1) = 0
        p_new[:, 0] = p_new[:, 1]  # dp/dr = 0 at r=0
        
        return q_new, p_new

    def check_major_disruption(self, q, p):
        # Major Disruption: q_edge < 2.0 OR max(grad_p) > 15
        q_edge = q[:, -1]
        
        grad_p = torch.zeros_like(p)
        grad_p[:, 1:-1] = (p[:, 2:] - p[:, :-2]) / (2 * self.dr)
        grad_p[:, 0] = (p[:, 1] - p[:, 0]) / self.dr
        grad_p[:, -1] = (p[:, -1] - p[:, -2]) / self.dr
        max_grad_p = torch.max(torch.abs(grad_p), dim=1)[0]
        
        disrupted = (q_edge < 2.0) | (max_grad_p > 15.0)
        return disrupted


def generate_dataset(num_traj=1000, steps=30, save_path=None):
    print(f"Generating {num_traj} trajectories...")
    model = TorchMHD1D()
    model.eval()
    
    r_vals = []
    t_vals = []
    I_coil_vals = []
    q_vals = []
    p_vals = []
    
    with torch.no_grad():
        for i in range(num_traj):
            # Random initial profiles
            q_center = 1.1 + torch.rand(1) * 0.5  # 1.1 to 1.6
            q_edge = 2.5 + torch.rand(1) * 1.5    # 2.5 to 4.0
            r = model.r
            q = q_center + (q_edge - q_center) * (r**2)
            q = q.unsqueeze(0)
            
            p = 2.0 * (1.0 - r**2)
            p = p.unsqueeze(0)
            
            # Random coil sequence for this trajectory
            I_coil_seq = (torch.rand(steps, 1) * 4.0 - 2.0)
            
            t = 0.0
            for step in range(steps):
                I_coil = I_coil_seq[step:step+1]
                q, p = model(q, p, I_coil)
                t += model.dt
                
                # We normalize t to some extent if needed, but dt=0.01 makes it small.
                for j in range(model.N):
                    r_vals.append(r[j].item())
                    t_vals.append(t)
                    I_coil_vals.append(I_coil.item())
                    q_vals.append(q[0, j].item())
                    p_vals.append(p[0, j].item())
                    
                if model.check_major_disruption(q, p)[0]:
                    break
                    
    dataset = {
        "r": torch.tensor(r_vals, dtype=torch.float32).unsqueeze(1),
        "t": torch.tensor(t_vals, dtype=torch.float32).unsqueeze(1),
        "I_coil": torch.tensor(I_coil_vals, dtype=torch.float32).unsqueeze(1),
        "q": torch.tensor(q_vals, dtype=torch.float32).unsqueeze(1),
        "p": torch.tensor(p_vals, dtype=torch.float32).unsqueeze(1)
    }
    
    if save_path:
        torch.save(dataset, save_path)
        print(f"Dataset saved to {save_path}")
        
    return dataset

if __name__ == "__main__":
    generate_dataset(num_traj=10, save_path="dummy_dataset.pt")
