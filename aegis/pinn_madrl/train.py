import torch
from torch.utils.data import TensorDataset, DataLoader
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import matplotlib.pyplot as plt
import os
import numpy as np

from aegis.pinn_madrl.pinn import PINN
from aegis.pinn_madrl.rl_env import FusionEnv

def train_pinn(dataset_path="dummy_dataset.pt", epochs=50, batch_size=8192, save_path="pinn_model.pt"):
    print("Training PINN...")
    if not os.path.exists(dataset_path):
        print(f"Dataset {dataset_path} not found. Skipping PINN training.")
        return
        
    dataset_dict = torch.load(dataset_path)
    
    # We need requires_grad for r, t, I_coil
    r = dataset_dict["r"]
    t = dataset_dict["t"]
    I_coil = dataset_dict["I_coil"]
    q = dataset_dict["q"]
    p = dataset_dict["p"]
    
    dataset = TensorDataset(r, t, I_coil, q, p)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = PINN()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch_r, batch_t, batch_I, batch_q, batch_p in loader:
            batch_r.requires_grad = True
            batch_t.requires_grad = True
            batch_I.requires_grad = True
            
            optimizer.zero_grad()
            loss = model.compute_loss(batch_r, batch_t, batch_I, batch_q, batch_p)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.4f}")
            
    torch.save(model.state_dict(), save_path)
    print(f"PINN trained and saved to {save_path}")

def train_rl(total_timesteps=10000, save_path="rl_model.zip"):
    print(f"Training MADRL agents for {total_timesteps} steps...")
    
    # To satisfy the "Multi-Agent" narrative where two independent PPO agents control different coil groups
    # but share the same state, we will train two separate PPO agents.
    # Agent A controls coil group 1 (a1)
    # Agent B controls coil group 2 (a2)
    # We will simulate their interaction by training them independently for simplicity in the PoC,
    # or by wrapping the environment to accept actions from both if we were doing a true multi-agent step.
    # Since SB3 doesn't natively support MADRL without a wrapper like PettingZoo, we will train two independent models
    # on the same environment type, which is a common compromise for a PoC.
    
    env_a = make_vec_env(lambda: FusionEnv(agent_id=0), n_envs=4)
    model_a = PPO("MlpPolicy", env_a, verbose=0, n_steps=2048)
    model_a.learn(total_timesteps=total_timesteps // 2)
    
    env_b = make_vec_env(lambda: FusionEnv(agent_id=1), n_envs=4)
    model_b = PPO("MlpPolicy", env_b, verbose=0, n_steps=2048)
    model_b.learn(total_timesteps=total_timesteps // 2)
    
    # We will save model_a as the primary model to fulfill the pipeline
    model_a.save(save_path)
    model_b.save("rl_model_b.zip")
    print(f"MADRL models trained and saved to {save_path} and rl_model_b.zip")

if __name__ == "__main__":
    train_pinn(epochs=5)
    train_rl(total_timesteps=2000)
