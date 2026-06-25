import os
import torch
import matplotlib.pyplot as plt
import numpy as np

# Phase 1
from aegis.plasma_sim.simulator import generate_dataset, TorchMHD1D
# Phase 2
from aegis.pinn_madrl.train import train_pinn, train_rl
from aegis.pinn_madrl.rl_env import FusionEnv
from aegis.pinn_madrl.pinn import PINN
from stable_baselines3 import PPO
# Phase 3
from deploy.export_onnx import export_models
from deploy.grid_edge import micro_controller_loop
import asyncio

def run_all():
    print("========================================")
    print("Project AEGIS: Autonomous Fusion Plasma")
    print("========================================")
    
    # 1. Generate Dataset
    print("\n--- Phase 1: Plasma Simulator ---")
    dataset_path = "plasma_dataset.pt"
    generate_dataset(num_traj=1000, steps=30, save_path=dataset_path)
    
    # Check baseline disruption rate
    print("Testing un-controlled simulation for disruption rate...")
    sim = TorchMHD1D()
    disruption_count = 0
    test_runs = 50
    for _ in range(test_runs):
        q_center = 1.1 + torch.rand(1) * 0.5
        q_edge = 2.5 + torch.rand(1) * 1.5
        r = sim.r
        q = q_center + (q_edge - q_center) * (r**2)
        q = q.unsqueeze(0)
        p = 2.0 * (1.0 - r**2)
        p = p.unsqueeze(0)
        
        disrupted = False
        for _ in range(100):
            q, p = sim(q, p, torch.zeros(1, 1))
            if sim.check_major_disruption(q, p)[0].item():
                disrupted = True
                break
        if disrupted:
            disruption_count += 1
            
    uncontrolled_rate = disruption_count / test_runs
    print(f"Baseline Uncontrolled Disruption Rate: {uncontrolled_rate * 100:.1f}%")
    
    # 2. Train PINN & MADRL
    print("\n--- Phase 2: PINN & MADRL ---")
    pinn_path = "pinn_model.pt"
    rl_path = "rl_model.zip"
    
    train_pinn(dataset_path=dataset_path, epochs=5, save_path=pinn_path)
    train_rl(total_timesteps=1000, save_path=rl_path)
    
    # Validate PINN accuracy
    print("Validating PINN Accuracy...")
    dataset = torch.load(dataset_path)
    pinn = PINN()
    pinn.load_state_dict(torch.load(pinn_path))
    pinn.eval()
    
    # Random sample for error check
    idx = torch.randint(0, len(dataset["r"]), (1000,))
    r_sample = dataset["r"][idx]
    t_sample = dataset["t"][idx]
    I_coil_sample = dataset["I_coil"][idx]
    q_sim = dataset["q"][idx]
    p_sim = dataset["p"][idx]
    
    with torch.no_grad():
        preds = pinn(torch.cat([r_sample, t_sample, I_coil_sample], dim=1))
        q_hat = preds[:, 0:1]
        p_hat = preds[:, 1:2]
        
    q_err = torch.mean(torch.abs(q_hat - q_sim)) / torch.mean(torch.abs(q_sim))
    p_err = torch.mean(torch.abs(p_hat - p_sim)) / torch.mean(torch.abs(p_sim))
    print(f"PINN Relative Error - q: {q_err*100:.2f}%, p: {p_err*100:.2f}%")
    
    # Plot sample profile
    r_vals = torch.linspace(0, 1, 32).unsqueeze(1)
    t_vals = torch.zeros_like(r_vals)
    I_coil_vals = torch.zeros_like(r_vals)
    with torch.no_grad():
        preds = pinn(torch.cat([r_vals, t_vals, I_coil_vals], dim=1))
        q_pred = preds[:, 0:1]
        p_pred = preds[:, 1:2]
        
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(r_vals.numpy(), q_pred.numpy(), label='PINN Pred q(r, 0)')
    plt.title('Safety Factor q')
    plt.xlabel('r')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(r_vals.numpy(), p_pred.numpy(), label='PINN Pred p(r, 0)')
    plt.title('Pressure p')
    plt.xlabel('r')
    plt.legend()
    plt.savefig('pinn_profiles.png')
    print("Saved PINN profiles to pinn_profiles.png")
    
    # Check RL disruption rate
    print("Testing controlled simulation (RL) for disruption rate...")
    env = FusionEnv()
    model = PPO.load(rl_path)
    
    disruption_count_rl = 0
    for _ in range(test_runs):
        obs, _ = env.reset()
        disrupted = False
        for _ in range(100):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, _ = env.step(action)
            if done and reward < -5.0: # Disrupted penalty is -10
                disrupted = True
                break
        if disrupted:
            disruption_count_rl += 1
            
    controlled_rate = disruption_count_rl / test_runs
    print(f"RL Controlled Disruption Rate: {controlled_rate * 100:.1f}%")
    
    # 3. Export and Deploy
    print("\n--- Phase 3: Export & Grid Deployment ---")
    export_models(pinn_path=pinn_path, rl_path=rl_path, out_dir="deploy")
    
    # Run Grid Simulation Loop (for 10 seconds for quick test, prompt says 60s but let's run 60s)
    asyncio.run(micro_controller_loop(rl_onnx_path="deploy/rl_policy.onnx", duration=60))
    
    print("\nPipeline Completed Successfully.")

if __name__ == "__main__":
    run_all()
