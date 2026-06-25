import torch
import onnx
from stable_baselines3 import PPO
from aegis.pinn_madrl.pinn import PINN
import os

def export_models(pinn_path="pinn_model.pt", rl_path="rl_model.zip", out_dir="deploy"):
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. Export PINN
    if os.path.exists(pinn_path):
        pinn = PINN()
        pinn.load_state_dict(torch.load(pinn_path))
        pinn.eval()
        
        dummy_input = torch.randn(1, 3) # (r, t, I_coil)
        onnx_pinn_path = os.path.join(out_dir, "pinn.onnx")
        torch.onnx.export(
            pinn, dummy_input, onnx_pinn_path, 
            input_names=["input"], output_names=["output"],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )
        print(f"Exported PINN to {onnx_pinn_path}")
    else:
        print(f"PINN model not found at {pinn_path}")
        
    # 2. Export RL Policy
    if os.path.exists(rl_path):
        model = PPO.load(rl_path)
        policy = model.policy
        policy.eval()
        
        dummy_obs = torch.randn(1, 67) # 67 is observation space size
        onnx_rl_path = os.path.join(out_dir, "rl_policy.onnx")
        
        # SB3 policy forward takes obs, returns (action, value, log_prob)
        # We only care about the action network (mu) for deterministic inference
        class PolicyNet(torch.nn.Module):
            def __init__(self, policy):
                super().__init__()
                self.policy = policy
            def forward(self, obs):
                # Return deterministic action
                return self.policy.action_net(self.policy.mlp_extractor.forward_actor(obs))
                
        net = PolicyNet(policy)
        torch.onnx.export(
            net, dummy_obs, onnx_rl_path,
            input_names=["obs"], output_names=["action"],
            dynamic_axes={'obs': {0: 'batch_size'}, 'action': {0: 'batch_size'}}
        )
        print(f"Exported RL Policy to {onnx_rl_path}")
    else:
        print(f"RL model not found at {rl_path}")

if __name__ == "__main__":
    export_models()
