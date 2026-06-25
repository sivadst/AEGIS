import asyncio
import numpy as np
import time
from deploy.tensorrt_mock import MockTensorRTEngine
import os

async def micro_controller_loop(rl_onnx_path="deploy/rl_policy.onnx", duration=60):
    print("Starting Grid Edge Micro-Controller Loop...")
    
    if not os.path.exists(rl_onnx_path):
        print(f"Error: {rl_onnx_path} not found. Ensure models are exported first.")
        return
        
    rl_engine = MockTensorRTEngine(rl_onnx_path)
    
    # Grid Simulation State
    freq = 50.0 # Base frequency Hz
    uptime = 0.0
    
    start_time = time.time()
    
    print("Time(s) | Solar/Wind | Load  | Reactor | Charge Rate | Freq (Hz)")
    print("-" * 65)
    
    while True:
        current_time = time.time() - start_time
        if current_time > duration:
            break
            
        # Simulate noisy sensor data
        # Solar/wind generation (noisy sine waves)
        renewables = max(0, 10.0 * np.sin(current_time * 0.1) + np.random.normal(0, 1.0))
        
        # Demand load
        demand = 15.0 + 5.0 * np.sin(current_time * 0.05) + np.random.normal(0, 0.5)
        
        # Fusion reactor availability signal (derived from stability margin)
        # Mocking this since we don't run the full simulator in this loop.
        stability_score = np.random.uniform(5.0, 10.0)
        reactor_power = 10.0 if stability_score > 6.0 else 0.0
        
        # We need a 67-dim state for the RL agent. 
        # In a real system, the RL agent balancing the grid would have a different state space 
        # than the plasma control agent. 
        # For the narrative PoC, we will distill/reuse the RL policy by padding the grid state into it 
        # or simply generating a dummy state, since the prompt says: 
        # "The RL policy (or a distilled version) outputs battery charge/discharge rates to balance the grid"
        # We will use the output action of the RL agent as the charge rate.
        dummy_state = np.zeros(67, dtype=np.float32)
        dummy_state[0] = renewables
        dummy_state[1] = demand
        dummy_state[2] = reactor_power
        
        action = rl_engine.execute(dummy_state)[0] # shape (2,)
        # Map actions to battery charge/discharge (-1 to 1) -> (-5 to +5 MW)
        charge_rate = (action[0] + action[1]) * 2.5 
        
        # Balance calculation
        net_power = renewables + reactor_power - demand - charge_rate
        load_mismatch = abs(net_power) / demand * 100
        
        # Frequency deviation (inertia simulation)
        freq_dev = net_power * 0.05
        freq = 50.0 + freq_dev
        
        uptime += 1.0
        
        print(f"{current_time:04.1f}   | {renewables:5.1f}      | {demand:5.1f} | {reactor_power:5.1f}   | {charge_rate:6.2f}      | {freq:5.2f}")
        
        await asyncio.sleep(1.0) # 1 second control loop
        
    print(f"Micro-Controller Loop Completed. Uptime: {uptime} cycles.")

def run_grid():
    asyncio.run(micro_controller_loop())

if __name__ == "__main__":
    run_grid()
