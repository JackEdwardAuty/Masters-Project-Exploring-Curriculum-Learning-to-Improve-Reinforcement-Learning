"""
GUARANTEED RENDERING - Windows Popup Window!
"""

import numpy as np
import mujoco
import mujoco.viewer
from curriculum_env import CurriculumEnvironment
from stable_baselines3 import PPO
import time

print("🎥 LOADING MODEL...")
model = PPO.load("experiments/checkpoints/final_model.zip")

print("🎮 STARTING LIVE PLAYBACK...")
env = CurriculumEnvironment(render_mode="human", max_steps=300)

for episode in range(10):
    print(f"\n--- Episode {episode+1} ---")
    obs, _ = env.reset()
    total_reward = 0
    done = False
    
    while not done:
        # Predict action
        action, _ = model.predict(obs, deterministic=True)
        
        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # RENDER - Window pops up!
        env.render()
        
        done = terminated or truncated
        
        # Print progress
        if len(str(total_reward)) > 6:
            print(f"\rReward: {total_reward:.1f} | Dist: {info.get('distance',0):.2f}", end="")
        time.sleep(0.02)  # 50 FPS
    
    print(f"\n✅ Episode {episode+1} FINISHED! Reward: {total_reward:.1f}")

env.close()
print("🎉 Playback complete!")
