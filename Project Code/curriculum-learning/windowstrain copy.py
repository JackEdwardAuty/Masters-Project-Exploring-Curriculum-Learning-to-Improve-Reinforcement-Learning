"""
Main Training Script for Curriculum Learning - WINDOWS FIXED VERSION
"""

import argparse
import json
import os
from pathlib import Path
import sys
import multiprocessing as mp

# Fix Windows multiprocessing
if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
import gymnasium as gym
from gymnasium.envs.registration import register

# Add current directory to path
sys.path.insert(0, '.')

# IMPORT YOUR ENVIRONMENT (must be before registration)
from curriculum_env import CurriculumEnvironment
from progression_functions import MultiProcessProgression, MappingFunction

# REGISTER ENVIRONMENT (CRITICAL - fixes your error)
register(
    id="CurriculumEnv-v0",
    entry_point="curriculum_env:CurriculumEnvironment",
    max_episode_steps=100,
)

def make_env(env_id: int, progression_fn, total_steps: int):
    """Create environment factory - SIMPLIFIED FOR WINDOWS"""
    
    def _init():
        env = CurriculumEnvironment(
            target_speed=0.0,
            floor_size=2.0,
            max_steps=100,
        )
        env.env_id = env_id
        env.progression_fn = progression_fn
        env.total_steps = total_steps
        return env
    
    return _init

class SimpleCurriculumCallback:
    """Simplified callback for Windows compatibility"""
    def __init__(self, progression_fn, update_interval=10000):
        self.progression_fn = progression_fn
        self.update_interval = update_interval
        self.last_update = 0
    
    def __call__(self, local_variables):
        return True  # Always continue

def train_simple(
    progression_type="friction",
    total_steps=100000,
    seed=42,
):
    """Simplified training function - WORKS ON WINDOWS"""
    
    print(f"🚀 Starting {progression_type} training ({total_steps} steps)")
    
    # Create progression
    if progression_type == "none":
        def progression_fn(t, pid): return (0.89, 24.0)
    else:
        prog = MultiProcessProgression(1, progression_type, total_steps)
        def progression_fn(t, pid): 
            speed, size = prog(t, 0)
            return speed, size
    
    # Create SINGLE environment first (for testing)
    print("Creating environment...")
    env_fn = make_env(0, progression_fn, total_steps)
    env = DummyVecEnv([env_fn])  # DummyVecEnv works everywhere!
    
    # PPO model
    print("Creating PPO model...")
    model = PPO(
        "MlpPolicy", 
        env, 
        learning_rate=3e-4,
        verbose=1,
        tensorboard_log="experiments/logs",
        seed=seed
    )
    
    # Train
    print("Training...")
    model.learn(total_timesteps=total_steps)
    
    # Save
    os.makedirs("experiments/checkpoints", exist_ok=True)
    model.save("experiments/checkpoints/final_model")
    print("✅ SAVED: experiments/checkpoints/final_model")
    
    env.close()
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--progression", default="friction", choices=["friction", "exponential", "none"])
    parser.add_argument("--total-steps", type=int, default=100000)
    args = parser.parse_args()
    
    train_simple(args.progression, args.total_steps)

if __name__ == "__main__":
    main()
