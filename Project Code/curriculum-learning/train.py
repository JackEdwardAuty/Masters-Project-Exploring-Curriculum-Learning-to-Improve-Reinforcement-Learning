"""
Main Training Script for Curriculum Learning

Trains an RL agent using PPO with curriculum learning progression functions.
Supports multiple progression strategies: linear, exponential, and friction-based.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
import gymnasium as gym

# Add this at the top after imports (line ~15)
import multiprocessing as mp
if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    # from gymnasium.envs.registration import register
    # register(
    #     id="CurriculumEnv-v0",
    #     entry_point="curriculum_env.env:CurriculumEnv",
    # )

# Add src directory to path
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from curDriculum_envs import CurriculumEnvironment
from progression_functions import (
    get_progression_function,
    MultiProcessProgression,
    MappingFunction,
)


def make_env(env_id: int, progression_fn, mapping_fn, total_steps: int):
    """Create environment factory function."""
    
    def _make():
        import curriculum_envs # to trigger reg on spawn (multiprocessing)
        import gymnasium as gym
        # Create environment
        env = gym.make(
            "CurriculumEnv-v0",
            target_speed=0.0,
            floor_size=2.0,
            max_steps=100,
        )
        
        # Store progression and mapping for curriculum update
        env.progression_fn = progression_fn
        env.mapping_fn = mapping_fn
        env.env_id = env_id
        env.total_steps = total_steps
        env.current_step = 0
        
        return env
    
    return _make


class CurriculumCallback:
    """Callback to update environment difficulty during training."""
    
    def __init__(
        self,
        env,
        progression_fn,
        mapping_fn,
        update_interval: int = 5000,
    ):
        self.env = env
        self.progression_fn = progression_fn
        self.mapping_fn = mapping_fn
        self.update_interval = update_interval
        self.last_update = 0
    
    def __call__(self, model, logs: dict) -> None:
        """Update curriculum if needed."""
        current_step = model.num_timesteps
        
        if current_step - self.last_update >= self.update_interval:
            self._update_curriculum(current_step)
            self.last_update = current_step
    
    def _update_curriculum(self, timestep: int) -> None:
        """Update all environment complexities."""
        if isinstance(self.env, SubprocVecEnv):
            # Update each subprocess environment
            for i, env in enumerate(self.env.envs):
                self._update_single_env(env, timestep, i)
        elif isinstance(self.env, DummyVecEnv):
            # Update dummy vec env
            for i, env in enumerate(self.env.envs):
                self._update_single_env(env, timestep, i)
        else:
            # Single environment
            self._update_single_env(self.env, timestep, 0)
    
    def _update_single_env(self, env, timestep: int, env_id: int) -> None:
        """Update complexity for single environment."""
        # Unwrap if needed
        while hasattr(env, "env"):
            env = env.env
        
        # Get new complexity parameters
        target_speed, floor_size = self.progression_fn(timestep, env_id)
        
        # Update environment
        if hasattr(env, "set_complexity"):
            env.set_complexity(target_speed, floor_size)


def train(
    progression_type: str = "friction",
    num_processes: int = 4,
    total_steps: int = 1000000,
    learning_rate: float = 3e-4,
    batch_size: int = 64,
    n_epochs: int = 4,
    checkpoint_dir: str = "experiments/checkpoints",
    log_dir: str = "experiments/logs",
    render: bool = False,
    seed: int = 42,
):
    """
    Train RL agent with curriculum learning.
    
    Args:
        progression_type: "linear", "exponential", or "friction"
        num_processes: Number of parallel environments
        total_steps: Total training steps
        learning_rate: PPO learning rate
        batch_size: PPO batch size
        n_epochs: PPO epochs per update
        checkpoint_dir: Directory to save model checkpoints
        log_dir: Directory to save training logs
        render: Whether to render environment
        seed: Random seed
    """
    
    # Create directories
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # # Register custom environment
    # gym.register(
    #     id="CurriculumEnv-v0",
    #     entry_point="curriculum_env:CurriculumEnvironment",
    #     max_episode_steps=100,
    # )
    
    # FIXED: Register environment BEFORE creating
    from gymnasium.envs.registration import register

    register(
        id="CurriculumEnv-v0",
        entry_point="curriculum_env:CurriculumEnvironment",
        max_episode_steps=100,
    )


    # Set seeds
    np.random.seed(seed)
    
    # Create progression and mapping functions
    if progression_type == "none":
        # No curriculum - fixed maximum difficulty
        progression_fn = lambda t, pid: (0.89, 24.0)
        mapping_fn = MappingFunction()
    else:
        multi_prog = MultiProcessProgression(
            num_processes=num_processes,
            progression_type=progression_type,
            total_steps=total_steps,
        )
        
        def progression_fn(timestep, process_id):
            target_speed, floor_size = multi_prog(timestep, process_id)
            return target_speed, floor_size
        
        mapping_fn = multi_prog.mapping
    
    # Create vectorized environment
    print(f"Creating {num_processes} parallel environments...")
    
    env_fns = [
        make_env(i, progression_fn, mapping_fn, total_steps)
        for i in range(num_processes)
    ]
    
    # Use SubprocVecEnv for actual parallel execution
    env = SubprocVecEnv(env_fns, start_method="spawn")
    
    # Create PPO model
    print("Creating PPO agent...")
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=1,
        tensorboard_log=log_dir,
        device="cpu"
    )
    
    # Create callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path=checkpoint_dir,
        name_prefix="curriculum_model",
        save_replay_buffer=False,
    )
    
    # Curriculum callback
    curriculum_callback_instance = CurriculumCallback(
        env=env,
        progression_fn=progression_fn,
        mapping_fn=mapping_fn,
        update_interval=5000,
    )
    
    # Train
    print(f"\nStarting training: {progression_type} progression, {total_steps} steps...")
    print(f"  Processes: {num_processes}")
    print(f"  Learning Rate: {learning_rate}")
    print(f"  Batch Size: {batch_size}")
    
    try:
        model.learn(
            total_timesteps=total_steps,
            callback=checkpoint_callback,
            progress_bar=True,
        )
        
        # Save final model
        final_path = os.path.join(checkpoint_dir, "final_model")
        model.save(final_path)
        print(f"\nTraining completed! Final model saved to {final_path}")
        
        # Save training config
        config = {
            "progression_type": progression_type,
            "num_processes": num_processes,
            "total_steps": total_steps,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "n_epochs": n_epochs,
            "seed": seed,
        }
        
        config_path = os.path.join(log_dir, "training_config.json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        
        print(f"Config saved to {config_path}")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        # Save interrupt model
        interrupt_path = os.path.join(checkpoint_dir, "interrupt_model")
        model.save(interrupt_path)
        print(f"Model saved to {interrupt_path}")
    
    finally:
        env.close()


def main():
    """Parse arguments and run training."""
    parser = argparse.ArgumentParser(
        description="Train RL agent with curriculum learning"
    )
    
    parser.add_argument(
        "--progression",
        type=str,
        default="friction",
        choices=["linear", "exponential", "friction", "none"],
        help="Progression function type",
    )
    
    parser.add_argument(
        "--num-processes",
        type=int,
        default=4,
        help="Number of parallel environments",
    )
    
    parser.add_argument(
        "--total-steps",
        type=int,
        default=1000000,
        help="Total training steps",
    )
    
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
        help="PPO learning rate",
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="PPO batch size",
    )
    
    parser.add_argument(
        "--n-epochs",
        type=int,
        default=4,
        help="PPO epochs per update",
    )
    
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="experiments/checkpoints",
        help="Checkpoint directory",
    )
    
    parser.add_argument(
        "--log-dir",
        type=str,
        default="experiments/logs",
        help="Log directory",
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    
    args = parser.parse_args()
    
    train(
        progression_type=args.progression,
        num_processes=args.num_processes,
        total_steps=args.total_steps,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
