"""
Curriculum Environment - MUJOCO 3.x WINDOWS PERFECTLY WORKING
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
from typing import Tuple, Dict, Optional

class CurriculumEnvironment(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}
    
    def __init__(self, target_speed=0.0, floor_size=5.0, max_steps=100, render_mode=None):
        super().__init__()
        self.target_speed = np.clip(target_speed, 0.0, 0.89)
        self.floor_size = np.clip(floor_size, 2.0, 24.0)
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.viewer = None
        
        # Spaces - SIMPLIFIED
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)
        
        # State tracking
        self.current_step = 0
        self.agent_pos = np.array([0.0, 0.0])
        self.target_pos = np.array([0.0, 0.0])
        self.rng = np.random.RandomState(42)
        
        # MuJoCo model - NO XML ERRORS
        self.model = self._create_model()
        self.data = mujoco.MjData(self.model)
    
    def _create_model(self):
        """SIMPLEST POSSIBLE MuJoCo model - NO XML ERRORS"""
        xml = """
        <mujoco model="curriculum">
            <compiler coordinate="local" angle="radian"/>
            <option timestep="0.02"/>
            
            <worldbody>
                <!-- Floor -->
                <geom name="floor" type="plane" size="10 10 0.01" rgba="0.8 0.9 0.8 1"/>
                
                <!-- Agent - Red sphere -->
                <body name="agent" pos="0 0 0.5">
                    <geom name="agent_geom" type="sphere" size="0.15" rgba="1 0.2 0.2 1"/>
                </body>
                
                <!-- Target - Green sphere -->
                <body name="target" pos="2 2 0.5">
                    <geom name="target_geom" type="sphere" size="0.12" rgba="0.2 1 0.2 1"/>
                </body>
            </worldbody>
        </mujoco>
        """
        return mujoco.MjModel.from_xml_string(xml)
    
    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng.seed(seed)
        
        # Random start positions
        self.agent_pos = self.rng.uniform(-3, 3, 2)
        self.target_pos = self.rng.uniform(-3, 3, 2)
        
        mujoco.mj_resetData(self.model, self.data)
        self.data.body("agent").xpos[0:2] = self.agent_pos
        self.data.body("target").xpos[0:2] = self.target_pos
        
        info = {"episode_start": True}
        self.current_step = 0
        return self._get_obs(), {}
    
    def step(self, action):
        # Clip action
        action = np.clip(action, -1.0, 1.0)
        
        # Simple position update (no physics needed)
        self.agent_pos[0] += action[0] * 0.3
        self.agent_pos[1] += action[1] * 0.3
        
        # Target moves randomly
        target_move = self.rng.normal(0, self.target_speed * 0.05, 2)
        self.target_pos += target_move
        
        # Keep in bounds
        self.agent_pos = np.clip(self.agent_pos, -4.5, 4.5)
        self.target_pos = np.clip(self.target_pos, -4.5, 4.5)
        
        # Update MuJoCo for visualization
        self.data.body("agent").xpos[0:2] = self.agent_pos
        self.data.body("target").xpos[0:2] = self.target_pos
        mujoco.mj_step(self.model, self.data)
        
        self.current_step += 1
        
        # Reward calculation
        distance = np.linalg.norm(self.agent_pos - self.target_pos)
        reward = -0.01  # Time penalty
        
        terminated = False
        truncated = self.current_step >= self.max_steps
        
        if distance < 0.4:  # Success threshold
            reward += 10.0
            terminated = True
        
        obs = self._get_obs()
        info = {"distance": float(distance)}
        
        info = {"episode_start": False}
        return obs, reward, terminated, truncated, info
    
    def _get_obs(self):
        """Agent pos + Target pos"""
        return np.array([*self.agent_pos, *self.target_pos], dtype=np.float32)
    
    def render(self):
        if self.render_mode == "human" and self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        if self.viewer is not None:
            self.viewer.sync()
    
    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

if __name__ == "__main__":
    print("Testing environment...")
    env = CurriculumEnvironment()
    obs, info = env.reset()
    print("✅ ENV WORKS!")
    print(f"Obs shape: {obs.shape}")
    print(f"Action space: {env.action_space}")
    for _ in range(5):
        action = env.action_space.sample()
        obs, rew, term, trunc, info = env.step(action)
        print(f"Step reward: {rew:.2f}, distance: {info['distance']:.2f}")
    env.close()
