"""
Core Curriculum Environment for Reinforcement Learning
Simplified single-agent seeker tracking a moving target
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Dict, Any, Optional
import mujoco
import mujoco.viewer


class CurriculumEnvironment(gym.Env):
    """
    Custom MuJoCo-based environment for curriculum learning.
    
    Single agent (seeker) must track and reach a moving target.
    Difficulty controlled by:
    - Target speed: [0, 0.89]
    - Floor size: [2, 24]
    
    Episodes terminate when:
    - Agent reaches target (reward +1)
    - Agent goes out of bounds
    - Max steps reached (100)
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    
    def __init__(
        self,
        target_speed: float = 0.0,
        floor_size: float = 2.0,
        max_steps: int = 100,
        render_mode: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        """
        Initialize curriculum environment.
        
        Args:
            target_speed: Target movement speed [0, 0.89]
            floor_size: Floor size [2, 24]
            max_steps: Maximum steps per episode
            render_mode: 'human' for visualization, None for headless
            seed: Random seed for reproducibility
        """
        super().__init__()
        
        self.target_speed = np.clip(target_speed, 0.0, 0.89)
        self.floor_size = np.clip(floor_size, 2.0, 24.0)
        self.max_steps = max_steps
        self.render_mode = render_mode
        
        # RNG
        self.rng = np.random.RandomState(seed)
        
        # MuJoCo model
        self.model = self._create_model()
        self.data = mujoco.MjData(self.model)
        
        # Viewer
        self.viewer = None
        
        # State tracking
        self.step_count = 0
        self.agent_pos = np.array([0.0, 0.0, 0.0])
        self.target_pos = np.array([0.0, 0.0, 0.0])
        self.target_velocity = np.array([0.0, 0.0])
        
        # Action/Observation spaces
        # Actions: [vx, vy, rotation_z] - velocities
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32
        )
        
        # Observations: [agent_x, agent_y, target_x, target_y, agent_vx, agent_vy]
        self.observation_space = spaces.Box(
            low=-100.0, high=100.0, shape=(6,), dtype=np.float32
        )
        
        # Reward threshold for success
        self.reach_threshold = 0.5
        
    def _create_model(self) -> mujoco.MjModel:
        """Create MuJoCo model with agent and target."""
        xml_string = f"""
        <mujoco model="curriculum_env">
            <compiler angle="radian" coordinate="local" inertiafromgeom="true"/>
            
            <option timestep="0.01" gravity="0 0 -9.81"/>
            
            <asset>
                <texture builtin="gradient" height="100" rgb1="1 1 1" rgb2="0 0 0" type="2d" width="100"/>
                <material name="grid" specular="0.1" texture="textureid0"/>
            </asset>
            
            <worldbody>
                <!-- Floor -->
                <geom conaffinity="1" condim="3" name="floor" pos="0 0 0" 
                      rgba="0.8 0.9 0.8 1" size="{self.floor_size} {self.floor_size} 0.1" type="box"/>
                
                <!-- Agent (Seeker) -->
                <body name="agent" pos="0 0 0.1">
                    <geom conaffinity="1" condim="3" name="agent_geom" 
                          rgba="1 0 0 1" size="0.1" type="sphere"/>
                    <inertial mass="1" pos="0 0 0"/>
                    <site name="agent_site" pos="0 0 0" size="0.1"/>
                </body>
                
                <!-- Target -->
                <body name="target" pos="1 1 0.1">
                    <geom conaffinity="0" condim="3" name="target_geom" 
                          rgba="0 1 0 1" size="0.08" type="sphere"/>
                    <site name="target_site" pos="0 0 0" size="0.08"/>
                </body>
            </worldbody>
            
            <actuator>
                <!-- Velocity actuators for agent -->
                <velocity name="agent_vx" joint="agent_vx" kv="1.0"/>
                <velocity name="agent_vy" joint="agent_vy" kv="1.0"/>
                <velocity name="agent_rz" joint="agent_rz" kv="1.0"/>
            </actuator>
        </mujoco>
        """
        
        # Parse XML and create model
        model = mujoco.MjModel.from_xml_string(xml_string)
        return model
    
    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Reset environment to initial state.
        
        Returns:
            observation: Initial observation
            info: Additional info dict
        """
        super().reset(seed=seed)
        
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        
        # Reset data
        mujoco.mj_resetData(self.model, self.data)
        
        # Random initial positions
        self.agent_pos = np.array([
            self.rng.uniform(-self.floor_size + 0.2, self.floor_size - 0.2),
            self.rng.uniform(-self.floor_size + 0.2, self.floor_size - 0.2),
            0.1
        ])
        
        self.target_pos = np.array([
            self.rng.uniform(-self.floor_size + 0.2, self.floor_size - 0.2),
            self.rng.uniform(-self.floor_size + 0.2, self.floor_size - 0.2),
            0.1
        ])
        
        # Random initial target velocity
        if self.target_speed > 0:
            angle = self.rng.uniform(0, 2 * np.pi)
            self.target_velocity = np.array([
                self.target_speed * np.cos(angle),
                self.target_speed * np.sin(angle)
            ])
        else:
            self.target_velocity = np.zeros(2)
        
        # Update MuJoCo state
        agent_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "agent")
        target_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target")
        
        self.data.body(agent_id).pos = self.agent_pos
        self.data.body(target_id).pos = self.target_pos
        
        self.step_count = 0
        
        return self._get_observation(), {}
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step in the environment.
        
        Args:
            action: [vx, vy, rotation] velocities
            
        Returns:
            observation: Current observation
            reward: Reward for this step
            terminated: Whether episode is over (success/failure)
            truncated: Whether episode is truncated (max steps)
            info: Additional info
        """
        # Clip actions
        action = np.clip(action, -1.0, 1.0)
        
        # Scale actions (max velocity = 0.5 units/step)
        scaled_action = action * 0.5
        
        # Update agent position based on action
        self.agent_pos[0] += scaled_action[0]
        self.agent_pos[1] += scaled_action[1]
        
        # Update target position (random walk)
        self.target_pos[0] += self.target_velocity[0]
        self.target_pos[1] += self.target_velocity[1]
        
        # Bounce target off walls
        if np.abs(self.target_pos[0]) > self.floor_size - 0.1:
            self.target_velocity[0] *= -1
            self.target_pos[0] = np.clip(self.target_pos[0], 
                                        -self.floor_size + 0.1,
                                        self.floor_size - 0.1)
        
        if np.abs(self.target_pos[1]) > self.floor_size - 0.1:
            self.target_velocity[1] *= -1
            self.target_pos[1] = np.clip(self.target_pos[1],
                                        -self.floor_size + 0.1,
                                        self.floor_size - 0.1)
        
        self.step_count += 1
        
        # Calculate reward and termination
        distance = np.linalg.norm(self.agent_pos[:2] - self.target_pos[:2])
        
        reward = 0.0
        terminated = False
        truncated = False
        
        # Success: reached target
        if distance < self.reach_threshold:
            reward = 1.0
            terminated = True
        
        # Failure: out of bounds
        if (np.abs(self.agent_pos[0]) > self.floor_size or 
            np.abs(self.agent_pos[1]) > self.floor_size):
            terminated = True
        
        # Max steps reached
        if self.step_count >= self.max_steps:
            truncated = True
        
        observation = self._get_observation()
        info = {
            "distance": distance,
            "success": reward > 0.5,
            "step": self.step_count,
        }
        
        if self.render_mode == "human":
            self.render()
        
        return observation, reward, terminated, truncated, info
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation."""
        return np.array([
            self.agent_pos[0],
            self.agent_pos[1],
            self.target_pos[0],
            self.target_pos[1],
            0.0,  # agent velocity x (placeholder)
            0.0,  # agent velocity y (placeholder)
        ], dtype=np.float32)
    
    def render(self) -> None:
        """Render environment."""
        if self.render_mode == "human":
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            
            # Update visualization
            self.viewer.sync()
    
    def close(self) -> None:
        """Close environment."""
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
    
    def set_complexity(self, target_speed: float, floor_size: float) -> None:
        """
        Update environment complexity.
        
        Args:
            target_speed: New target speed [0, 0.89]
            floor_size: New floor size [2, 24]
        """
        self.target_speed = np.clip(target_speed, 0.0, 0.89)
        self.floor_size = np.clip(floor_size, 2.0, 24.0)
        
        # Reset model for new floor size
        self.model = self._create_model()
        self.data = mujoco.MjData(self.model)
