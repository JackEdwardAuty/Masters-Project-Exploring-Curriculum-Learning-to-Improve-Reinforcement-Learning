"""
🎥 PERFECT WINDOWS RENDERING - NO MuJoCo viewer needed!
Custom OpenGL + Matplotlib - Agent chases target LIVE!
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from stable_baselines3 import PPO
import sys
sys.path.insert(0, '.')

import matplotlib
matplotlib.use("QtAgg")  # or "QtAgg"
import matplotlib.pyplot as plt


try:
    from curriculum_env import CurriculumEnvironment
except:
    print("❌ Run training first: python windowstrain.py --total-steps 10000")
    exit()

print("🎥 LOADING MODEL...")
model = PPO.load("experiments/checkpoints/final_model.zip")
print("✅ MODEL LOADED!")

class LiveRenderer:
    def __init__(self):
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.ax.set_xlim(-6, 6)
        self.ax.set_ylim(-6, 6)
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_title("🎓 Masters Project: Agent (Red) Chases Target (Green)", fontsize=16)
        
        self.agent_circle = plt.Circle((0, 0), 0.2, color='red', label='Agent')
        self.target_circle = plt.Circle((0, 0), 0.15, color='green', label='Target')
        self.ax.add_patch(self.agent_circle)
        self.ax.add_patch(self.target_circle)
        self.ax.legend()
        
        self.rewards = []
        self.distances = []
    
    def update(self, frame):
        obs, _ = env.reset()
        total_reward = 0
        steps = 0
        done = False
        
        agent_trail = []
        target_trail = []
        
        while not done and steps < 300:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated
            
            # Update positions
            agent_pos = obs[:2]
            target_pos = obs[2:]
            
            agent_trail.append(agent_pos)
            target_trail.append(target_pos)
            
            # Update circles
            self.agent_circle.center = agent_pos
            self.target_circle.center = target_pos
            
            # Trail
            if len(agent_trail) > 1:
                agent_pts = np.array(agent_trail[-20:])
                target_pts = np.array(target_trail[-20:])
                self.ax.plot(agent_pts[:,0], agent_pts[:,1], 'r-', alpha=0.6, linewidth=2)
                self.ax.plot(target_pts[:,0], target_pts[:,1], 'g-', alpha=0.6, linewidth=2)
            
            # Stats
            distance = np.linalg.norm(agent_pos - target_pos)
            self.rewards.append(total_reward)
            self.distances.append(distance)
            
            status = "✅ SUCCESS!" if distance < 0.4 else "🔄 CHASING"
            self.ax.set_title(f"🎓 Curriculum Learning | Dist: {distance:.2f} | {status}", fontsize=14)
            
            return self.agent_circle, self.target_circle
        
        print(f"Episode Reward: {total_reward:.1f} | Steps: {steps} | Success: {total_reward > 5}")
        return self.agent_circle, self.target_circle

# Create environment (headless)
env = CurriculumEnvironment(render_mode=None, max_steps=300)

# Animate!
renderer = LiveRenderer()
ani = animation.FuncAnimation(renderer.fig, renderer.update, frames=10, interval=100, blit=True, repeat=True)

plt.tight_layout()
plt.show()

env.close()
print("🎉 PERFECT RENDERING COMPLETE!")
