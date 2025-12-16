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
    from curriculum_envs import CurriculumEnvironment
except:
    print("❌ Run training first: python train.py --total-steps 100000")
    # print("or enter path for model")
    exit()

MODEL_PATH = "experiments/checkpoints/final_model.zip"

print("🎥 LOADING MODEL...")
model = PPO.load(MODEL_PATH)
print("✅ MODEL LOADED!")

class LiveRenderer:
    def __init__(self):
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.ax.set_xlim(-6, 6)
        self.ax.set_ylim(-6, 6)
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)

        self.title_text = self.ax.set_title("🎓 Masters Project: Agent (Red) Chases Target (Green)", fontsize=16)
        # mark as animated for blitting
        # self.title_text.set_animated(True)

        # self.ax.set_title("AI Masters Project: Agent (Red) Chases Target (Green)", fontsize=16)
        
        self.agent_circle = plt.Circle((0, 0), 0.2, color='brown', label='Agent')
        self.target_circle = plt.Circle((0, 0), 0.15, color='blue', label='Target')
        # self.agent_circle.set_animated(True)
        # self.target_circle.set_animated(True)

        self.ax.add_patch(self.agent_circle)
        self.ax.add_patch(self.target_circle)

        self.ax.legend()
        
        self.rewards = []
        self.distances = []

        # log and visualise
        self.episode_idx = 0
        self.steps_in_episode = 0

        # initial reset
        self.obs, info = env.reset()
        self._handle_episode_start(info)

    def _handle_episode_start(self, info):
        self.episode_idx += 1
        self.steps_in_episode = 0
        # visual cue for new round
        self.ax.set_facecolor("#333366")
        # self.ax.set_title(f"NEW ROUND {self.episode_idx}", fontsize=16)
        self.title_text.set_text(f"NEW ROUND {self.episode_idx}")
        print(f"=== EPISODE {self.episode_idx} START ===")

    def update(self, frame):
        # one step per animation frame
        action, _ = model.predict(self.obs, deterministic=True)
        self.obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        self.steps_in_episode += 1

        agent_pos = self.obs[:2]
        target_pos = self.obs[2:]

        self.agent_circle.center = agent_pos
        self.target_circle.center = target_pos

        distance = np.linalg.norm(agent_pos - target_pos)
        status = "✅ SUCCESS!" if distance < 0.4 else "🔄 CHASING"

        # after a couple of frames, revert background so the flash is visible
        print()
        print(f"+=+ STEP {self.steps_in_episode} RENDER +=+")
        if self.steps_in_episode <= 3:
            # self.agent_circle.set_color("yellow")
            self.target_circle.set_color("blue")
            self.target_circle.set_radius(0.20)

            self.target_circle.set_edgecolor("cyan")
            self.target_circle.set_linewidth(2.0)

            self.ax.set_facecolor("black")
            prefix = f"NEW ROUND {self.episode_idx}"
        else:
            self.ax.set_facecolor("#333366")

            prefix = f"Ep {self.episode_idx}"

            self.agent_circle.set_color("red")
            self.target_circle.set_color("blue")
            self.target_circle.set_radius(0.15)

        self.title_text.set_text(
            f"{prefix} | Dist: {distance:.2f} | {status}"
        )


        if done:
            self.obs, info = env.reset()
            self._handle_episode_start(info)

        return self.agent_circle, self.target_circle, self.title_text


env = CurriculumEnvironment(render_mode=None, max_steps=300)


renderer = LiveRenderer()
ani = animation.FuncAnimation(
    renderer.fig,
    renderer.update,
    interval=150,
    blit=False,
)
plt.show()



env.close()
print("🎉 PERFECT RENDERING COMPLETE!")
