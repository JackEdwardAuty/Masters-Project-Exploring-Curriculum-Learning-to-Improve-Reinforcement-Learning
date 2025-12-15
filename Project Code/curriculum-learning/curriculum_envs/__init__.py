# curriculum_env/__init__.py
from gymnasium.envs.registration import register
from .curriculum_env import CurriculumEnvironment

register(
    id="CurriculumEnv-v0",
    entry_point="curriculum_envs.curriculum_env:CurriculumEnvironment",
)
