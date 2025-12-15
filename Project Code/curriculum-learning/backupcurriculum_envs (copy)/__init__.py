# curriculum_env/__init__.py
from gymnasium.envs.registration import register
from .curriculum_envs import CurriculumEnv

register(
    id="CurriculumEnv-v0",
    entry_point="curriculum_env.env:CurriculumEnv",
)
