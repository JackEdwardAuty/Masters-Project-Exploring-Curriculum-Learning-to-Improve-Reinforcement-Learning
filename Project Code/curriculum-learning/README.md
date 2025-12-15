# Curriculum Learning for Reinforcement Learning - Complete Codebase

A comprehensive implementation of **Curriculum Learning** applied to **Reinforcement Learning**, demonstrating how training agents on progressively harder tasks improves convergence speed and final performance compared to traditional RL approaches.

## Project Overview

This project explores curriculum learning techniques using:
- **Environment**: Modified OpenAI Multi-Agent Emergence Environment (simplified to single-agent)
- **Algorithm**: Proximal Policy Optimization (PPO) via Stable Baselines 3
- **Physics Engine**: MuJoCo
- **Progression Functions**: Linear, Exponential, and Friction-Based (Adaptive)

### Key Results

From the original research:
- **Friction-Based Progression**: ~80% success rate
- **Exponential Progression**: ~70% success rate  
- **No Curriculum (Baseline)**: ~18% success rate

This demonstrates the **massive improvement** curriculum learning provides!

## Project Structure

```
curriculum-learning/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── config/
│   ├── __init__.py
│   └── config.yaml                    # Training configuration
├── src/
│   ├── __init__.py
│   ├── environment/
│   │   ├── __init__.py
│   │   ├── curriculum_env.py          # Main environment with curriculum support
│   │   └── base_env.py                # Base environment class (MuJoCo-based)
│   ├── progression/
│   │   ├── __init__.py
│   │   ├── progression_functions.py   # Linear, Exponential, Friction-Based
│   │   └── mapping_functions.py       # Complexity to task mapping
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py                 # Training loop orchestrator
│   │   ├── callbacks.py               # Custom callbacks for monitoring
│   │   └── parallel_training.py       # Parallel environment execution
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── logger.py                  # Experiment logging
│   │   ├── plotting.py                # Visualization utilities
│   │   └── helpers.py                 # Helper functions
│   └── models/
│       ├── __init__.py
│       └── ppo_agent.py               # PPO agent wrapper
├── scripts/
│   ├── train.py                       # Main training script
│   ├── evaluate.py                    # Evaluation script
│   └── plot_results.py                # Plot training results
└── experiments/
    ├── results/                       # Training results and logs
    └── checkpoints/                   # Model checkpoints
```

## Installation

1. **Clone the repository**:
```bash
git clone <your-repo>
cd curriculum-learning
```

2. **Create a virtual environment**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Note about MuJoCo**: MuJoCo is now free! However, you still need to set up your MuJoCo key:
   - Get it from: https://www.deepmind.com/documents/314/mujoco_license.txt
   - Store in `~/.mujoco/mjkey.txt`

## Usage

### Basic Training with Curriculum Learning

```bash
# Train with Friction-Based (Adaptive) progression
python scripts/train.py --progression friction --num_processes 4 --total_steps 1000000

# Train with Exponential progression
python scripts/train.py --progression exponential --slope 0.73 --num_processes 4

# Train without curriculum (baseline)
python scripts/train.py --progression none --num_processes 4
```

### Configuration

Edit `config/config.yaml` to customize:
- Training hyperparameters (learning rate, batch size, epochs)
- Environment parameters (floor size, target speed)
- Progression function settings
- Parallel execution parameters

### Evaluation

```bash
# Evaluate trained model
python scripts/evaluate.py --model experiments/checkpoints/best_model.zip --episodes 100

# Plot training curves
python scripts/plot_results.py --log-dir experiments/results/
```

## Key Components

### 1. Curriculum Environment (`src/environment/curriculum_env.py`)

Handles:
- Single-agent seeker tracking a moving target
- Complexity parameters: floor size, target speed
- Reward: +1 if target found, 0 otherwise
- Episode termination: target found or max steps (100)

### 2. Progression Functions (`src/progression/progression_functions.py`)

**Linear Progression**:
```
Π_l(t, t_e) = max(t/t_e, 1)
```

**Exponential Progression**:
```
Π_e(t, t_e, s) = 1 - max((α - β) / (1 - β), 0)
where α = exp(-t / (t_e * s)), β = exp(-1/s)
```

**Friction-Based (Adaptive)**:
```
Π_f(t, c_{t-1}) = 1 - Uniform(s_t, s_min)
Adapts based on agent performance
```

### 3. Mapping Function (`src/progression/mapping_functions.py`)

Maps complexity factor (0 to 1) to task parameters:
```
M_t(s, f) = Φ(c_t) = (0.89 * c_t, 22 * c_t + 2)
Target Speed: [0, 0.89]
Floor Size: [2, 24]
```

### 4. Training Pipeline (`src/training/trainer.py`)

Orchestrates:
- Parallel environment execution
- PPO training with multiple processes
- Progression function updates
- Logging and checkpointing

## Experimental Results

### Performance Comparison (from original project)

| Method | Final Success Rate | Convergence Speed | Notes |
|--------|------------------|-------------------|-------|
| Friction-Based | ~80% | Fast | Adaptive to agent performance |
| Exponential | ~70% | Medium | Pre-determined curriculum |
| No Curriculum | ~18% | Very Slow | Baseline - maximum difficulty |

### Key Observations

1. **Curriculum Learning is Effective**: 4-5x improvement over baseline
2. **Adaptive > Fixed**: Friction-Based outperforms Exponential by ~10%
3. **Faster Convergence**: Curriculum methods reach peak performance in ~half the time
4. **Transfer Learning**: Knowledge transfers across difficulty levels

## Advanced Usage

### Custom Progression Function

```python
from src.progression import ProgressionFunction

class CustomProgression(ProgressionFunction):
    def __call__(self, timestep, previous_complexity):
        # Your logic here
        return new_complexity
```

### Multi-Agent Extension

The codebase is designed for easy extension to multiple agents. Uncomment multi-agent sections in `curriculum_env.py`.

### Custom Reward Functions

Modify the `get_reward()` method in `CurriculumEnvironment` to implement:
- Distance-based rewards
- Time penalties
- Multi-objective rewards
- Sparse vs. dense rewards

## Troubleshooting

### MuJoCo Installation Issues
```bash
# Try alternative installation
pip install mujoco dm-control
```

### Out of Memory with Parallel Training
Reduce `num_processes` in config or command line:
```bash
python scripts/train.py --num_processes 2
```

### Training Divergence
- Reduce learning rate: `--learning_rate 0.0001`
- Increase batch size: `--batch_size 2048`
- Use shorter progression: `--progression_steps 100000`

## Project Status

✅ Core implementation complete
✅ Friction-based progression working
✅ Parallel training support
✅ Results matching original paper
⚠️ Multi-agent version in development
⚠️ Advanced obstacle support planned

## References

1. **Curriculum Learning Paper**: Bassich et al., "Curriculum learning with a progression function" (2020) [arXiv:2008.00511](https://arxiv.org/abs/2008.00511)

2. **Multi-Agent Emergence**: Baker et al., "Emergent tool use from multi-agent autocurricula" (2020)

3. **PPO Algorithm**: Schulman et al., "Proximal Policy Optimization Algorithms" (2017)

## License

This project is for educational and research purposes. Uses open-source dependencies.

## Contributing

Feel free to:
- Open issues for bugs or questions
- Submit pull requests for enhancements
- Share results and findings
- Suggest improvements

## Contact

For questions about the implementation or curriculum learning concepts, reach out!

---

**Happy Training! 🎓🚀**
