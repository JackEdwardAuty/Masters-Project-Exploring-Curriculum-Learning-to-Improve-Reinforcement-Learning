"""
Curriculum Learning Progression Functions

Implements fixed and adaptive progression strategies for controlling
task difficulty during training.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Callable


class ProgressionFunction(ABC):
    """Base class for progression functions."""
    
    @abstractmethod
    def __call__(
        self,
        timestep: int,
        previous_complexity: Optional[float] = None,
        agent_performance: Optional[float] = None,
    ) -> float:
        """
        Calculate complexity factor for current timestep.
        
        Returns:
            Complexity factor in range [0, 1]
        """
        pass
    
    def reset(self) -> None:
        """Reset internal state if needed."""
        pass


class LinearProgression(ProgressionFunction):
    """
    Fixed linear progression function.
    
    Πl(t, te) = max(t/te, 1)
    
    Complexity increases linearly from 0 to 1 over training period.
    """
    
    def __init__(self, total_steps: int):
        """
        Initialize linear progression.
        
        Args:
            total_steps: Total training steps to reach maximum complexity
        """
        self.total_steps = total_steps
    
    def __call__(
        self,
        timestep: int,
        previous_complexity: Optional[float] = None,
        agent_performance: Optional[float] = None,
    ) -> float:
        """Linear complexity increase."""
        complexity = min(timestep / self.total_steps, 1.0)
        return float(complexity)


class ExponentialProgression(ProgressionFunction):
    """
    Fixed exponential progression function.
    
    Πe(t, te, s) = 1 - max((α - β) / (1 - β), 0)
    where:
        α = exp(-t / (te * s))
        β = exp(-1/s)
    
    Complexity increases exponentially. Parameter s controls curve shape:
    - Small s: steep initial increase (fast early difficulty)
    - s = 2: defaults to linear
    - Large s: shallow curve (slow difficulty increase)
    """
    
    def __init__(self, total_steps: int, slope: float = 1.0):
        """
        Initialize exponential progression.
        
        Args:
            total_steps: Total training steps to reach maximum complexity
            slope: Shape parameter s (default 1.0 for steep curve)
        """
        self.total_steps = total_steps
        self.slope = slope
        
        # Pre-calculate beta
        self.beta = np.exp(-1.0 / self.slope)
    
    def __call__(
        self,
        timestep: int,
        previous_complexity: Optional[float] = None,
        agent_performance: Optional[float] = None,
    ) -> float:
        """Exponential complexity increase."""
        # Calculate alpha
        alpha = np.exp(-timestep / (self.total_steps * self.slope))
        
        # Calculate complexity
        complexity = 1.0 - max((alpha - self.beta) / (1.0 - self.beta), 0.0)
        return float(np.clip(complexity, 0.0, 1.0))


class FrictionBasedProgression(ProgressionFunction):
    """
    Adaptive friction-based progression function (from Bassich et al. 2020).
    
    Πf(t, c_{t-1}) = 1 - Uniform(s_t, s_min)
    
    Adapts complexity based on agent performance:
    - If agent is learning well: increase complexity faster
    - If agent is struggling: keep complexity stable
    - Uses "friction" concept inspired by physics
    
    The performance metric is used to adjust progression speed dynamically.
    """
    
    def __init__(
        self,
        total_steps: int,
        min_performance: float = 0.1,
        friction_coeff: float = 0.5,
    ):
        """
        Initialize friction-based adaptive progression.
        
        Args:
            total_steps: Total training steps
            min_performance: Minimum performance threshold [0, 1]
            friction_coeff: Friction coefficient controlling adaptation rate
        """
        self.total_steps = total_steps
        self.min_performance = min_performance
        self.friction_coeff = friction_coeff
        
        self.current_complexity = 0.0
        self.performance_history = []
    
    def __call__(
        self,
        timestep: int,
        previous_complexity: Optional[float] = None,
        agent_performance: Optional[float] = None,
    ) -> float:
        """
        Adaptive complexity adjustment based on performance.
        
        Args:
            timestep: Current training step
            previous_complexity: Previous complexity value
            agent_performance: Agent's current performance metric [0, 1]
        
        Returns:
            New complexity factor
        """
        if previous_complexity is None:
            previous_complexity = 0.0
        
        if agent_performance is None:
            agent_performance = 0.5
        
        # Track performance history
        self.performance_history.append(agent_performance)
        
        # Calculate moving average of recent performance (last 100 steps)
        window = min(100, len(self.performance_history))
        avg_performance = np.mean(self.performance_history[-window:])
        
        # Determine if agent is learning well
        is_learning_well = avg_performance > self.min_performance
        
        # Calculate progression speed based on performance
        if is_learning_well:
            # Agent learning: increase complexity
            progression_speed = 1.0 + self.friction_coeff * (avg_performance - self.min_performance)
        else:
            # Agent struggling: slow down
            progression_speed = 1.0 - self.friction_coeff * (self.min_performance - avg_performance)
        
        # Update complexity with speed factor
        max_complexity = min(timestep / self.total_steps, 1.0)
        
        # Smooth update: gradually move toward max complexity
        new_complexity = (
            previous_complexity * progression_speed +
            max_complexity * (1.0 - progression_speed / 2.0)
        )
        
        self.current_complexity = np.clip(new_complexity, 0.0, 1.0)
        
        return float(self.current_complexity)
    
    def reset(self) -> None:
        """Reset internal state."""
        self.current_complexity = 0.0
        self.performance_history = []
    
    def get_statistics(self) -> dict:
        """Get progression statistics."""
        if not self.performance_history:
            return {}
        
        return {
            "avg_performance": float(np.mean(self.performance_history)),
            "max_performance": float(np.max(self.performance_history)),
            "min_performance": float(np.min(self.performance_history)),
            "current_complexity": float(self.current_complexity),
        }


class MappingFunction:
    """
    Maps complexity factor [0, 1] to environment parameters.
    
    For the curriculum environment:
    Target Speed: [0, 0.89]
    Floor Size: [2, 24]
    """
    
    def __call__(self, complexity: float) -> tuple:
        """
        Map complexity to task parameters.
        
        Args:
            complexity: Complexity factor [0, 1]
        
        Returns:
            (target_speed, floor_size)
        """
        complexity = np.clip(complexity, 0.0, 1.0)
        
        # Linear mapping to parameter ranges
        target_speed = 0.89 * complexity
        floor_size = 22.0 * complexity + 2.0
        
        return float(target_speed), float(floor_size)


class MultiProcessProgression:
    """
    Manages independent progression functions for parallel training.
    
    Each process can have a different progression function (e.g., different
    slopes for exponential) to create diverse curriculum variations.
    """
    
    def __init__(
        self,
        num_processes: int,
        progression_type: str = "exponential",
        total_steps: int = 1000000,
        **kwargs
    ):
        """
        Initialize multi-process progression.
        
        Args:
            num_processes: Number of parallel processes
            progression_type: "linear", "exponential", or "friction"
            total_steps: Total training steps
            **kwargs: Additional arguments for progression function
        """
        self.num_processes = num_processes
        self.progression_type = progression_type
        self.total_steps = total_steps
        
        # Create independent progression functions
        self.progressions = []
        
        if progression_type == "linear":
            for _ in range(num_processes):
                self.progressions.append(LinearProgression(total_steps))
        
        elif progression_type == "exponential":
            # Different slopes for each process
            slopes = kwargs.get("slopes", None)
            if slopes is None:
                # Create diverse slopes: [0.1, 0.73, 1.37, 2.0]
                slopes = np.linspace(0.1, 2.0, num_processes)
            
            for slope in slopes:
                self.progressions.append(
                    ExponentialProgression(total_steps, slope=float(slope))
                )
        
        elif progression_type == "friction":
            for _ in range(num_processes):
                self.progressions.append(
                    FrictionBasedProgression(
                        total_steps,
                        min_performance=kwargs.get("min_performance", 0.1),
                        friction_coeff=kwargs.get("friction_coeff", 0.5),
                    )
                )
        
        else:
            raise ValueError(f"Unknown progression type: {progression_type}")
        
        self.mapping = MappingFunction()
    
    def __call__(
        self,
        timestep: int,
        process_id: int,
        agent_performance: Optional[float] = None,
    ) -> tuple:
        """
        Get task parameters for a specific process.
        
        Args:
            timestep: Current training step
            process_id: Which process [0, num_processes)
            agent_performance: Optional performance metric for adaptive functions
        
        Returns:
            (target_speed, floor_size)
        """
        if process_id < 0 or process_id >= len(self.progressions):
            raise ValueError(f"Invalid process_id: {process_id}")
        
        progression = self.progressions[process_id]
        
        # Get complexity from progression function
        if process_id == 0:
            # First process doesn't use previous complexity
            complexity = progression(timestep, agent_performance=agent_performance)
        else:
            # Subsequent processes use their own history
            prev_complexity = getattr(progression, "current_complexity", 0.0)
            complexity = progression(
                timestep,
                previous_complexity=prev_complexity,
                agent_performance=agent_performance
            )
        
        # Map to parameters
        return self.mapping(complexity)
    
    def reset(self) -> None:
        """Reset all progression functions."""
        for prog in self.progressions:
            prog.reset()


# Convenience factory function
def get_progression_function(
    progression_type: str,
    total_steps: int = 1000000,
    **kwargs
) -> ProgressionFunction:
    """
    Factory function to create progression functions.
    
    Args:
        progression_type: "linear", "exponential", or "friction"
        total_steps: Total training steps
        **kwargs: Function-specific arguments
    
    Returns:
        Progression function instance
    """
    if progression_type == "linear":
        return LinearProgression(total_steps)
    
    elif progression_type == "exponential":
        slope = kwargs.get("slope", 1.0)
        return ExponentialProgression(total_steps, slope=slope)
    
    elif progression_type == "friction":
        return FrictionBasedProgression(
            total_steps,
            min_performance=kwargs.get("min_performance", 0.1),
            friction_coeff=kwargs.get("friction_coeff", 0.5),
        )
    
    else:
        raise ValueError(f"Unknown progression type: {progression_type}")
