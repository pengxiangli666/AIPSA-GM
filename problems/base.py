"""
Base class for all optimization problems.
Defines the unified problem interface used across all SA variants.
"""

from abc import ABC, abstractmethod


class Problem(ABC):
    """Abstract base class for optimization problems."""

    @abstractmethod
    def init_solution(self):
        """Generate and return a random initial solution."""
        pass

    @abstractmethod
    def neighbor(self, solution):
        """
        Generate a neighbor solution from the given solution.
        Should not modify the input solution in-place.
        Returns a new solution.
        """
        pass

    @abstractmethod
    def cost(self, solution):
        """
        Compute and return the cost (objective value) of a solution.
        Lower is better (minimization).
        """
        pass

    @abstractmethod
    def distance(self, sol_a, sol_b):
        """
        Compute a diversity/distance measure between two solutions.
        Used by guided migration to assess diversity.
        """
        pass
