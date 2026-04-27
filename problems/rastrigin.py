import random
import math
import numpy as np
from problems.base import Problem


class Rastrigin(Problem):
    def __init__(self, n_dims=10, sigma=0.1, bounds=(-5.12, 5.12), seed=None):
        """
        Args:
            n_dims: number of dimensions
            sigma: std dev for Gaussian perturbation neighbor
            bounds: (lower, upper) bounds for each dimension
            seed: random seed
        """
        self.n_dims = n_dims
        self.sigma = sigma
        self.bounds = bounds
        self.A = 10  # Rastrigin constant

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def init_solution(self):
        """Uniform random initialization within bounds."""
        low, high = self.bounds
        return list(np.random.uniform(low, high, self.n_dims))

    def neighbor(self, solution):
        """
        Gaussian perturbation: add N(0, sigma) noise to each dimension,
        then clip to bounds.
        """
        low, high = self.bounds
        perturbed = [
            max(low, min(high, x + random.gauss(0, self.sigma)))
            for x in solution
        ]
        return perturbed

    def cost(self, solution):
        """
        Rastrigin function: f(x) = A*n + sum(x_i^2 - A*cos(2*pi*x_i))
        Global minimum = 0 at x = [0, ..., 0]
        """
        n = self.n_dims
        A = self.A
        total = A * n
        for x in solution:
            total += x ** 2 - A * math.cos(2 * math.pi * x)
        return total

    def distance(self, sol_a, sol_b):
        """
        Normalized Euclidean distance between two solutions.
        Normalized by the diameter of the search space.
        """
        low, high = self.bounds
        max_dist = math.sqrt(self.n_dims) * (high - low)
        euclidean = math.sqrt(sum((a - b) ** 2 for a, b in zip(sol_a, sol_b)))
        return euclidean / max_dist