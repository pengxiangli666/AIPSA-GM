
import random
import math
import numpy as np
from problems.base import Problem


class TSP(Problem):
    def __init__(self, n_cities=50, seed=None, coords=None):
        """
        Args:
            n_cities: number of cities (ignored if coords provided)
            seed: random seed for reproducibility
            coords: optional list of (x, y) tuples; if None, generated randomly
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        if coords is not None:
            self.coords = np.array(coords)
            self.n_cities = len(coords)
        else:
            self.n_cities = n_cities
            self.coords = np.random.rand(n_cities, 2) * 1000  # cities in [0,1000]^2

        # Precompute distance matrix (vectorized)
        self.dist_matrix = self._compute_dist_matrix()

    def _compute_dist_matrix(self):
        """
        Precompute pairwise Euclidean distances using numpy broadcasting.
        Much faster than the double for-loop for large n_cities.
        """
        c = self.coords
        # Shape: (n, 1, 2) - (1, n, 2) => (n, n, 2)
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dist = np.sqrt((diff ** 2).sum(axis=2))
        return dist

    def init_solution(self):
        """Return a random permutation of cities."""
        tour = list(range(self.n_cities))
        random.shuffle(tour)
        return tour

    def neighbor(self, solution):
        """
        2-opt swap: reverse a random sub-segment of the tour.
        Returns a new solution (does not modify input).
        """
        tour = solution[:]
        i, j = sorted(random.sample(range(self.n_cities), 2))
        tour[i:j+1] = reversed(tour[i:j+1])
        return tour

    def cost(self, solution):
        """Total tour length (including return to start)."""
        s = np.array(solution)
        # Use numpy indexing for fast cost computation
        return float(self.dist_matrix[s, np.roll(s, -1)].sum())

    def distance(self, sol_a, sol_b):
        """
        Edge-based distance: fraction of edges in sol_a not present in sol_b.
        Range: [0, 1]. Higher = more diverse.
        """
        n = len(sol_a)
        edges_a = set()
        edges_b = set()
        for k in range(n):
            u, v = sol_a[k], sol_a[(k + 1) % n]
            edges_a.add((min(u, v), max(u, v)))
            u, v = sol_b[k], sol_b[(k + 1) % n]
            edges_b.add((min(u, v), max(u, v)))
        shared = len(edges_a & edges_b)
        return 1.0 - shared / n