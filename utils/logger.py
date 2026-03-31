"""
Logging utility for SA experiments.
Records: iteration, wall-clock time, current cost, best cost, acceptance rate.
"""

import time
import csv
import os


class SALogger:
    def __init__(self, log_file=None, island_id=None):
        """
        Args:
            log_file: path to CSV log file (optional)
            island_id: identifier for parallel islands (optional)
        """
        self.log_file = log_file
        self.island_id = island_id
        self.records = []
        self.start_time = None

        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True) if os.path.dirname(log_file) else None
            with open(log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                header = ['iteration', 'elapsed_time', 'temperature',
                          'current_cost', 'best_cost', 'acceptance_rate']
                if island_id is not None:
                    header.insert(0, 'island_id')
                writer.writerow(header)

    def start(self):
        self.start_time = time.time()

    def log(self, iteration, temperature, current_cost, best_cost, acceptance_rate):
        elapsed = time.time() - self.start_time if self.start_time else 0.0
        record = {
            'iteration': iteration,
            'elapsed_time': round(elapsed, 4),
            'temperature': round(temperature, 6),
            'current_cost': round(current_cost, 6),
            'best_cost': round(best_cost, 6),
            'acceptance_rate': round(acceptance_rate, 4),
        }
        if self.island_id is not None:
            record['island_id'] = self.island_id
        self.records.append(record)

        if self.log_file:
            with open(self.log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                row = [record['iteration'], record['elapsed_time'],
                       record['temperature'], record['current_cost'],
                       record['best_cost'], record['acceptance_rate']]
                if self.island_id is not None:
                    row.insert(0, self.island_id)
                writer.writerow(row)

    def get_records(self):
        return self.records
