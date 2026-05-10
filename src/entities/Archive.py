import numpy as np
from src.config import CONFIG
from src.entities import Solution


class Archive:
    def __init__(self, max_size: int = CONFIG.archive_size):
        self.front = []
        self.dominated = []
        self.max_size = max_size

    def update_archive(self, neighbors: list[Solution]) -> None:
        all_solutions = self.front + self.dominated + neighbors

        self.front, all_dominated = self._get_non_dominated_solutions(all_solutions)

        if len(self.front) > self.max_size:
            self.front = self._select_by_crowding_distance(self.front, self.max_size)

        self.dominated = []
        if len(self.front) < self.max_size:
            self.dominated = self._select_by_crowding_distance(all_dominated, min(self.max_size - len(self.front), len(all_dominated)))

    def _get_non_dominated_solutions(self, solutions: list[Solution]) -> tuple[list[Solution], list[Solution]]:
        if len(solutions) <= 1:
            return solutions, []
        return self._fast_non_dominated_sort(solutions)

    def _fast_non_dominated_sort(self, solutions: list[Solution]) -> tuple[list[Solution], list[Solution]]:
        n = len(solutions)
        if n <= 1:
            return solutions, []
        
        # Initialize dominance structures
        domination_count = [0] * n  # Number of solutions that dominate solution i
        dominated_solutions = [[] for _ in range(n)]  # Solutions dominated by solution i
        
        # Calculate dominance relationships - O(N²M)
        for i in range(n):
            for j in range(i + 1, n):
                if solutions[i].dominates(solutions[j]):
                    dominated_solutions[i].append(j)
                    domination_count[j] += 1
                elif solutions[j].dominates(solutions[i]):
                    dominated_solutions[j].append(i)
                    domination_count[i] += 1
        
        # Find non-dominated solutions (domination_count = 0)
        non_dominated = []
        dominated = []
        
        for i in range(n):
            if domination_count[i] == 0:
                non_dominated.append(solutions[i])
            else:
                dominated.append(solutions[i])
        
        return non_dominated, dominated

    def _select_by_crowding_distance(self, solutions: list[Solution], k: int) -> list[Solution]:
        self._assign_crowding_distance(solutions)
        solutions.sort(key=lambda s: s.crowding_distance, reverse=True)
        return solutions[:k]

    def _assign_crowding_distance(self, solutions: list[Solution]) -> None:
        num_solutions = len(solutions)
        if num_solutions == 0:
            return

        for s in solutions:
            s.crowding_distance = 0

        for i in range(len(solutions[0].score)):
            solutions.sort(key=lambda s: s.score[i])
            solutions[0].crowding_distance = float("inf")
            solutions[-1].crowding_distance = float("inf")

            max_score = solutions[-1].score[i]
            min_score = solutions[0].score[i]
            if max_score == min_score:
                continue  # Skip this objective if all scores are the same

            for j in range(1, num_solutions - 1):
                if solutions[j + 1].score[i] != solutions[j - 1].score[i]:
                    solutions[j].crowding_distance += (
                        solutions[j + 1].score[i] - solutions[j - 1].score[i]
                    ) / (max_score - min_score)
                else:
                    solutions[j].crowding_distance += 0

    def select_solution_to_optimize(self, iteration: int) -> Solution:
        # Decide whether to use front or dominated solutions
        use_front = np.random.random() < CONFIG.front_selection_prob or not self.dominated
        candidates = self._get_available_candidates(self.front if use_front else self.dominated)

        # Select based on iteration parity
        if iteration % 2 == 0:
            solution = self._select_by_reward(candidates)
        else:
            solution = self._select_by_rssi(candidates)

        solution.visited = True
        return solution

    def _get_available_candidates(self, solutions: list) -> list:
        candidates = [s for s in solutions if not s.visited]
        
        if not candidates:
            # Reset all solutions if no unvisited ones remain
            for solution in solutions:
                solution.visited = False
            candidates = solutions
            
        return candidates

    def _select_by_reward(self, candidates: list[Solution]) -> Solution:
        rewards = np.array([s.score[0] for s in candidates])
        if rewards.sum() == 0:
            return np.random.choice(candidates)
        probabilities = rewards / np.sum(rewards)
        return np.random.choice(candidates, p=probabilities)

    def _select_by_rssi(self, candidates: list[Solution]) -> Solution:
        rssi_scores = np.array([1 / s.score[1] for s in candidates])
        probabilities = rssi_scores / np.sum(rssi_scores)
        return np.random.choice(candidates, p=probabilities)
