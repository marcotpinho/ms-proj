from tqdm import tqdm
import time

from src.entities.Archive import Archive
from src.entities.Evaluator import Evaluator
from src.entities.Map import Map
from src.entities.Neighborhood import Neighborhood
from src.entities.Solution import Solution
from src.operators import local_search
from src.operators.perturb_solution import perturb_solution

class Runner:
    def __init__(self,
                 max_time: float,
                 max_iterations: int,
                 archive: Archive = None,
                 neighborhood: Neighborhood = None,
                 map: Map = None,
                 evaluator: Evaluator = None):
        self.archive = archive
        self.map = map
        self.max_iterations = max_iterations
        self.max_time = max_time
        self.neighborhood = neighborhood
        self.evaluator = evaluator
        self.log = []

    def run(self):
        start_time = time.perf_counter()

        initial_solution = Solution(distmx=self.map.distmx, rvalues=self.map.rvalues)
        self.evaluator.evaluate(initial_solution)
        for neighborhood_id in tqdm(range(self.neighborhood.num_neighborhoods), desc="Initial local search", unit="neighborhood", dynamic_ncols=True):
            neighbors = local_search(
                initial_solution,
                self.neighborhood,
                neighborhood_id
            )
            self.evaluator.evaluate(neighbors)
            self.archive.update_archive(neighbors)

        progress_bar = tqdm(total=self.max_iterations, desc="Progress", unit="it", dynamic_ncols=True)
        iteration = 0

        while iteration < self.max_iterations and time.perf_counter() - start_time < self.max_time:
            solution = self.archive.select_solution_to_optimize(iteration)

            for neighborhood_id in range(self.neighborhood.num_neighborhoods):
                if time.perf_counter() - start_time > self.max_time:
                    break

                perturbed_solution = perturb_solution(
                    solution,
                    self.neighborhood,
                )
                neighbors = local_search(
                    perturbed_solution,
                    self.neighborhood,
                    neighborhood_id
                )

                new_solutions = [perturbed_solution] + neighbors
                self.evaluator.evaluate(new_solutions)

                self.archive.update_archive([perturbed_solution] + neighbors)

                self.save_statistics()

            iteration += 1
            progress_bar.update(1)

        progress_bar.close()

        elapsed_time = time.perf_counter() - start_time
        return self.log, elapsed_time

    def save_statistics(self) -> None:
        if not self.archive.front:
            return

        max_reward = max(solution.score[0] for solution in self.archive.front)
        max_rssi = max(solution.score[1] for solution in self.archive.front)
        self.log.append([max_reward, max_rssi])