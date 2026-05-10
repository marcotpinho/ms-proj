import numpy as np
from ..entities import Solution, Neighborhood


def local_search(
    solution: Solution,
    neighborhood: Neighborhood,
    neighborhood_id: int,
) -> Solution:
    neighbors = []

    for agent in range(len(solution.paths)):
        local_search_operator = neighborhood.get_local_search_operator(neighborhood_id)
        neighbors.extend(local_search_operator(solution, agent))
    return neighbors