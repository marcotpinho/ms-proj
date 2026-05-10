from ..entities import Solution, Neighborhood


def perturb_solution(
    solution: Solution,
    neighborhood: Neighborhood,
) -> Solution:
    perturb_operator = neighborhood.get_perturbation_operator()
    new_solution = perturb_operator(solution)

    return new_solution