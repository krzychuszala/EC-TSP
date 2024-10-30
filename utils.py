import csv
import math
import time
from typing import Iterable

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

Solution = Iterable[int]


class TspInstance:
    def euclidean_distance(self, x1, y1, x2, y2):
        return round(math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))

    def read_tsp_instance(self, file_path):
        node_positions = []
        node_costs = []

        with open(file_path, "r") as file:
            reader = csv.reader(file, delimiter=";")
            for row in reader:
                x, y, cost = map(int, row)
                node_positions.append((x, y))
                node_costs.append(cost)

        size = len(node_positions)

        distance_matrix = np.zeros((size, size), dtype=int)
        for start in range(size):
            for end in range(start + 1, size):
                distance_matrix[start][end] = distance_matrix[end][start] = (
                    self.euclidean_distance(
                        *node_positions[start],
                        *node_positions[end],
                    )
                )

        self.node_positions = np.array(node_positions)
        self.node_costs = np.array(node_costs)
        self.size = size
        self.distance_matrix = np.array(distance_matrix)

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.read_tsp_instance(file_path)

    def get_cost(self, solution: Solution):
        cost = 0

        for path_index, node_index in enumerate(solution):
            node_cost = self.node_costs[node_index]
            cost += node_cost

            if path_index < len(solution) - 1:
                edge_from = solution[path_index]
                edge_to = solution[path_index + 1]
            else:
                edge_from = solution[-1]
                edge_to = solution[0]

            if edge_from is not None and edge_to is not None:
                edge_cost = self.distance_matrix[edge_from][edge_to]
                cost += edge_cost

        return cost

    def run_experiments(
        self, solution_getter: callable, *params
    ) -> tuple[float, float, float, Solution]:
        # start = time.time()

        min = max = best = None
        min_time = max_time = None
        avg_time = 0
        avg = 0

        for start_node in range(200):
            # show progress at the same line
            print((f"{start_node + 1}/{self.size}") + " " * 10, end="\r")

            start_iter = time.time()
            solution = solution_getter(self, start_node, *params)
            cost = self.get_cost(solution)
            
            stop_iter = time.time()
            iter_time = stop_iter - start_iter

            # Cost
            if min is None or cost < min:
                best = solution
                min = cost

            if max is None or cost > max:
                max = cost

            # Time
            if min_time is None or iter_time < min_time:
                min_time = iter_time

            if max_time is None or iter_time > max_time:
                max_time = iter_time

            avg += cost
            avg_time += iter_time

        avg /= self.size
        avg_time /= self.size

        # stop = time.time()
        # print(
        #     f"{solution_getter.__name__}: {time.strftime('%M:%S', time.gmtime(stop - start))}"
        # )

        return (min, max, avg, min_time, max_time, avg_time, best)

    def plot(self, solution: Solution, title: str):
        plt.figure(figsize=(16, 8))

        x = tuple(map(lambda x: x[0], self.node_positions))
        y = tuple(map(lambda x: x[1], self.node_positions))
        c = self.node_costs

        # Plot edges
        for i in range(len(solution) - 1):
            plt.plot(
                [
                    self.node_positions[solution[i]][0],
                    self.node_positions[solution[i + 1]][0],
                ],
                [
                    self.node_positions[solution[i]][1],
                    self.node_positions[solution[i + 1]][1],
                ],
                "k-",
                zorder=1,
            )
        plt.plot(
            [self.node_positions[solution[-1]][0], self.node_positions[solution[0]][0]],
            [self.node_positions[solution[-1]][1], self.node_positions[solution[0]][1]],
            "k-",
            zorder=1,
        )

        # Plot nodes
        plt.scatter(
            x, y, c=c, cmap=cm.cividis, s=100, zorder=2
        )  # cividis, inferno, magma, plasma, viridis, gray

        # Order Labels
        for i, node_idx in enumerate(solution):
            plt.text(
                self.node_positions[node_idx][0],
                self.node_positions[node_idx][1] + 30,
                f"{i}",
                fontsize=12,
                color="black",
                ha="center",
                zorder=3,
            )

        plt.title(title)
        plt.colorbar()
        plt.show()


def weighted_regret(tsp: TspInstance, start_node: int):
    penalty = np.zeros_like(tsp.node_costs)
    INF = int(1e9)
    penalty[start_node] = INF
    solution = [start_node, int(np.argmin(tsp.distance_matrix[start_node] + tsp.node_costs + penalty))]
    solution_size = len(solution)

    # Cache node costs and distance differences
    node_costs = tsp.node_costs
    distance_matrix = tsp.distance_matrix

    while solution_size < np.ceil(tsp.size / 2):
        selected_node_index = selected_insertion_index = max_weighted_sum = None

        for node_index in range(tsp.size):
            if node_index in solution or node_index == start_node:
                continue

            # Compute insertion costs for all positions at once
            insertion_costs = [
                node_costs[node_index]
                + distance_matrix[solution[insertion_index]][node_index]
                + distance_matrix[node_index][
                    solution[(insertion_index + 1) % solution_size]
                ]
                - distance_matrix[solution[insertion_index]][
                    solution[(insertion_index + 1) % solution_size]
                ]
                for insertion_index in range(solution_size)
            ]

            # Find the first and second minimum insertion costs
            first_min, second_min = np.partition(insertion_costs, 1)[:2]
            regret = second_min - first_min
            selected_insertion_index = np.argmin(insertion_costs)

            # Compute weighted sum
            weighted_sum = 0.5 * regret - 0.5 * first_min

            if max_weighted_sum is None or weighted_sum > max_weighted_sum:
                selected_node_index = node_index
                max_weighted_sum = weighted_sum

        # Insert the selected node into the solution
        solution.insert(selected_insertion_index + 1, selected_node_index)
        solution_size += 1

    return np.array(solution)

def random_solution(tsp: TspInstance, _: int):
    num_nodes = tsp.distance_matrix.shape[0]
    num_nodes_to_select = math.ceil(num_nodes / 2)
    selected_nodes = np.random.choice(num_nodes, num_nodes_to_select, replace=False)
    return selected_nodes

def greedy_cycle(tsp: TspInstance, start_node: int):
    penalty = np.zeros_like(tsp.node_costs)
    INF = int(1e9)
    penalty[start_node] = INF
    solution = [start_node, int(np.argmin(tsp.distance_matrix[start_node] + tsp.node_costs + penalty))]

    while len(solution) < np.ceil(tsp.size / 2):
        selected_node_index = selected_path_index = min_cost = None

        for node_index in range(tsp.size):
            if node_index in solution:
                continue

            for insertion_index in range(len(solution)):
                start = solution[insertion_index]
                end = solution[(insertion_index + 1) % len(solution)]
                cost = (
                    tsp.node_costs[node_index]
                    + tsp.distance_matrix[start][node_index]
                    + tsp.distance_matrix[node_index][end]
                    - tsp.distance_matrix[start][end]
                )

                if min_cost is None or cost < min_cost:
                    selected_node_index = node_index
                    selected_path_index = insertion_index
                    min_cost = cost

        solution.insert(selected_path_index + 1, selected_node_index)

    return np.array(solution)

def intra_route_move(tsp, solution, intra_type, steepest):
    solution = solution.copy()
    indexes = range(len(solution))
    neighborhood = [(u, v) for u in indexes for v in indexes if u != v]
    np.random.shuffle(neighborhood)

    min_cost = 0
    selected_a = selected_b = None

    for index_a, index_b in neighborhood:
        # we can draw first edge with any node, but second have to start from not node_a and not node_a+1 or node_a-1
        if intra_type != "nodes" and abs(index_a - index_b) == 1:
            continue

        a = solution[index_a]
        b = solution[index_b]

        a_prev = solution[index_a - 1]
        b_prev = solution[index_b - 1]
        if index_a - 1 < 0:
            a_prev = solution[len(solution) - 1]
        if index_b - 1 < 0:
            b_prev = solution[len(solution) - 1]

        a_next = solution[(index_a + 1) % len(solution)]
        b_next = solution[(index_b + 1) % len(solution)]

        if intra_type == "nodes":
            if abs(index_a - index_b) == 1: # Adjacent case
                if index_a < index_b:  # Case where a is directly before b
                    cost_change = (
                        tsp.distance_matrix[a_prev, b]
                        + tsp.distance_matrix[a, b_next]
                        - tsp.distance_matrix[a_prev, a]
                        - tsp.distance_matrix[b, b_next]
                    )
                else:  # Case where b is directly before a
                    cost_change = (
                        tsp.distance_matrix[b_prev, a]
                        + tsp.distance_matrix[b, a_next]
                        - tsp.distance_matrix[b_prev, b]
                        - tsp.distance_matrix[a, a_next]
                    )
            elif abs(index_a - index_b) == len(solution) - 1:  # Adjacent case - last and first node
                if index_a > index_b:
                    cost_change = (
                        tsp.distance_matrix[a_prev, b]
                        + tsp.distance_matrix[a, b_next]
                        - tsp.distance_matrix[a_prev, a]
                        - tsp.distance_matrix[b, b_next]
                    )
                else:
                    cost_change = (
                        tsp.distance_matrix[b_prev, a]
                        + tsp.distance_matrix[b, a_next]
                        - tsp.distance_matrix[b_prev, b]
                        - tsp.distance_matrix[a, a_next]
                    )
            else:
                # Non-adjacent case, use the general formula
                cost_change = (
                    tsp.distance_matrix[a_prev, b]
                    + tsp.distance_matrix[b, a_next]
                    + tsp.distance_matrix[b_prev, a]
                    + tsp.distance_matrix[a, b_next]
                    - tsp.distance_matrix[a_prev, a]
                    - tsp.distance_matrix[a, a_next]
                    - tsp.distance_matrix[b_prev, b]
                    - tsp.distance_matrix[b, b_next]
                )

        else:
            cost_change = (
                tsp.distance_matrix[a, b]
                + tsp.distance_matrix[a_next, b_next]
                - tsp.distance_matrix[a, a_next]
                - tsp.distance_matrix[b, b_next]
            )

        if cost_change < min_cost:
            min_cost = cost_change
            selected_a = index_a
            selected_b = index_b

            if not steepest:
                break

    if selected_a is not None:
        if intra_type == "nodes":
            # Just swap the two nodes
            solution[selected_a], solution[selected_b] = (
                solution[selected_b],
                solution[selected_a],
            )
        else:
            if selected_a < selected_b:
                subsequence = solution[(selected_a + 1):selected_b+1][::-1]
                solution[(selected_a + 1):selected_b+1] = subsequence
            else:
                subsequence = solution[(selected_b + 1):selected_a+1][::-1]
                solution[(selected_b + 1):selected_a+1] = subsequence

    return solution, min_cost


def inter_route_move(tsp, solution, steepest):
    import itertools
    solution = solution.copy()
    possible_unselect = set(solution)
    possible_select = set(range(tsp.size)) - possible_unselect
    neighborhood = list(itertools.product(possible_unselect, possible_select))
    np.random.shuffle(neighborhood)

    min_cost = 0
    to_unselect = to_select = None

    for test_unselect, test_select in neighborhood:
        insert_index = np.where(solution == test_unselect)[0][0]

        prev = solution[insert_index - 1]
        next = solution[(insert_index + 1) % len(solution)]

        cost_change = (
            tsp.node_costs[test_select]
            + tsp.distance_matrix[prev, test_select]
            + tsp.distance_matrix[test_select, next]
            - tsp.node_costs[test_unselect]
            - tsp.distance_matrix[prev, test_unselect]
            - tsp.distance_matrix[test_unselect, next]
        )

        if cost_change < min_cost:
            min_cost = cost_change
            to_unselect = test_unselect
            to_select = test_select

            if not steepest:
                break

    if to_unselect is not None and not steepest:
        solution[np.where(solution == to_unselect)] = to_select

    if steepest:
        return min_cost, to_unselect, to_select
    else:
        return solution, min_cost

def local_search(
    tsp: TspInstance,
    start_node: int,
    initial_solution_getter: callable,
    intra_type: str,
    steepest: bool,
):
    solution = initial_solution_getter(tsp, start_node)
 
    while True:
        
        if steepest:
            cost_change_1, to_unselect, to_select = inter_route_move(tsp, solution, steepest)
            solution_2, cost_change_2 = intra_route_move(
                tsp, solution, intra_type, steepest
            )
            if cost_change_1 < cost_change_2:
                cost_change = cost_change_1
                if to_unselect is not None:
                    solution[np.where(solution == to_unselect)] = to_select
            else:
                solution, cost_change = solution_2, cost_change_2
        else:
            if np.random.rand() < 0.5:
                solution, cost_change = inter_route_move(tsp, solution, steepest)

                if cost_change == 0:
                    solution, cost_change = intra_route_move(
                        tsp, solution, intra_type, steepest
                    )
            else:
                solution, cost_change = intra_route_move(
                    tsp, solution, intra_type, steepest
                )

                if cost_change == 0:
                    solution, cost_change = inter_route_move(tsp, solution, steepest)

        if cost_change == 0:
            break
        
    return solution
