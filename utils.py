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
        self, solution_getter: callable, cycle=False
    ) -> tuple[float, float, float, Solution]:
        start = time.time()

        min = max = best = None

        avg = 0

        for start_node in range(self.size):
            solution = solution_getter(self, start_node)
            cost = self.get_cost(solution)

            if min is None or cost < min:
                best = solution
                min = cost

            if max is None or cost > max:
                max = cost

            avg += cost

        avg /= self.size

        stop = time.time()
        print(
            f"{solution_getter.__name__}: {time.strftime('%M:%S', time.gmtime(stop - start))}"
        )

        return (min, max, avg, best)

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
