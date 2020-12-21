import os
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from environments.loggers import getLogger
from environments.cube3 import Cube3State, Cube3 as Cube3Environment

MOVES_DEFINITIONS = [
    [33, 1, 2, 30, 4, 5, 27, 7, 8, 0, 10, 11, 3, 13, 14, 6, 16, 17, 15, 19, 20, 12, 22, 23, 9, 25, 26, 18, 28, 29, 21, 31, 32, 24, 34, 35, 38, 41, 44, 37, 40, 43, 36, 39, 42, 45, 46, 47, 48, 49, 50, 51, 52, 53],
    [9, 1, 2, 12, 4, 5, 15, 7, 8, 24, 10, 11, 21, 13, 14, 18, 16, 17, 27, 19, 20, 30, 22, 23, 33, 25, 26, 6, 28, 29, 3, 31, 32, 0, 34, 35, 42, 39, 36, 43, 40, 37, 44, 41, 38, 45, 46, 47, 48, 49, 50, 51, 52, 53],
    [0, 1, 35, 3, 4, 32, 6, 7, 29, 9, 10, 2, 12, 13, 5, 15, 16, 8, 18, 19, 17, 21, 22, 14, 24, 25, 11, 27, 28, 20, 30, 31, 23, 33, 34, 26, 36, 37, 38, 39, 40, 41, 42, 43, 44, 51, 48, 45, 52, 49, 46, 53, 50, 47],
    [0, 1, 11, 3, 4, 14, 6, 7, 17, 9, 10, 26, 12, 13, 23, 15, 16, 20, 18, 19, 29, 21, 22, 32, 24, 25, 35, 27, 28, 8, 30, 31, 5, 33, 34, 2, 36, 37, 38, 39, 40, 41, 42, 43, 44, 47, 50, 53, 46, 49, 52, 45, 48, 51],
    [2, 5, 8, 1, 4, 7, 0, 3, 6, 9, 10, 11, 12, 13, 14, 45, 48, 51, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 38, 41, 44, 36, 37, 17, 39, 40, 16, 42, 43, 15, 35, 46, 47, 34, 49, 50, 33, 52, 53],
    [6, 3, 0, 7, 4, 1, 8, 5, 2, 9, 10, 11, 12, 13, 14, 44, 41, 38, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 51, 48, 45, 36, 37, 33, 39, 40, 34, 42, 43, 35, 15, 46, 47, 16, 49, 50, 17, 52, 53],
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 47, 50, 53, 12, 13, 14, 15, 16, 17, 20, 23, 26, 19, 22, 25, 18, 21, 24, 36, 39, 42, 30, 31, 32, 33, 34, 35, 11, 37, 38, 10, 40, 41, 9, 43, 44, 45, 46, 29, 48, 49, 28, 51, 52, 27],
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 42, 39, 36, 12, 13, 14, 15, 16, 17, 24, 21, 18, 25, 22, 19, 26, 23, 20, 53, 50, 47, 30, 31, 32, 33, 34, 35, 27, 37, 38, 28, 40, 41, 29, 43, 44, 45, 46, 9, 48, 49, 10, 51, 52, 11],
    [45, 46, 47, 3, 4, 5, 6, 7, 8, 15, 12, 9, 16, 13, 10, 17, 14, 11, 38, 37, 36, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 0, 1, 2, 39, 40, 41, 42, 43, 44, 20, 19, 18, 48, 49, 50, 51, 52, 53],
    [36, 37, 38, 3, 4, 5, 6, 7, 8, 11, 14, 17, 10, 13, 16, 9, 12, 15, 47, 46, 45, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 20, 19, 18, 39, 40, 41, 42, 43, 44, 0, 1, 2, 48, 49, 50, 51, 52, 53],
    [0, 1, 2, 3, 4, 5, 51, 52, 53, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 44, 43, 42, 33, 30, 27, 34, 31, 28, 35, 32, 29, 36, 37, 38, 39, 40, 41, 6, 7, 8, 45, 46, 47, 48, 49, 50, 26, 25, 24],
    [0, 1, 2, 3, 4, 5, 42, 43, 44, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 53, 52, 51, 29, 32, 35, 28, 31, 34, 27, 30, 33, 36, 37, 38, 39, 40, 41, 26, 25, 24, 45, 46, 47, 48, 49, 50, 6, 7, 8],
    [0, 34, 2, 3, 31, 5, 6, 28, 8, 9, 1, 11, 12, 4, 14, 15, 7, 17, 18, 16, 20, 21, 13, 23, 24, 10, 26, 27, 19, 29, 30, 22, 32, 33, 25, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53],
    [0, 10, 2, 3, 13, 5, 6, 16, 8, 9, 25, 11, 12, 22, 14, 15, 19, 17, 18, 28, 20, 21, 31, 23, 24, 34, 26, 27, 7, 29, 30, 4, 32, 33, 1, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53],
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 46, 49, 52, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 37, 40, 43, 33, 34, 35, 36, 14, 38, 39, 13, 41, 42, 12, 44, 45, 32, 47, 48, 31, 50, 51, 30, 53],
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 43, 40, 37, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 52, 49, 46, 33, 34, 35, 36, 30, 38, 39, 31, 41, 42, 32, 44, 45, 12, 47, 48, 13, 50, 51, 14, 53],
    [0, 1, 2, 48, 49, 50, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 41, 40, 39, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 3, 4, 5, 42, 43, 44, 45, 46, 47, 23, 22, 21, 51, 52, 53],
    [0, 1, 2, 39, 40, 41, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 50, 49, 48, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 23, 22, 21, 42, 43, 44, 45, 46, 47, 3, 4, 5, 51, 52, 53],
]

CUBE2_INDICES_IN_CUBE3_VECTOR = [0, 2, 6, 8, 9, 11, 15, 17, 18, 20, 24, 26, 27, 29, 33, 35, 36, 38, 42, 44, 45, 47, 51, 53]
MOVES_INDICES_TO_NAMES = {0: 'LU', 1: 'LD', 2: 'RU', 3: 'RD', 4: 'FL', 5: 'FR', 6: 'BL', 7: 'BR', 8: 'UL', 9: 'UR', 10: 'DL', 11: 'DR', 12: 'MU', 13: 'MD', 14: 'ML_X', 15: 'MR_X', 16: 'ML_Y', 17: 'MR_Y'}
MOVES_NAMES_TO_INDICES = {'LU': 0, 'LD': 1, 'RU': 2, 'RD': 3, 'FL': 4, 'FR': 5, 'BL': 6, 'BR': 7, 'UL': 8, 'UR': 9, 'DL': 10, 'DR': 11, 'MU': 12, 'MD': 13, 'ML_X': 14, 'MR_X': 15, 'ML_Y': 16, 'MR_Y': 17}
FIXED_CUBIE_MOVES_INDICES_TO_NAMES = {0: 'LU', 1: 'LD', 6: 'BL', 7: 'BR', 8: 'UL', 9: 'UR', 12: 'MU', 13: 'MD', 14: 'ML_X', 15: 'MR_X', 16: 'ML_Y', 17: 'MR_Y'}
FIXED_CUBIE_MOVES_NAMES_TO_INDICES = {'LU': 0, 'LD': 1, 'BL': 6, 'BR': 7, 'UL': 8, 'UR': 9, 'MU': 12, 'MD': 13, 'ML_X': 14, 'MR_X': 15, 'ML_Y': 16, 'MR_Y': 17}

MOVES_NAMES = ["LU", "LD", "RU", "RD", "FL", "FR", "BL", "BR", "UL", "UR", "DL", "DR", "MU", "MD", "ML_X", "MR_X", "ML_Y", "MR_Y"]
REVERSE_MOVES_NAMES = ["LD", "LU", "RD", "RU", "FR", "FL", "BR", "BL", "UR", "UL", "DR", "DL", "MD", "MU", "MR_X", "ML_X", "MR_Y", "ML_Y"]

logger = getLogger(__name__)

class Cube3SolvedCorners(Cube3Environment):

    moves: List[str] = [x for x in MOVES_NAMES if x in FIXED_CUBIE_MOVES_NAMES_TO_INDICES]
    moves_rev: List[str] = [x for x in REVERSE_MOVES_NAMES if x in FIXED_CUBIE_MOVES_NAMES_TO_INDICES]

    def __init__(self):
        super().__init__()
        self.dtype = np.uint8
        self.cube_len = 3

        # solved state
        self.goal_colors: np.ndarray = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1,
                                                 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                                 3, 3, 3, 3, 3, 3, 3, 3, 3,
                                                 4, 4, 4, 4, 4, 4, 4, 4, 4,
                                                 5, 5, 5, 5, 5, 5, 5, 5, 5,
                                                 6, 6, 6, 6, 6, 6, 6, 6, 6], dtype=self.dtype)

    def generate_states(self, num_states: int, backwards_range: Tuple[int, int]) -> Tuple[List[Cube3State], List[int]]:
        """
        Generates 3-cube states with corner cubies placed correctly.
        """
        cube3_states, scramble_nums = super(Cube3SolvedCorners, self).generate_states(num_states, backwards_range)
        return self.solve_corners(cube3_states), scramble_nums

    def solve_corners(self, cube3_states: List[Cube3State]) -> List[Cube3State]:
        """
        Solves corner pieces of 3-cubes by reducing them to 2-cubes, solving the smaller cubes,
        and applying the same solution to the larger cube.
        """
        states_np = np.array([state.colors for state in cube3_states], dtype=self.dtype)
        cubes3_with_solved_corners = []
        cubes2_as_ints = self.convert_3_cubes_to_2_cubes(states_np)
        for i, cube2_state_int in enumerate(cubes2_as_ints):
            cube2_solution = find_solution(cube2_state_int)
            cube3_with_solved_corners = self.apply_cube2_solution_to_cube3(cube2_solution, states_np[i])
            cubes3_with_solved_corners.append(Cube3State(cube3_with_solved_corners))
        return cubes3_with_solved_corners

    def convert_3_cubes_to_2_cubes(self, cubes : np.ndarray) -> List[int]:
        cube2_vectors = cubes[:, CUBE2_INDICES_IN_CUBE3_VECTOR]
        ints = []
        for cube2_vector in cube2_vectors:
            ints.append(int(''.join(['{0:03b}'.format(i) for i in cube2_vector]), 2))
        return ints

    def apply_cube2_solution_to_cube3(self, cube2_solution, cube3_state):
        """
        Applies a list of cube2 moves to cube3 state
        """
        new_cube3_state = np.array([cube3_state])
        for move_name in cube2_solution:
            move = MOVES_DEFINITIONS[MOVES_NAMES_TO_INDICES[move_name]]
            new_cube3_state = new_cube3_state[:, move]
        return new_cube3_state[0]

    def _move_np(self, states_np: np.ndarray, action_index: int):
        """
        Applies an action to a numpy array of cube states
        """
        move_name = self.moves[action_index]
        move_id = MOVES_NAMES_TO_INDICES[move_name]
        move_definition = MOVES_DEFINITIONS[move_id]
        states_next_np = states_np[:, move_definition]
        transition_costs: List[float] = [1.0 for _ in range(states_np.shape[0])]

        return states_next_np, transition_costs

    def _compute_rotation_idxs(self, cube_len: int,
                               moves: List[str]) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Doesn't do anything, it's here just for compat with parent Cube3 class
        """
        return {}, {}


# Cube2 solver {{{

lookup_table_path = (Path(__file__).parent / "results-cubies-fixed.txt").__str__()
lookup_table_path_gz = lookup_table_path + ".gz"
if not os.path.exists(lookup_table_path) and os.path.exists(lookup_table_path_gz):
    import gzip
    import shutil
    with gzip.open(lookup_table_path_gz, 'rb') as f_in:
        with open(lookup_table_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

LOOKUP = {}

def load_lookup_table():
    if len(LOOKUP) > 0:
        return
    logger.info("Loading cube2 lookup table...")
    with open(lookup_table_path.__str__()) as fp:
        for line in fp:
            moves, binary_rep = line.strip().split(' ')
            state = int(binary_rep, 2)
            moves_nb = int(moves)
            LOOKUP[state] = moves_nb
    logger.info("Loaded!")


def find_solution(state:int):
    from environments.cube2_bitwise_ops import FIXED_CUBIE_OPS
    load_lookup_table()
    path = []
    if state not in LOOKUP:
        raise Exception("State not found in lookup table!")

    distance = LOOKUP[state]
    for i in range(distance):
        for op in FIXED_CUBIE_OPS:
            new_state = op(state)
            new_distance = LOOKUP[new_state]
            if new_distance < distance:
                path.append(op.__name__.upper())
                state, distance = new_state, new_distance
                break
        else:
            raise Exception("Did not find any move leading to a shorter distance")
    return path

load_lookup_table()

# }}}


if __name__ == "__main__":
    c = Cube3SolvedCorners()
    print([x.colors for x in c.generate_states(1, (0, 5))[0]])

