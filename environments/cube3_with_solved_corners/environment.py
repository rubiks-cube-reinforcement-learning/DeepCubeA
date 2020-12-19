from typing import List, Dict, Tuple
import numpy as np

from environments.cube3 import Cube3State, Cube3 as Cube3Environment
from xcs229ii_cube.cube2.cube import Cube2
from xcs229ii_cube.cube2.solver import load_lookup_table, find_solution
from xcs229ii_cube.cube3.generated_lists import apply_move_np, MOVES_DEFINITIONS, MOVES_NAMES, MOVES_NAMES_TO_INDICES, REVERSE_MOVES_NAMES, FIXED_CUBIE_MOVES_NAMES_TO_INDICES
from xcs229ii_cube.glue2_to_3 import convert_3_cubes_to_2_cubes
from xcs229ii_cube.utils import StickerVectorSerializer
from xcs229ii_cube.loggers import getLogger

logger = getLogger(__name__)
logger.info("Loading cube2 lookup table...")
load_lookup_table()
logger.info("Loaded!")

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
        cube3_states, scramble_nums = super(Cube3SolvedCorners, self).generate_states(num_states, backwards_range)
        return self.solve_corners(cube3_states), scramble_nums

    def solve_corners(self, cube3_states):
        states_np = np.array([state.colors for state in cube3_states], dtype=self.dtype)
        cubes3_with_solved_corners = []
        cubes_2_states = convert_3_cubes_to_2_cubes(states_np)
        for i, cube2_state in enumerate(cubes_2_states):
            cube2_solution = self.solve_cube_2(cube2_state)
            cube3_with_solved_corners = self.apply_cube2_solution_to_cube3(cube2_solution, states_np[i])
            cubes3_with_solved_corners.append(Cube3State(cube3_with_solved_corners))
        return cubes3_with_solved_corners

    def solve_cube_2(self, cube2_state):
        cube_2_as_int = StickerVectorSerializer(Cube2).unserialize(cube2_state).as_stickers_int
        return find_solution(cube_2_as_int)

    def apply_cube2_solution_to_cube3(self, cube2_solution, cube3_state):
        new_cube3_state = np.array([cube3_state])
        for move_name in cube2_solution:
            move = MOVES_DEFINITIONS[MOVES_NAMES_TO_INDICES[move_name]]
            new_cube3_state = apply_move_np(new_cube3_state, move)
        return new_cube3_state[0]

    def _move_np(self, states_np: np.ndarray, action: int):
        states_next_np = apply_move_np(states_np, MOVES_DEFINITIONS[action])
        transition_costs: List[float] = [1.0 for _ in range(states_np.shape[0])]

        return states_next_np, transition_costs

    def _compute_rotation_idxs(self, cube_len: int,
                               moves: List[str]) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Doesn't do anything, it's here just for compat with parent Cube3 class
        """
        return {}, {}


if __name__ == "__main__":
    c = Cube3SolvedCorners()
    print([x.colors for x in c.generate_states(20, (2, 2))[0]])
    # print(c._move_np(states, 1))
