from typing import List, Dict, Tuple
import numpy as np

from environments.cube3 import Cube3 as Cube3Environment

MOVES_DEFINITIONS = [
    [14, 1, 12, 3, 0, 5, 2, 7, 6, 9, 4, 11, 8, 13, 10, 15, 17, 19, 16, 18, 20, 21, 22, 23],
    [4, 1, 6, 3, 10, 5, 8, 7, 12, 9, 14, 11, 2, 13, 0, 15, 18, 16, 19, 17, 20, 21, 22, 23],
    [0, 15, 2, 13, 4, 1, 6, 3, 8, 7, 10, 5, 12, 9, 14, 11, 16, 17, 18, 19, 22, 20, 23, 21],
    [0, 5, 2, 7, 4, 11, 6, 9, 8, 13, 10, 15, 12, 3, 14, 1, 16, 17, 18, 19, 21, 23, 20, 22],
    [1, 3, 0, 2, 4, 5, 20, 22, 8, 9, 10, 11, 12, 13, 17, 19, 16, 7, 18, 6, 15, 21, 14, 23],
    [2, 0, 3, 1, 4, 5, 19, 17, 8, 9, 10, 11, 12, 13, 22, 20, 16, 14, 18, 15, 6, 21, 7, 23],
    [0, 1, 2, 3, 21, 23, 6, 7, 9, 11, 8, 10, 16, 18, 14, 15, 5, 17, 4, 19, 20, 13, 22, 12],
    [0, 1, 2, 3, 18, 16, 6, 7, 10, 8, 11, 9, 23, 21, 14, 15, 12, 17, 13, 19, 20, 4, 22, 5],
    [20, 21, 2, 3, 6, 4, 7, 5, 17, 16, 10, 11, 12, 13, 14, 15, 0, 1, 18, 19, 9, 8, 22, 23],
    [16, 17, 2, 3, 5, 7, 4, 6, 21, 20, 10, 11, 12, 13, 14, 15, 9, 8, 18, 19, 0, 1, 22, 23],
    [0, 1, 22, 23, 4, 5, 6, 7, 8, 9, 19, 18, 14, 12, 15, 13, 16, 17, 2, 3, 20, 21, 11, 10],
    [0, 1, 18, 19, 4, 5, 6, 7, 8, 9, 23, 22, 13, 15, 12, 14, 16, 17, 11, 10, 20, 21, 2, 3],
]

MOVES_INDICES_TO_NAMES = {0: 'LU', 1: 'LD', 2: 'RU', 3: 'RD', 4: 'FL', 5: 'FR', 6: 'BL', 7: 'BR', 8: 'UL', 9: 'UR', 10: 'DL', 11: 'DR'}
MOVES_NAMES_TO_INDICES = {'LU': 0, 'LD': 1, 'RU': 2, 'RD': 3, 'FL': 4, 'FR': 5, 'BL': 6, 'BR': 7, 'UL': 8, 'UR': 9, 'DL': 10, 'DR': 11}
FIXED_CUBIE_MOVES_INDICES_TO_NAMES = {0: 'LU', 1: 'LD', 6: 'BL', 7: 'BR', 8: 'UL', 9: 'UR'}
FIXED_CUBIE_MOVES_NAMES_TO_INDICES = {'LU': 0, 'LD': 1, 'BL': 6, 'BR': 7, 'UL': 8, 'UR': 9}

MOVES_NAMES = ["LU", "LD", "RU", "RD", "FL", "FR", "BL", "BR", "UL", "UR", "DL", "DR"]
REVERSE_MOVES_NAMES = ["LD", "LU", "RD", "RU", "FR", "FL", "BR", "BL", "UR", "UL", "DR", "DL"]

class Cube2(Cube3Environment):

    moves: List[str] = [x for x in MOVES_NAMES if x in FIXED_CUBIE_MOVES_NAMES_TO_INDICES]
    moves_rev: List[str] = [x for x in REVERSE_MOVES_NAMES if x in FIXED_CUBIE_MOVES_NAMES_TO_INDICES]

    def __init__(self):
        super().__init__()
        self.dtype = np.uint8
        self.cube_len = 2

        # solved state
        self.goal_colors: np.ndarray = np.array([1, 1, 1, 1,
                                                 2, 2, 2, 2,
                                                 3, 3, 3, 3,
                                                 4, 4, 4, 4,
                                                 5, 5, 5, 5,
                                                 6, 6, 6, 6], dtype=self.dtype)

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
        Doesn't do anything, it's here just for compat with parent class
        """
        return {}, {}


if __name__ == "__main__":
    c = Cube2()
    solved = np.array([c.generate_states(20, (0, 0))[0][0].colors])
    print(solved.shape)
    print(c._move_np(solved, 1))
    print(c._move_np(c._move_np(solved, 1)[0], 0))
    # print([x.colors for x in ])
    # print([x.colors for x in c.generate_states(20, (2, 2))[0]])
    # print(c._move_np(states, 1))
