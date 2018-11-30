"""GenericList.py
This file is a part of the Fdrot package.
"""

from typing import Sequence, Iterable, Union, List, Tuple
from warnings import warn

Shape = Union[Tuple[int, int], Tuple[int, int, int]]


class GenericList:
    """ A generic class for a list of simulation files. All lists should inherit from it.

        Attributes:
            single_time_step: Duration of one iteration.
            ids: Available iterations.
            grid_unit: Length of one cell in a simulation. (dx)
            sim_box_shape: The shape of the simulation box.
            data_stored: Fields accessible through this index. Stored as string keys (exp. ['Bz', n_e']).
    """
    def __init__(self, single_time_step: float, ids: Sequence[int],
                 grid_unit: float, sim_box_shape: Shape, data_stored: Sequence[str]) -> None:
        self.single_time_step = single_time_step
        self.ids = ids
        self.grid_unit = grid_unit
        self.sim_box_shape = sim_box_shape
        self.data_stored = data_stored

    @property
    def ids(self) -> List[int]:
        return self._ids

    # Ids should be sorted and with unique values.
    @ids.setter
    def ids(self, id_list: Sequence) -> None:
        u_sorted = sorted(set(id_list))
        # check the uniqueness.
        if len(id_list) != len(u_sorted):
            warn('The IDs should be unique. Removing obsolete values and continuing.')
        self._ids = u_sorted

    def open(self, iteration: int, field: str):  # -> np.ndarray
        """Should return field data for a specific iteration. Has to be overridden."""
        field = field.strip()
        if field in self.data_stored:
                raise NotImplementedError("This is just a generic class. Use one of the child classes.")
        else:
            raise ValueError("This FilesList object is not set to store this type of a simulation data.")
