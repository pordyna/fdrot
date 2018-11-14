import numpy as np
from typing import Sequence
from warnings import warn


class GenericList:
    def __init__(self, single_time_step, ids, grid_unit, sim_box_shape, data_stored: Sequence[str]):
        self.single_time_step = single_time_step  # duration of a single time step in fs. # to a child class?
        self.ids = ids  # An array of available time steps (integers).
        self.grid_unit = grid_unit  # x step in microns
        self.sim_box_shape = sim_box_shape
        self.data_stored = data_stored
    @property
    def ids(self):
        return self._ids
    @ids.setter
    def ids(self, id_list: Sequence) -> None:
        u_sorted = np.unique(id_list)
        # check the uniqueness.
        if len(id_list) != u_sorted.size:
            warn('The IDs should be unique. Removing obsolete values and continuing.')
        self._ids = u_sorted

    def open(self, iteration: int, field: str):  # -> np.ndarray
        field = field.strip()
        if field in self.data_stored:
                raise NotImplementedError("This is just a generic class. Use one of the child classes.")
        else:
            raise ValueError("This FilesList object is not set to store this type of a simulation data.")
