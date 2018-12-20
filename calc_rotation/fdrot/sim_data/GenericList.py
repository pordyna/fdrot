"""GenericList.py
This file is a part of the Fdrot package.
"""

from typing import Sequence, Iterable, Union, List, Tuple, Optional, NamedTuple
from warnings import warn

Shape = Union[Tuple[int, int], Tuple[int, int, int]]

class AxisOrder(NamedTuple):
    x: int
    y: int
    z: int

class GenericList:
    """ A generic class for a list of simulation files. All lists should inherit from it.

        Attributes:
            single_time_step: Duration of one iteration.
            ids: Available iterations.
            grid_unit: Length of one cell in a simulation. (dx)
            sim_box_shape: The shape of the simulation box.
            data_stored: Fields accessible through this index. Stored as string keys (example: ['Bz', n_e']).
            axis_map : ('x', 'y', 'z') or any permutation of its values. Set only in the 3D case. It should
                correspond with the orientation of the fields components.
    """
    def __init__(self, single_time_step: float, ids: Sequence[int],
                 grid_unit: float, sim_box_shape: Shape, data_stored: Sequence[str],
                 axis_map: Optional[Sequence[str]] = None) -> None:
        self.single_time_step = single_time_step
        self.ids = ids
        self.grid_unit = grid_unit
        self.sim_box_shape = sim_box_shape
        self.data_stored = data_stored
        # self.axis_map = axis_map
        if axis_map is None and self.data_dim == 3:
            raise TypeError("`axis_map` can't be None if data is in 3D.")
        if axis_map is not None:
            self.axis_map =


    @property
    def data_dim(self) -> int:
        l = len(self.sim_box_shape)
        if l == 2:
            return 2
        elif self.sim_box_shape[2] == 0:
            return 2
        return 3

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

    def open(self, iteration: int, field: str,
             dim1_cut: Optional[Tuple[int, int]] = None,
             dim2_cut: Optional[Tuple[int, int]] = None,
             dim3_cut: Optional[Tuple[int, int]] = None):  # -> np.ndarray

        if dim1_cut is None:
            dim1_cut = (0, self.sim_box_shape[0])
        if dim2_cut is None:
            dim2_cut = (0, self.sim_box_shape[1])
        if dim3_cut is None and self.data_dim == 3:
            dim3_cut = (0, self.sim_box_shape[2])

        """Should return field data for a specific iteration. Has to be overridden."""
        field = field.strip()
        if field in self.data_stored:
                raise NotImplementedError("This is just a generic class. Use one of the child classes.")
        else:
            raise ValueError("This FilesList object is not set to store this type of a simulation data.")
