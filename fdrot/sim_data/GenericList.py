"""
This file is a part of the fdrot package.

Authors: Paweł Ordyna
"""

from typing import Sequence, Union, List, Tuple, Optional
from warnings import warn

Shape = Union[Tuple[int, int], Tuple[int, int, int]]


class GenericList:

    """ A generic class for a list of simulation files.

    All lists should inherit from it.

        Attributes:
            single_time_step: Duration of one iteration.
            grid: Grid spacing in the simulation in all directions.
            sim_box_shape: The shape of the simulation box.
            data_stored: Fields accessible through this index. Stored as
              string keys; for example::

                ['Bz', 'n_e']

            axis_map (dict): keys [x, y, z] possible values [1, 2, 3].
    """

    def __init__(self, single_time_step: float, ids: Sequence[int],
                 grid: Sequence[float], sim_box_shape: Shape,
                 data_stored: Sequence[str],
                 axis_order: Sequence[str]) -> None:
        """Initializes a GenericList object.

        Args:
            axis_order: ('x', 'y', 'z') or any permutation of its
              values. In the 2D case use just 'x' and 'y'. It should
              correspond with the orientation of the fields components.
              (for example Bz has to be in z direction).

        Raises:
            ValueError: If data dimensionality is not 2 or 3.
        """

        self.single_time_step = single_time_step
        self.ids = ids
        self.grid = tuple(grid)
        self.sim_box_shape = sim_box_shape
        self.data_stored = data_stored
        self.axis_map = {}
        for idx, axis in enumerate(axis_order):
            self.axis_map[axis] = idx
        if self.data_dim not in [2, 3]:
            raise ValueError("Data has to be either in 2D or 3D.")
        self.iteration = None

    @property
    def data_dim(self) -> int:
        """dimensionality of the simulation data (2 or 3)"""
        l = len(self.sim_box_shape)
        if l == 2:
            return 2
        elif self.sim_box_shape[2] == 0:
            return 2
        return 3

    @property
    def ids(self) -> List[int]:
        """available iterations"""
        return self._ids

    # Ids should be sorted and with unique values.
    @ids.setter
    def ids(self, id_list: Sequence) -> None:
        u_sorted = sorted(set(id_list))
        # check the uniqueness.
        if len(id_list) != len(u_sorted):
            warn('The IDs should be unique. Removing obsolete values and '
                 'continuing.')
        self._ids = u_sorted

    def open_iteration(self, iteration: int):
        raise NotImplementedError

    def close_iteration(self):
        raise NotImplementedError

    def open(self, field: str,
             dim1_cut: Optional[Tuple[int, int]] = None,
             dim2_cut: Optional[Tuple[int, int]] = None,
             dim3_cut: Optional[Tuple[int, int]] = None):  # -> np.ndarray
        """Opens the field data, for a specific iteration and field.

        To obtain only a specific chunk of data set dimension cuts.

        Args:
            field: Field to return. Has to be included in data_stored.
            dim1_cut: Interval along the 1st axis that should be
              included. If None the whole axis is included. For (a, b)
              a is included, b is not.
            dim2_cut: Interval along the 2nd axis that should be
              included. If None the whole axis is included. For (a, b)
              a is included, b is not.
            dim3_cut: Interval along the 3rd axis that should be
              included. If None the whole axis is included. For (a, b)
              a is included, b is not.

        Raises:
            ValueError: If `field` is not in `data_stored`.
            NotImplementedError: If not overridden.

        Returns: None, It should return field data for a specific
          iteration and has to be overridden.
        """

        if dim1_cut is None:
            dim1_cut = (0, self.sim_box_shape[0])
        if dim2_cut is None:
            dim2_cut = (0, self.sim_box_shape[1])
        if dim3_cut is None and self.data_dim == 3:
            dim3_cut = (0, self.sim_box_shape[2])


        field = field.strip()
        if field in self.data_stored:
            raise NotImplementedError("This is just a generic class. Use "
                                      "one of the child classes.")
        else:
            raise ValueError("This FilesList object is not set to store this "
                             "type of a simulation data.")
