"""
This file is a part of the fdrot package.

Authors: Paweł Ordyna
"""
from typing import Tuple, Sequence,  Optional
import numpy as np
import openpmd_api

from . import GenericList


class OpenPMDList(GenericList):

    """A file index for simulation data stored in the openPMD standard.
     It uses the openpmd-api.

    Attributes:
            single_time_step: Duration of one iteration.
            grid: Grid spacing in the simulation in all directions.
            sim_box_shape: The shape of the simulation box.
            data_stored: Fields accessible through this index. Stored as
              string keys; for example::

                ['Bz', 'n_e']

            axis_map (dict): keys [x, y, z] possible values [1, 2, 3].
            series: An openPMD API Series object.
    """

    def __init__(self, series: openpmd_api.Series, data_stored: Sequence,
                 single_time_step: Optional[float] = None,
                 sim_box_shape: Optional[Tuple[int, int]] = None,
                 grid: Optional[Sequence[float]] = None,
                 axis_order: Optional[Sequence[str]] = None,
                 fields_mapping: Optional[dict] = None):
        """Initializes an OpenPMDList object.

        Args:
            axis_order: ('x', 'y', 'z') or any permutation of its
              values. In the 2D case use just 'x' and 'y'. It should
              correspond with the orientation of the fields components.
              (for example Bz has to be in z direction).
        """

        # Maps the fields to their location in the openPMD container.
        # The location is specified as a tuple of group names.
        if fields_mapping is None:
            self.fields_mapping = {'Bx': ('B', 'x'), 'By': ('B', 'y'),
                                   'Bz': ('B', 'z'), 'n_e': ('e_density',
                                                             '\x0bScalar')}
        else:
            self.fields_mapping = fields_mapping
        self.series = series
        ids = list(series.iterations)  # this is a bit tricky

        # Obtaining the parameters from the series. It's assumed that, they
        # stay the same, for the whole series.
        iteration = series.iterations[ids[0]]
        if axis_order is None:
            key = self.fields_mapping[data_stored[0]][0]
            axis_order = iteration.meshes[key].axis_labels
        if single_time_step is None:
            single_time_step = (iteration.dt
                                * iteration.time_unit_SI)
        if sim_box_shape is None:
            sim_box_shape = tuple(self._get_mesh_record(iteration, data_stored[0]).shape)
        if grid is None:
            key = self.fields_mapping[data_stored[0]][0]
            unit = iteration.meshes[key].grid_unit_SI
            spacing = iteration.meshes[key].grid_spacing
            grid = tuple(unit * np.asarray(spacing))

        super().__init__(single_time_step, ids, grid, sim_box_shape,
                         data_stored, axis_order=axis_order)

    def _get_mesh_record(self, iteration: openpmd_api.Iteration, field: str) -> openpmd_api.Mesh:
        """Returns a mesh object for a specific iteration and field."""
        mesh = iteration.meshes
        for key in self.fields_mapping[field]:
            mesh = mesh[key]
        return mesh

    def open_iteration(self, iteration: int):
        self.iteration = self.series.iterations[iteration]

    def close_iteration(self):
        self.iteration.close()
        self.iteration = None

    def open(self, field: str,
             dim1_cut: Optional[Tuple[int, int]] = None,
             dim2_cut: Optional[Tuple[int, int]] = None,
             dim3_cut: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """Opens field data for a specific iteration and field.

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

        Returns: Chunk of data
        """

        if self.iteration is None:
            raise AssertionError("open_iteration has to be called first")

        field = field.strip()
        if field not in self.data_stored:
            raise ValueError("This instance is not set to store this"
                             " type of a simulation data.")

        if dim1_cut is None:
            dim1_cut = (0, self.sim_box_shape[0])
        if dim2_cut is None:
            dim2_cut = (0, self.sim_box_shape[1])
        if dim3_cut is None and self.data_dim == 3:
            dim3_cut = (0, self.sim_box_shape[2])

        data_mesh = self._get_mesh_record(self.iteration, field)

        if self.data_dim == 3:
            offset = [dim1_cut[0], dim2_cut[0], dim3_cut[0]]
            extent = (dim1_cut[1] - dim1_cut[0],
                      dim2_cut[1] - dim2_cut[0],
                      dim3_cut[1] - dim3_cut[0])
        else:
            offset = [dim1_cut[0], dim2_cut[0]]
            extent = (dim1_cut[1] - dim1_cut[0],
                      dim2_cut[1] - dim2_cut[0])
        data = data_mesh.load_chunk(offset, extent)
        self.series.flush()
        data *= data_mesh.unit_SI
        return data
