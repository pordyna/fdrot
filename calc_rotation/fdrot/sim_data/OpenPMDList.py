"""OpenPMDList.py
This file is a part of the Fdrot package.
"""
import numpy as np
from typing import Tuple, Sequence,  Optional
import openPMD

from . import GenericList


class OpenPMDList(GenericList):
    """An file index for simulation data stored in the openPMD standard. It uses the openpmd-api.

    Attributes:
            single_time_step: Duration of one iteration.
            ids: Available iterations.
            grid_unit: Length of one cell in a simulation. (dx)
            sim_box_shape: The shape of the simulation box.
            data_stored: Fields accessible through this index. Stored as string keys (exp. ['Bz', n_e']).
            series: An openPMD API Series object.
    """
    def __init__(self, series: openPMD.Series, data_stored: Sequence,
                 single_time_step: Optional[float],
                 sim_box_shape: Optional[Tuple[int, int]],
                 grid_unit: Optional[int]):
        """Initializes an OpenPMDList object.

        Optional arguments, if they are not specified, are obtained from the series."""
        # Maps the fields to their location in the openPMD container. The location is specified
        # as a tuple of group names.
        self._fields_mapping = {'Bx': ('B', 'x'), 'By': ('B', 'y'), 'Bz': ('B', 'z'), 'n_e': ('e_density',)}
        self.series = series
        ids = series.iterations.items

        # Obtaining the parameters from the series. It's assumed that, they stay the same, for the whole series.
        if single_time_step is None:
            single_time_step = series.iterations[ids[0]].dt
        if sim_box_shape is None:
            sim_box_shape = tuple(self._get_mesh_record(ids[0], data_stored[0]).shape)
        if grid_unit is None:
            key = self._fields_mapping[data_stored[0]][0]
            grid_unit = series.iterations[ids[0]][key].grid_unit_SI
        super().__init__(single_time_step, ids, grid_unit, sim_box_shape, data_stored)

    def _get_mesh_record(self, iteration: int, field: str ) -> openPMD.Mesh:
        """Returns the mesh from the series, for a specific iteration and field."""
        mesh = self.series.iterations[iteration]
        for key in self._fields_mapping[field]:
            mesh = mesh[key]
        return mesh

    def open(self, iteration: int, field: str) -> np.ndarray:
        """Opens the field data, for a specific iteration and field, as a numpy array."""
        field = field.strip()
        if field not in self.data_stored:
            raise ValueError("This FilesList object is not set to store this type of a simulation data.")

        data = self._get_mesh_record(iteration, field)
        data = data.load_chunk([0, 0], self.sim_box_shape)  # TODO: 3D
        return data
