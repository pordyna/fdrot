import numpy as np
from typing import Tuple, Sequence,  Optional
import openPMD

from . import GenericList


class OpenPMDList(GenericList):
    def __init__(self, series: openPMD.Series, data_stored: Sequence,
                 single_time_step: Optional[float],
                 sim_box_shape: Optional[Tuple[int, int]],
                 grid_unit: Optional[int]):
        self.fields_mapping = {'Bx': ('B', 'x'), 'By': ('B', 'y'), 'Bz': ('B', 'z'), 'n_e': ('e_density',)}
        self.series = series
        ids = series.iterations.items
        if single_time_step is None:
            single_time_step = series.iterations[ids[0]].dt
        if sim_box_shape is None:
            # Assume the shape is same for every field, just check for first one
            sim_box_shape = tuple(self._get_mesh_record(ids[0], data_stored[0]).shape)
        if grid_unit is None:
            key = self.fields_mapping[data_stored[0]][0]
            grid_unit = series.iterations[ids[0]][key].grid_unit_SI
        super().__init__(single_time_step, ids, grid_unit, sim_box_shape, data_stored)

    def _get_mesh_record(self, iteration: int, field: str ) -> openPMD.Mesh:
        mesh = self.series.iterations[iteration]
        for key in self.fields_mapping[field]:
            mesh = mesh[key]
        return mesh

    def open(self, iteration: int, field: str) -> np.ndarray:
        field = field.strip()
        if field not in self.data_stored:
            raise ValueError("This FilesList object is not set to store this type of a simulation data.")

        data = self._get_mesh_record(iteration, field)
        data = data.load_chunk([0,0], self.sim_box_shape) # TODO: 3D
        return data
