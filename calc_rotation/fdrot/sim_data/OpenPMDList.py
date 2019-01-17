"""OpenPMDList.py
This file is a part of the Fdrot package.
"""
import numpy as np
from typing import Tuple, Sequence,  Optional
import openpmd_api

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
    def __init__(self, series: openpmd_api.Series, data_stored: Sequence,
                 single_time_step: Optional[float] = None,
                 sim_box_shape: Optional[Tuple[int, int]] = None,
                 grid: Optional[Sequence[float]] = None, axis_map: Optional[Sequence[str]] = None,
                 fields_mapping: Optional[dict] = None):
        """Initializes an OpenPMDList object.

        Optional arguments, if they are not specified, are obtained from the series."""
        # Maps the fields to their location in the openPMD container. The location is specified
        # as a tuple of group names.
        if fields_mapping is None:
            self.fields_mapping = {'Bx': ('B', 'x'), 'By': ('B', 'y'),
                                   'Bz': ('B', 'z'), 'n_e': ('e_density', '\x0bScalar')}
        else: self.fields_mapping = fields_mapping
        self.series = series
        ids = list(series.iterations) # this is a bit tricky

        # Obtaining the parameters from the series. It's assumed that, they stay the same, for the whole series.
        if axis_map is None:
            key = self.fields_mapping[data_stored[0]][0]
            axis_map = series.iterations[ids[0]].meshes[key].axis_labels
        if single_time_step is None:
            single_time_step = series.iterations[ids[0]].dt() * series.iterations[ids[0]].time_unit_SI()
        if sim_box_shape is None:
            sim_box_shape = tuple(self._get_mesh_record(ids[0], data_stored[0]).shape)
        if grid is None:
            key = self.fields_mapping[data_stored[0]][0]
            unit = series.iterations[ids[0]].meshes[key].grid_unit_SI
            spacing = series.iterations[ids[0]].meshes[key].grid_spacing
            grid = tuple(unit * np.asarray(spacing))
        super().__init__(single_time_step, ids, grid, sim_box_shape, data_stored, axis_order=axis_map)

    def _get_mesh_record(self, iteration: int, field: str) -> openpmd_api.Mesh:
        """Returns the mesh from the series, for a specific iteration and field."""
        mesh = self.series.iterations[iteration].meshes
        for key in self.fields_mapping[field]:
            mesh = mesh[key]
        return mesh

    def open(self, iteration: int, field: str,
             dim1_cut: Optional[Tuple[int, int]] = None,
             dim2_cut: Optional[Tuple[int, int]] = None,
             dim3_cut: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """Opens the field data, for a specific iteration and field, as a numpy array."""

        field = field.strip()
        if field not in self.data_stored:
            raise ValueError("This FilesList object is not set to store this type of a simulation data.")

        if dim1_cut is None:
            dim1_cut = (0, self.sim_box_shape[0])
        if dim2_cut is None:
            dim2_cut = (0, self.sim_box_shape[1])
        if dim3_cut is None and self.data_dim == 3:
            dim3_cut = (0, self.sim_box_shape[2])

        data_mesh = self._get_mesh_record(iteration, field)
        if self.data_dim == 3:
            offset = [dim1_cut[0], dim2_cut[0], dim3_cut[0]]
        else:
            offset = [dim1_cut[0], dim2_cut[0]]
        data = data_mesh.load_chunk(offset, (dim1_cut[1], dim2_cut[1], dim3_cut[1]))
        self.series.flush()
        data *= data_mesh.unit_SI
        return data
