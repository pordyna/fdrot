"""UniversalSingle.py
This file is a part of the Fdrot package.
"""
import numpy as np
from typing import Union, Callable, Tuple, Sequence, Optional
import re
import os

from . import GenericList


class UniversalSingle(GenericList):
    """An universal file index  for the simulation data.

    It works with field data, if all files are in the same directory and the file names differ only
    in the iteration number.

        Attributes:
            single_time_step: Duration of one iteration.
            ids: Available iterations.
            grid_unit: Length of one cell in a simulation. (dx)
            sim_box_shape: The shape of the simulation box.
            data_stored: Fields accessible through this index. Stored as string keys (exp. ['Bz', n_e']).
            path: Path to the directory with the files.
            name_front: The constant part of a filename, before the iteration number.
            name_end:  The constant part of a filename  after the number (incl. file extension).


    """
    def __init__(self, path: str, data_stored: str, single_time_step: float, grid_unit: float,
                 name_front: str, name_end: str, export_func: Callable[[str, int], np.ndarray], ids: Optional[Sequence[int]] = None,
                 sim_box_shape: Optional[Tuple[int, int]] = None, axis_map: Optional[Sequence[str]] = None) -> None:
        """ Initializes UniversalSingle object.

        If available iterations are not specified, they are obtained from file names.
        """
        self.path = path
        self.name_front = name_front
        self.name_end = name_end

        if ids is None:
            super().__init__(single_time_step, [0], grid_unit, sim_box_shape, [data_stored], axis_map=axis_map)
            self.find_ids()
        else:
            super().__init__(single_time_step, ids, grid_unit, sim_box_shape, [data_stored], axis_map=axis_map)

        self.export_func = export_func

        # check if files are there:
        isok, bad = self.check(no_print=True)
        if not isok:
            print('File check at the initialization performed. Some files are not there. Missing iterations are:')
            print(bad)

        # If the shape is not given, obtain it from one of the files.
        # It's assumed that is same for all files.
        if sim_box_shape is None:
            self. sim_box_shape = self.open(self.ids[0], data_stored).shape

    def find_ids(self):
        """Searches for the available iterations in the matching file names."""
        regex = re.compile(self.name_front +r'\d+' + self.name_end)
        matching_files = []
        for filename in os.listdir(self.path):
            matching_files += regex.findall(filename)
        ids = [-1] * len(matching_files)
        front_offset = len(self.name_front)
        rear_offset = len(self.name_end)
        for ii, filename in enumerate(matching_files):

            ids[ii] = int(filename[front_offset:-rear_offset])
        self.ids = ids

    def full_path(self, iteration: int) -> str:
        """Returns a full path to a file, with a specific iteration."""
        if iteration not in self.ids:
            raise ValueError('The id has to be listed in `ids` Attribute!')
        full_path = os.path.join(self.path, self.name_front + str(iteration) + self.name_end)
        full_path = os.path.normpath(full_path)
        return full_path

    def check(self, ids: Union[Sequence[int], int] = None, no_print: bool = False) -> Tuple[bool, list]:
        """Check if all listed files exist."""
        if ids is None:
            ids = self.ids
        else:
            try:
                iter(ids)
            except TypeError:
                ids = [ids]
        bad_ids = []
        for id in ids:
            if not os.path.isfile(self.full_path(id)):
                bad_ids.append(id)
        ok = not bool(bad_ids)
        if not no_print:
            if not ok:
                print("Some files are missing. Ids of the missing files are:")
                print(bad_ids)
            if ok:
                print("All files exist.")
        return ok, bad_ids

    def open(self, iteration: int, field: str,
             dim1_cut: Optional[Tuple[int, int]] = None,
             dim2_cut: Optional[Tuple[int, int]] = None,
             dim3_cut: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """Opens the field data, for a specific iteration., as a numpy array."""

        field = field.strip()
        if field not in self.data_stored:
            raise ValueError("This FilesList object is not set to store this type of a simulation data.")

        if dim1_cut is None:
            dim1_cut = (0, self.sim_box_shape[0])
        if dim2_cut is None:
            dim2_cut = (0, self.sim_box_shape[1])
        if dim3_cut is None and self.data_dim == 3:
            dim3_cut = (0, self.sim_box_shape[2])

        data = self.export_func(self.full_path(iteration), iteration)

        if self.sim_box_shape != data.shape:
            raise ValueError('Shape of the opened array is different than the `sim_box_shape` Attribute.')
        cut_data = data[dim1_cut[0]:dim1_cut[1], dim2_cut[0]:dim2_cut[1], dim3_cut[0]:dim3_cut[1]]

        return cut_data
