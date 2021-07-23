"""
This file is a part of the fdrot package.

Authors: Paweł Ordyna
"""
import numpy as np
from typing import Union, Callable, Tuple, Sequence, Optional
import re
import os

from . import GenericList


class UniversalSingle(GenericList):

    """An universal file index  for the simulation data.

    It works with field data, if all files are in the same directory
    and the file names differ only in the iteration number.

        Attributes:
            single_time_step: Duration of one iteration.
            grid: Grid spacing in the simulation in all directions.
            sim_box_shape: The shape of the simulation box.
            data_stored: Fields accessible through this index.
              Stored as string keys; for example::

                ['Bz', 'n_e']

            axis_map (dict): keys [x, y, z] possible values [1, 2, 3].
            path: Path to the directory with the files.
            name_front: The constant part of a filename, before the
              iteration number.
            name_end:  The constant part of a filename  after the
              number (incl. file extension).
    """

    def __init__(self, path: str, data_stored: str, single_time_step: float,
                 grid: Sequence[float], name_front: str, name_end: str,
                 export_func: Callable[[str, int], np.ndarray],
                 axis_order: Sequence[str],
                 ids: Optional[Sequence[int]] = None,
                 sim_box_shape: Optional[Tuple[int, int]] = None) -> None:
        """ Initializes UniversalSingle object.



        Args:
            axis_order: ('x', 'y', 'z') or any permutation of its
              values. In the 2D case use just 'x' and 'y'. It should
              correspond with the orientation of the fields components.
              (for example Bz has to be in z direction).
            ids: Available iterations. If  not specified, they are
              automatically generated from the file names.
        """

        self.path = path
        self.name_front = name_front
        self.name_end = name_end

        if ids is None:
            super().__init__(single_time_step, [0], grid, sim_box_shape,
                             [data_stored], axis_order=axis_order)
            self.find_ids()
        else:
            super().__init__(single_time_step, ids, grid, sim_box_shape,
                             [data_stored], axis_order=axis_order)

        self.export_func = export_func

        # check if files are there:
        isok, bad = self.check(no_print=True)
        if not isok:
            print('File check at the initialization performed. Some files are'
                  ' not there. Missing iterations are:')
            print(bad)

        # If the shape is not given, obtain it from one of the files.
        # It's assumed that is same for all files.
        if sim_box_shape is None:
            self. sim_box_shape = self.open(self.ids[0], data_stored).shape

    def find_ids(self):
        """Looks for available iterations in the matching file names."""
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
        """Returns a full path to a file, for a specific iteration."""
        if iteration not in self.ids:
            raise ValueError('The id has to be listed in `ids` Attribute!')
        full_path = os.path.join(self.path, self.name_front + str(iteration)
                                 + self.name_end)
        full_path = os.path.normpath(full_path)
        return full_path

    def check(self, ids: Union[Sequence[int], int] = None,
              no_print: bool = False) -> Tuple[bool, list]:
        """Checks if all listed files exist.

        Args:
            ids: iterations to be checked if None all are checked.
            no_print: False - verbose mode, True - no console output.

        Returns:
            * True if all files are present - False otherwise.
            * Iterations with missing files.
        """

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

    def open_iteration(self, iteration: int):
        self.iteration = iteration

    def close_iteration(self):
        self.iteration = None

    def open(self,  field: str,
             dim1_cut: Optional[Tuple[int, int]] = None,
             dim2_cut: Optional[Tuple[int, int]] = None,
             dim3_cut: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """Opens field data for a specific iteration and field.

        To obtain only a specific chunk of data set dimension cuts.
        *Warning:* Only a chunk of data wil be returned but the complete
        simulation box is loaded in the process.

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

        data = self.export_func(self.full_path(self.iteration), self.iteration)

        if self.sim_box_shape != data.shape:
            raise ValueError('Shape of the opened array is different than the '
                             '`sim_box_shape` Attribute.')
        if self.data_dim == 3:
            cut_data = data[dim1_cut[0]:dim1_cut[1], dim2_cut[0]:dim2_cut[1],
                            dim3_cut[0]:dim3_cut[1]]
        else:
            cut_data = data[dim1_cut[0]:dim1_cut[1], dim2_cut[0]:dim2_cut[1]]
        return cut_data
