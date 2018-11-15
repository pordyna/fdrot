import numpy as np
from typing import Union, Callable, Tuple, Sequence, Optional
import re
import os

from ..FilesLists import GenericList


class UniversalSingle(GenericList):
    """

    """
    def __init__(self, path: str, data_stored: str, single_time_step: float, grid_unit: float,
                 name_front: str, name_end: str, export_func: Callable[[str, int], np.ndarray], ids: Optional[Sequence[int]] = None,
                 sim_box_shape: Optional[Tuple[int, int]] = None) -> None:
        """
        """
        # TODO move descriptions to the class doc_string.
        self.path = path  # path to files_B
        self.name_front = name_front  # the constant part of a filename  before the number (id)
        self.name_end = name_end  # the constant part of a filename  after the number (id) (incl. '.extension')
        if ids is None:
            super().__init__(single_time_step, [0], grid_unit, sim_box_shape, [data_stored])
            self.find_ids()
        else:
            super().__init__(single_time_step, ids, grid_unit, sim_box_shape, [data_stored])

        self.export_func = export_func # a function which opens the file. it should take the path to the file
            #  as the first argument and interation as second.
        #
        # check if files are there:
        isok, bad = self.check(no_print=True)
        if not isok:
            print('File check at the initialization performed. Some files are not there. Ids of the missing files are:')
            print(bad)

        if sim_box_shape is None:
            self. sim_box_shape = self.open(self.ids[0], data_stored).shape

    def find_ids(self):
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

    def full_path(self, idd: int) -> str:
        if idd not in self.ids:
            raise ValueError('The id has to be listed in `ids` Attribute!')
        full_path = os.path.join(self.path, self.name_front + str(idd) + self.name_end)
        full_path = os.path.normpath(full_path)
        return full_path

    def check(self, ids: Union[Sequence[int], int] = None, no_print: bool = False) -> Tuple[bool, list]:

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

    def open(self, iteration: int, field: str) -> np.ndarray:
        field = field.strip()
        if field not in self.data_stored:
            raise ValueError("This FilesList object is not set to store this type of a simulation data.")

        data = self.export_func(self.full_path(iteration), iteration)
        if self.sim_box_shape is None:
            self.sim_box_shape = data.shape
        else:
            if self.sim_box_shape != data.shape:
                raise ValueError('Shape of the opened array is different than the `sim_box_shape` Attribute.')
        return data
