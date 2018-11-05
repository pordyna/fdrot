import numpy as np
from typing import Union, Iterable, Callable, Tuple, Sequence, MutableSequence, Optional, Any
from warnings import warn
from os import path


class FilesList:
    """

    """
    def __init__(self, path: str, single_time_step: float, ids: Sequence[int], x_step: float,
                 name_front: str, name_end: str, export_func: Callable[[str, ...], np.ndarray], args_export: Optional[Sequence[Any]] = None,
                 sim_box_shape: Optional[Tuple[int, int]] = None) -> None:
        """
        """
        # TODO move descriptions to the class doc_string.
        self.path = path# path to files
        self.single_time_step = single_time_step # duration of a single time step in fs. # to a child class?
        self.ids = ids  # An array of available time steps (integers).
        self.name_front = name_front.strip() # the constant part of a filename  before the number (id)
        self.name_end = name_end.strip()  # the constant part of a filename  after the number (id) (incl. '.extension')
        self.x_step = x_step # x step in microns
        self.sim_box_shape = sim_box_shape
        self.export_func = export_func # a function which opens the file. it should take the path to the file
            #  as the first argument, additional arguments are acceptable  It should return the data as an ndarray.
        #
        if args_export is None:
            self. args_export = ()
        else:
            self.args_export = args_export
        # check if files are there:
        isok, bad = self.check(noprint=True)
        if not isok:
            print('File check at the initialization performed. Some files are not there. Ids of the missing files are:')
            print(bad)

    @property
    def ids(self):
        return self.ids
    @ids.setter
    def ids(self, id_list: Sequence) -> None:
        u_sorted = np.unique(id_list)
        # check the uniqueness.
        if len(id_list) != u_sorted.size():
            warn('The IDs should be unique. Removing obsolete values and continuing.')
        self.ids = u_sorted

    def full_path(self, id: int) -> str:
        if id not in self.ids:
            raise ValueError('The id has to be listed in `ids` Attribute!')
        full_path = path.join(self.path, self.name_front + str(id) + self.name_end)
        full_path = path.normpath(full_path)
        return full_path

    def check(self, ids: Union[Sequence[int], int] = None, noprint: bool = False) -> Tuple[bool, list]:

        if ids is None:
            ids = self.ids
        else:
            try:
                iter(ids)
            except TypeError:
                ids = [ids]
        bad_ids = []

        for id in ids:
            if not path.isfile(self.full_path(id)):
                bad_ids.append(id)
        ok = bool(bad_ids)
        if not noprint:
            if not ok:
                print("Some files are missing. Ids of the missing files are:")
                print(bad_ids)
            if ok:
                print("All files exist.")
        return ok, bad_ids

    def open(self, iteration: int) -> np.ndarray:

        data = self.export_func(self.full_path(iteration), *self.args_export)
        if self.sim_box_shape is None:
            self.sim_box_shape = data.shape
        else:
            if self.sim_box_shape != data.shape:
                raise ValueError('Shape of the opened array is different than the `sim_box_shape` Attribute.')
        return data


# class OpenPmdList(FilesList):    coming soon :)
 # also introducing functions, which index files from a directory and create an FilesList/Open... object, would be nice.


