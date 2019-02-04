import fdrot
from fdrot.sim_data.export_hdf5 import export_h5py_bz
from fdrot.sim_data.export_hdf5 import export_h5py_ne
import h5py
import numpy as np


def open_file(path, iter:int ):
    return np.load(path)


def open_file2(path, iter:int):
    a = np.load(path)
    return a[:, :-1]


def main():
    path = '/home/pawel/Work/HOME/Faraday_Rotation/data/constant_samples/'
    file = h5py.File(path + 'simData_0.h5')
    dt = 1
    grid_unit = 1
    file_list = fdrot.sim_data.UniversalSingle(path, 'Bz', dt, grid_unit, 'step_', '.npy', open_file)
    file_list2 = fdrot.sim_data.UniversalSingle(path, 'n_e', dt, grid_unit, 'step_', '.npy',
                                                open_file)

    inc_time = 50036 * dt
    #inc_time = 0
    pulse = np.ones(1)
    end = file_list.sim_box_shape[1]
    sequence = fdrot.sim_sequence.SimSequence({'Bz': file_list, 'n_e': file_list2}, 1, (0,1,1), [(0, end)], 0, end)
    rotated = sequence.rotation_2d_perp(pulse)

    file_list = fdrot.sim_data.UniversalSingle(path, 'Bz', dt, grid_unit, 'step_', '.npy', open_file2)
    file_list2 = fdrot.sim_data.UniversalSingle(path, 'n_e', dt, grid_unit, 'step_', '.npy',
                                                open_file2)
    end = file_list.sim_box_shape[1]
    sequence = fdrot.sim_sequence.SimSequence({'Bz': file_list, 'n_e': file_list2}, 1, (0, 1, 1), [(0, end)], 0, end)
    rotated = sequence.rotation_2d_perp(pulse)


if __name__ == "__main__":
    main()
