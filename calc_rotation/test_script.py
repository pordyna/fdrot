import calc_rotation
from calc_rotation.sim_data.export_hdf5 import export_h5py_bz
import h5py
def main():
    path = '/home/pawel/Work/PIConGPU/151_PizDaintCopper30nmPerfectContrast/simOutput/h5/'
    file = h5py.File(path + 'simData_0.h5')
    grid_unit = file['data/0/fields/e_density'].attrs['gridUnitSI']
    dt = file['data/0'].attrs['dt'] * file['data/0'].attrs['timeUnitSI']
    file_list = calc_rotation.sim_data.FilesLists.UniversalSingle(path, 'Bz', dt, grid_unit, 'simData_', '.h5',
                                                      export_h5py_bz)
    file_list2 = calc_rotation.sim_data.FilesLists.UniversalSingle(path, 'n_e', dt, grid_unit, 'simData_', '.h5',
                                                                   export_h5py_bz)
    inc_time = 50036 * dt
    #inc_time = 0
    sequence = calc_rotation.SimSequence.seq_cells(0, 256, inc_time, 2500, 1, {'Bz': file_list, 'n_e': file_list2})
    rotated = sequence.rotation_2d_perp()
if __name__ == "__main__":
    main()
