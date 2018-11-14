import h5py

def export_h5py_bz(path, step):
    with h5py.File(path) as file:
        B_z_h5py = file['data/'+str(step)+'/fields/B/z']
        B_z = B_z_h5py.value * B_z_h5py.attrs['unitSI']
        return B_z


def export_h5py_ne(path, step):
    with h5py.File(path) as file:
        ne_h5py = file['data/'+str(step)+'/fields/e_density']
        ne = ne_h5py.value * ne_h5py.attrs['unitSI']
        return ne
