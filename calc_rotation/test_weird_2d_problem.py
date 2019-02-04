# coding: utf-8

# In[1]:


import fdrot
import numpy as np
import os
import matplotlib.pyplot as plt

from scipy.constants import c

# In[2]:


get_ipython().run_line_magic('matplotlib', 'notebook')

# In[3]:


grid_x = 2251
grid_y = 4660
x_s = 22.5
x_e = 37.5
y_s = -15
y_e = 15

grid_size_x = (x_e - x_s) / grid_x  # in micron
grid_size_x *= 1e-6
grid_size_y = (y_e - y_s) / grid_y  # in micron
grid_size_y *= 1e-6
dt = 35.2e-15  # 35.2 fs


# In[4]:


def open_sim_data_density(path: str, *args):
    with open(path, mode='rb') as file:
        print(0, "started loading denisty")
        laser_wavelength = 1.057  # micron
        n_c_laser = 1.1148074e21 / laser_wavelength ** 2  # in 1/cm^3 (?)
        density_unit = 1100 * n_c_laser  # in 1/cm^3
        density_unit *= 100 ** 3  # in 1/m^3
        print(1)
        density = np.genfromtxt(file, dtype=np.float64)
        print(2)
        # Modify density due to interleaves (whatever it means; it gets
        # rid of those ugly lines).
        density[29 - 1::29, :] *= 2
        density[29::29, :] *= 2
        density *= density_unit
        print("opening n_e")
    return density


def open_sim_data_bz(path: str, *args):
    with open(path, mode='rb') as file:
        bfield_unit = 10.7 * 10 ** (3)  # in T
        b_field = np.genfromtxt(file, dtype=np.float64)
        b_field *= bfield_unit
        print("opening B")
    return b_field


# In[5]:


path_global = ('/net/cns/projects/HPLsim/ions/kluget/lingen/0Postdata/NewShockStudy/Vulcan/'
               'ppl0_with_ioni/500fs_a4_Al_5um/')

path_bz = os.path.join(path_global, 'emps')
path_n_e = os.path.join(path_global, 'dnss')



files_bz = fdrot.sim_data.UniversalSingle(path_bz,
                                          data_stored='Bz',
                                          single_time_step=dt,
                                          grid=(grid_size_x, grid_size_y),
                                          name_front='Ave_Bz_',
                                          name_end='.dat',
                                          axis_map=('y', 'x'),
                                          sim_box_shape=(4660, 2251),
                                          export_func=open_sim_data_bz)
files_n_e = fdrot.sim_data.UniversalSingle(path_n_e,
                                           data_stored='n_e',
                                           single_time_step=dt,
                                           grid=(grid_size_x, grid_size_y),
                                           axis_map=('y', 'x'),
                                           name_front='e_density',
                                           name_end='.dat',
                                           sim_box_shape=(4660, 2251),
                                           export_func=open_sim_data_density)

# In[7]:


files = {'Bz': files_bz, 'n_e': files_n_e}

# In[8]:


incoming_time = 40 * dt  # Start at 40th iteration.


# In[9]:


from scipy.constants import physical_constants

h = physical_constants['Planck constant in eV s'][0]
E = 6.5e3  # eV
wvl = h * c / E


# In[10]:


# create a gauss pulse
def gauss(x, sigma, mu):
    gauss = np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    return gauss


def gauss_pulse(length, sigma):
    x = np.arange(length) + 0.5
    mu = length / 2
    gs = gauss(x, sigma, mu)
    gs = gs / np.sum(gs)
    return x, gs


# In[11]:


sigma = (10e-15 * c / (grid_size_y)) / 2.35
pulse_length_cells = 3.5 * 2 * sigma
pulse_length_cells = int(round(pulse_length_cells))
x, y = gauss_pulse(pulse_length_cells, sigma)
pulse = y


# In[12]:


sequence = fdrot.sim_sequence.seq_cells(start=0, end=grid_y,
                                        inc_time=incoming_time,
                                        iter_step=1,
                                        pulse_length_cells=pulse_length_cells,
                                        files=files, propagation_axis='y')

# In[13]:



# In[ ]:


rotated = sequence.rotation_2d_perp(pulse=pulse, wavelength=wvl, cut_second_axis=(600, 1601))

# In[ ]:
