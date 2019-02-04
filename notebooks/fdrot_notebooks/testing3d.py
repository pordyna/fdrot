
# coding: utf-8

# In[1]:


import fdrot
import matplotlib.pyplot as plt
import numpy as np
from os import path


# In[8]:


#input_rand_Bx = np.random.rand(10, 5, 8)
#input_rand_Bz = np.random.rand(10, 5, 8)
#input_rand_By = np.random.rand(10, 5, 8)
#input_rand_n_e = np.random.rand(10, 5, 8)


# In[17]:


# ! mkdir data_3d_testing
#! rm data_3d_testing/*
#! ls data_3d_testing/


# In[2]:


# fdrot.sim_data.__file__


# In[2]:


#for field, arr in [('Bx', input_rand_Bx),
#              ('Bz', input_rand_Bz),
#              ('By', input_rand_By),
#              ('n_e', input_rand_n_e)]:
#    np.save(path.join('data_3d_testing/', field + '1'), arr)


# In[3]:


dt = 1
grid = (1,1,1)


# In[4]:


def open_fkt(path:str, *args):
    return np.load(path)


# In[5]:


files = {}
for field in ['Bx', 'Bz', 'By', 'n_e']:
    files[field] = fdrot.sim_data.UniversalSingle(
        path='data_3d_testing/',
        data_stored=field,
        single_time_step=dt,
        grid=grid,
        name_front=field,
        name_end='.npy',
        sim_box_shape=(10,5,8),
        export_func=open_fkt,
        axis_map=('z', 'y', 'x')
    )


# In[7]:



sequence = fdrot.sim_sequence.seq_cells(start=0, end=8,
                             inc_time=1,
                             iter_step=1,
                             pulse_length_cells=1,
                             files=files, propagation_axis='x')


# In[12]:


rotated = sequence.rotation_3d_perp(np.ones(1, dtype=np.float64), wavelength=1, second_axis_output='y')

