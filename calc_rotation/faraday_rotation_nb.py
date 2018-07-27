
# coding: utf-8

# In[119]:


import faraday_rotation as fr
import matplotlib.pyplot as plt
from  matplotlib import ticker
import numpy as np


# In[3]:


#%qtconsole


# In[4]:


# target settings:
target_settings = {
    'x_s': 0,  # start position in x direction, in micron
    'x_e': 7.2,  # end position in x direction, in micron
    'y_s': -13.5,  # start in y direction, in micron
    'y_e': 13.5,  # end in y direction, in micron
    'trans': 0.72 #target transmission
}


# In[8]:


# target = fr.Target(**target_settings)


# In[9]:


# simulation settings:
simulation_settings = {
    'm': 1801,  # Nx: grid number in x direction
    'n': 6040,  # Ny: grid number if y direction, MUST be an even number
    'timestep': 16, # Number of time steps in the simulation.
    'energy': 6.457  # Photon energy in keV of processed simulation file.
}

simulation = fr.Simulation(fr.Target(**target_settings), **simulation_settings)


# In[10]:


# optical setup:

#Transmission of all channelcuts including spectral bandwidth.
trans_channel = 0.43*0.6/0.8 

config_settings = {
    'an_position': 10, # analyzer position (mrad)
    'impurity': 1e-5,  # polarization impurity
    'an_extinction': 2e-7,  # analyzer extinction
    'det_obs_energy': 6.457, # observation energy
    'det_trans_channel': trans_channel, # as above
    'det_pixel_size': 13,  # pixel size in micrometer
    'det_beam_width': 320,  # beam_width in micrometer
    'n_0': 1e12,  # initial number of photons
    'm': 30,  # magnification
    'trans_telescope': 1  # Transmission of CRLs due to beam size
        # mismatch (asymmetry).
}

config = fr.Configuration(**config_settings)


# In[11]:


# load simulation data:
path_to_file = "rotation_data/Rotation_16.dat"
#simulation.load_data(path=path_to_file)


# In[12]:


#np.save('sim', simulation.data)


# In[13]:



simulation.data = np.load('sim.npy')


# In[14]:


experiment = fr.Detection(config, simulation)


# In[15]:


experiment.calc_rotation()


# In[16]:


experiment.calc_det_shape()


# In[17]:


experiment.emulate_intensity()


# In[18]:


experiment.cfg.calc_ph_per_px_on_axis()


# In[19]:


experiment.calc_beam_profile()


# Displaying meany images with a common norm. (One colorbar, comparable; multiply some if the range is to wide)
# Look here: https://matplotlib.org/gallery/images_contours_and_fields/multi_image.html 

# In[20]:


experiment.ideal_detector = experiment.beam_profile*experiment.intensity_px


# In[65]:


exp_with_noise = experiment.add_noise(accumulation=1)


# In[66]:





# In[22]:


exp_noise_acc = experiment.add_noise(accumulation = 30)


# In[23]:


base_noise_acc = experiment.add_noise(accumulation=30, image=experiment.beam_profile)


# In[24]:


rot = experiment.reobtain_rotation(exp_noise_acc, base_noise_acc)


# Plots:

# In[56]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[134]:


rot_fig, axis = plt.subplots(1, 3, figsize=(15,8))
ax_rot_sim, ax_rot_sim_px, ax_rot_exp = axis

ax_rot_sim.set_title('Simulated rotation')
ax_rot_sim_px.set_title('Analyzer intensity profile')
ax_rot_exp.set_title('Reobtained rotation')

# Image boundaries, used for correct ticks labeling:
edges = (experiment.sim.target.x_s, experiment.sim.target.x_e,
         experiment.sim.target.y_s, experiment.sim.target.y_e)

ax_rot_sim.imshow(experiment.rotation, extent=edges)
ax_rot_sim_px.imshow(experiment.intensity_px, extent=edges)
ax_rot_exp.imshow(rot, extent=edges)

for ax in axis:
    rot_fig.colorbar(ax.images[0], ax = ax)
    
plt.tight_layout()    


# In[133]:


exp_fig, axis = plt.subplots(1, 3, figsize=(18,10))
ax_ref_acc, ax_ex_noise, ax_ex_acc = axis

ax_ref_acc.set_title('Reference image, accumulated ... times.')
ax_ex_noise.set_title('Signal with added noise.')
ax_ex_acc.set_title('Accumulated noisy signal')
# Image boundaries, used for correct ticks labeling:
edges = (experiment.sim.target.x_s, experiment.sim.target.x_e,
         experiment.sim.target.y_s, experiment.sim.target.y_e)
ax_ref_acc.imshow(base_noise_acc, extent=edges)
ax_ex_noise.imshow(exp_with_noise, extent=edges)
ax_ex_acc.imshow(exp_noise_acc, extent=edges)

for ax in axis:
    # Add colorbars:
    exp_fig.colorbar(ax.images[0], ax = ax)
plt.tight_layout()    


# In[80]:


ticks_formater_cells_x(val, pos):
    


# In[102]:


get_ipython().run_line_magic('pinfo', 'fr.Simulation')


# In[127]:


a= ax_ref_acc.images[0]


# In[130]:


get_ipython().run_line_magic('pinfo', 'a.convert_xunits')

