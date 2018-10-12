""" Faraday Rotation.

This module delivers some classes for simulating an synthetic Faraday
rotation experiment. In such an experiment a polarized beam passes
through an optically active sample. The change in the polarization
direction is than measured by placing an analyzer in the beam line.
For a given rotation it allows a simulation of the detected image
and the quality of the reobtained information for different optical
configurations.

It has been developed to be used together with an jupyter notebook.
 """
# import matplotlib.pyplot as plt
import math
import numpy as np
from skimage import transform


def _beam_profile_fkt(x, y, x_0, y_0, c):
    """Function defining the beam profile.

    Args:
      x (float) : x coordinate.
      y (float) : y coordinate.
      x_0 (float): beam center (x coordinate).
      y_0 (float): beam center (y coordinate).
      c (float):  scaling parameter.

    Returns:
      float: The function value at (x,y).
    """

    # I have simplified the expression a bit, but it should be
    # still same as in the previous script.
    return np.exp((-2/c**2 * ((x - x_0)**2 + (y - y_0)**2)))


class Target:
    """ Holds information about target and its placing.

    Attributes:

      x_s (int): targets start coordinate in x direction.
      x_e (int): targets end coordinate in x direction.
      y_s (int): targets start coordinate in y direction.
      y_e (int): targets end coordinate in y direction.
      trans (float): sample transmission.
    """

    def __init__(self, x_s, x_e, y_s, y_e, trans):
        """Initiates Target object

        takes arguments identically named as class attributes and sets
        them correspondingly.
        """
        self.x_s = x_s
        self.x_e = x_e
        self.y_s = y_s
        self.y_e = y_e
        self.trans = trans


class Simulation:
    """Keeps simulation parameters and data.

    Attributes:
      target (Target): Target configuration.
      grid_size (tuple) : shape of the grid as a tuple of two integers.
      cellsize_x: Cell size in x direction.
          Target length over grid size (both in x direction).
      cellsize_y: Cell size in y direction.
          Target length over grid size (both in y direction).
      energy (float): Photon energy in keV of processed simulation file.
      data (ndarray): Simulation data. Can be loaded from a file with
          the Simulation.load_data method.

    """
    def __init__(self, target, m, n, energy):
        """Initiates an Simulation object, sets attributes.

        Args:
          m (int): grid size in x direction.
          n (int): grid size in y direction.
          target (Target), time step, obs_energy: see attributes section
              in the class doc-string.
        """

        # set some attributes:
        self.target = target
        self.grid_size = (m, n)
        self.energy = energy

        # calculate cell sizes:
        self.cellsize_x = (target.x_e - target.x_s)/self.grid_size[0]
        self.cellsize_y = (target.y_e - target.y_s)/self.grid_size[1]

        # create an empty array for the simulation data:
        self.data = np.empty(self.grid_size)

    def load_data(self, path):
        """ Loads rotation data from a simulation data file(text)

        Args:
          path (string):  The path to the data file.
        """

        # self.data = ascii.read()
        # full_path = path  + 'Rotation' + str(self.timestep) + '.dat'
        with open(path, mode='rb') as rot_file:
            self.data = np.genfromtxt(rot_file, encoding=None)  # ? without enc.


class Configuration:
    """Contains information about the optical setup of the experiment.

    Attributes:
      an_position (float): Rotation in radians) of the analyzer from its default
          position (at pi/2 to the primary beam polarization).
      impurity (float): Beam polarization impurity. Part of the intensity
       remaining in the other polarization.
      an_extinction (float): Extinction of the analyzer.

      det_obs_energy (float): Observation energy in keV.
      det_trans_channel : Transmission of all channel cuts including spectral
         bandwidth # max trans*BW_cc/BW_lcls, asymm.
      det_pixel_size (float): Pixel size in micron.
      det_beam_width (float): beam width after lens B in micron.
          # what's lens B?
      m (float): magnification # ???
      n_0: source number of photons in total.
      trans_telescope (float): Transmission of CRLs due to beam size
          mismatch (asymmetry).
      ph_per_px_on_axis : Photons per pixel on axis only with lenses.
     """

    def __init__(self, an_position, impurity,
                 an_extinction, det_obs_energy, det_trans_channel,
                 det_pixel_size, det_beam_width, n_0, m, trans_telescope):
        """Initiates an Configuration object, sets attributes.

        Args: an_position, impurity, an_extinction, det_obs_energy,
          det_trans_channel, det_pixel_size, det_beam_width, n_0, m,
          trans_telescope:
              see attributes section in the class doc string.
        """

        self.an_position = an_position
        self.impurity = impurity
        self.an_extinction = an_extinction

        self.det_obs_energy = det_obs_energy
        self.det_trans_channel = det_trans_channel
        self.det_pixel_size = det_pixel_size
        self.det_beam_width = det_beam_width

        self.m = m  # what is it? magnification?
        self.n_0 = n_0
        self.trans_telescope = trans_telescope

        # Attributes set to None at first:
        self.ph_per_px_on_axis = None

    def calc_ph_per_px_on_axis(self, a=1.6e7, b=1e11, c=13):
        """Calculates number of photons per pixel on axis only with lenses.

        Function: a * n_0 *(1/b) * pixel_size^2/c^2.
        Args:
            a (optional, default = 1.6e7)
            b (optional, default = 1e11)
            c (optional, default = 13)
        """

        self.ph_per_px_on_axis = (a * self.n_0 / b
                                  * self.det_pixel_size**2 / c**2)
        return self.ph_per_px_on_axis


class Detection:
    """Used to compute the detected signal.


    Attributes:
    cfg (Configuration): optical configuration of the experiment.
    sim (Simulation): Simulation class object, containing
        simulation data and its parameters.
    rotation (ndarray): Simulated rotation, scaled to the observation energy.
    det_shape (tuple): Detector size as tuple (pixels in x, pixels in y).
    intensity_px (ndarray): intensity profile, on the detector pixels,
        generated by passing through the analyzer. Beam profile not included.
    beam_profile (ndarray): Beam intensity profile (without the analyzer).
    ideal_detector (ndarray): Signal on the detector, without noise.
    """

    def __init__(self, configuration, simulation):
        """Initiates an Detection object, sets attributes.

        Args:
          configuration: as 'cfg' in the class doc string.
          simulation:  as 'sim' in the class doc string.
        """
        self.cfg = configuration
        self.sim = simulation

        # Attributes set to None at first:
        self.rotation = None
        self.det_shape = None
        self.intensity_px = None
        self.beam_profile = None
        self.ideal_detector = None

    def calc_rotation(self):
        """Calculates rotation from the simulation data.

        Converts the simulated rotation to the observation energy by
        scaling it with an (sim_energy/obs_energy)^2 factor. Sets
        the rotation attribute.
        """

        self.rotation = (self.sim.data * self.sim.energy**2
                         / self.cfg.det_obs_energy**2)

    def calc_det_shape(self):
        """ Calculates detector shape, sets the 'det_shape' attribute."""
        spatial_resolution = self.cfg.det_pixel_size / self.cfg.m

        # why +1 ?
        px_x = math.ceil(((self.sim.target.x_e - self.sim.target.x_s)
                          / spatial_resolution) + 1)
        px_y = math.ceil(((self.sim.target.y_e - self.sim.target.y_s)
                          / spatial_resolution) + 1)

        self.det_shape = (px_y, px_x)

    def emulate_intensity(self):
        """Emulates influence of the analyzer on the intensity.

        It calculates the proportion of the intensity, left after passing
        through the analyzer, for every cell and transfers it to detector pixels.
        Bilinear interpolation is used for the rescaling.

        Sets the outcome to the 'intensity_px' attribute.
        """

        # angle between the direction of the rotated polarization
        # and the analyst position.
        # (changed) Left the '+'. After redefining angles should be a '-'.
        theta = (self.rotation - self.cfg.an_position)/1000

        # mrad -> rad => 1/1000 factor
        # for the light with the main polarization:
        intensity_1 = ((1 - self.cfg.impurity) *
                       (1 - (1 - self.cfg.an_extinction)
                        * np.cos(theta)**2))
        # for the light with the other polarization:
        intensity_2 = (self.cfg.impurity
                       * (1 - (1 - self.cfg.an_extinction)
                          * np.sin(theta)**2))
        intensity = intensity_1 + intensity_2

        # Transforming the values to pixels.
        # Rescaled with a bilinear interpolation (default for skimage).
        intensity = transform.resize(intensity, self.det_shape,
                                     anti_aliasing=False,
                                     # default is True, it probably
                                     # should be on.
                                     preserve_range=True,
                                     # default is False, not sure about it.
                                     )
        self.intensity_px = intensity

    def calc_beam_profile(self):
        """Calculates a beam profile. Sets it to the 'beam_profile' attr."""

        x = np.arange(1, self.det_shape[0] + 1)
        y = np.arange(1, self.det_shape[1] + 1)

        # c - scaling parameter ~ (standard deviation)^2
        c = self.cfg.det_beam_width / self.cfg.det_pixel_size

        # x[:, None], y [None, :] adds a second empty dimension.
        # Faster than np.meshgrid; non mixed expressions like x^2
        # are still calculated in 1D.
        beam_profile = _beam_profile_fkt(x[:, None], y[None, :],
                                         self.det_shape[0]/2,
                                         self.det_shape[1]/2, c)
        # scaling with transmission values:
        beam_profile = (self.cfg.ph_per_px_on_axis * self.sim.target.trans
                        * self.cfg.det_trans_channel * self.cfg.trans_telescope
                        * beam_profile)

        self.beam_profile = beam_profile

    def add_noise(self, accumulation=1, image=None, std=None):
        """Adds and accumulates a normal distributed noise to an image.

        Adds normally distributed noise to an image, by default to the ideal
        detection image (attr. 'ideal_detector'). The standard deviation is
        assumed to be sqrt(N) (Poisson statistics), but it can be specified.
        The method also allows to accumulate over meany noisy images.

        Args:
            accumulation (int, default=1): Number of noisy images to be
                generated and accumulated over. For 1 it just adds noise.
                For 0 the original image is returned.
            image (ndarray, optional): Base image, (The shape must be the
                same as 'det_shape'.) to which noise is supposed to be added.
                Set this parameter, if it's supposed to be different then
                'ideal_detector'.
            std (ndarray, optional): An array of standard deviations
                for each pixel. (Shape must be the same as 'det_shape').

        Returns:
            accumulated(ndarray): Accumulated image.
        """

        # No noise is added for accumulation = 0
        # Choosing the image for adding noise:
        if image is None:
            without_noise = self.ideal_detector
            # TODO Add sth for a situation, when image is not an ndarray.
        else:
            without_noise = image

        if accumulation == 0:
            return without_noise

        if std is None:
            # Set the standard deviation to sqrt(N),
            # if it's not explicitly specified:
            std = np.sqrt(without_noise)

        accumulated = np.zeros(self.det_shape)  # no accumulation yet.

        # Accumulating noise:
        for _ in range(accumulation):
            # Generate an detection image with some random noise:
            noise = ((without_noise + std
                      * np.random.randn(*self.det_shape)).astype('int32'))
            accumulated = accumulated + noise
        return accumulated

    def reobtain_rotation(self, detected_img, baseline_img):
        """Reobtains rotation from the detector output

        It divide the detector image by the reference image to obtain
        the intensity profile of the analyzer. The rotation is then
        calculated from it.

        Args:
            detected_img (ndarray): Main detector output, usually
            accumulated noisy images.
            baseline_img (ndarray): Reference image, obtained without
            the analyzer in the beam line.

        Returns:
            rotation(ndarray): The reobtained faraday rotation.
        """

        intensity = detected_img/baseline_img
        # Calculate cos^2(theta):
        # Theta is an angle between the default analyzer position
        # (full extinction - pi/2) and the main polarization direction.

        beta = self.cfg.an_extinction
        alpha = self.cfg.impurity
        cos_2theta = ((intensity - 0.5 * (1 + beta)) /
                      ((beta - 1) * (0.5 - alpha)))

        # get theta:
        # As long as the analyzer position angle (def. in same direction as rot)
        # is smaller than the smallest rotation, theta is greater then zero.
        # That means that there is no sign ambiguity when obtaining theta with
        # arccos.
        theta = 0.5 * np.arccos(cos_2theta)
        # Calculate rotation:
        # (changed) analyzer position +- correct definition -> sign in eq. changes.
        # Factor 1000 is for the rad -> mrad conversion.
        rotation = theta*1000 + self.cfg.an_position
        # It's on pixels. Would an interpolation back to the cells make
        # any sens?
        return rotation