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


def _beam_profile_fkt(x, y, x_0, y_0, c, asymmetry = 1):
    """Function defining the gaussian beam profile. (not normed)

    Args:
      x (float) : x coordinate.
      y (float) : y coordinate.
      x_0 (float): beam center (x coordinate).
      y_0 (float): beam center (y coordinate).
      asymmetry(float): beam width asymmetry. Ratio of the standard
          deviation in x direction to the standard deviation in y direction.

    Returns:
      float: The function value at (x,y).
    """

    # I have simplified the expression a bit, but it should be
    # still same as in the previous script.
    # add the asymmetry:
    std_x = c * np.sqrt(asymmetry)
    std_y = c / np.sqrt(asymmetry)
    profile = np.exp(-2**(-1) * ((x - x_0)**2 / std_x**2 + (y - y_0)**2 / std_y**2))
    return profile


class Target:
    """ Holds information about target.

    Attributes:
      trans (float): sample transmission.
    """

    def __init__(self, trans):
        """Initiates Target object

        takes arguments identically named as class attributes and sets
        them correspondingly.
        """
        self.trans = trans


class Simulation:
    """Keeps simulation parameters and data.

    Attributes:
      target (Target): Target configuration.

      ta_x_s (int): targets start coordinate in x direction.
      ta_x_e (int): targets end coordinate in x direction.
      ta_y_s (int): targets start coordinate in y direction.
      ta_y_e (int): targets end coordinate in y direction.
      grid_size (tuple) : shape of the grid as a tuple of two integers.

      cellsize_x: Cell size in x direction.
          Target length over grid size (both in x direction).
      cellsize_y: Cell size in y direction.
          Target length over grid size (both in y direction).
      energy (float): Photon energy in keV of processed simulation file.
      data (ndarray): Simulation data. Can be loaded from a file with
          the Simulation.load_data method.

    """
    def __init__(self, target, ta_x_s, ta_x_e, ta_y_s, ta_y_e, m, n, energy):
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
        # set target dimensions.
        self.ta_x_s = ta_x_s
        self.ta_x_e = ta_x_e
        self.ta_y_s = ta_y_s
        self.ta_y_e = ta_y_e

        # calculate cell sizes:
        self.cellsize_x = (ta_x_e - ta_x_s)/self.grid_size[0]
        self.cellsize_y = (ta_y_e - ta_y_s)/self.grid_size[1]

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
      det_beam_width (optional, float): beam width at the detector in micron.
          It has to be set, if the beam distribution was not specified and
          the default gauss profile is in use.
      m (float): magnification # ???
      n_0: source number of photons in total.
      trans_telescope (float): Transmission of CRLs due to beam size
          mismatch (asymmetry).

      beam_distribution (function): Function used for calculating the
              beam_profile. It should take as the first 2 positional
              arguments: x, y, x_0, y_0; x_0 and y_0 are the coordinates of
              the beam center. It should work with x,y as numpy arrays
              (1D +1(empty dim.)).
              It should return the distribution value at (x,y), or an array
              of values if x,y are arrays.
              If not specified  a gaussian profile is used. The number of photons
              at the axis (per pixel) and the beam width has to be specified.
      beam_position (tuple) : Position of the beam on the probe and detector;
          both values should be between  0 and 1. Set (0.5, 0.5) for a centered
          beam.
      ph_per_px_on_axis : Photons per pixel on axis only with lenses
     """

    def __init__(self, an_position, impurity,
                 an_extinction, det_obs_energy, det_trans_channel,
                 det_pixel_size, n_0, m, trans_telescope, det_beam_width =None,
                 beam_distribution=None, beam_position=None):
        """Initiates an Configuration object, sets attributes.

        Args: an_position, impurity, an_extinction, det_obs_energy,
          det_trans_channel, det_pixel_size, n_0, m,
          trans_telescope.det_beam_width (optional),
          beam_distribution (optional)
          beam_position (optional):
          see attributes section in the class doc string.
        """

        self.an_position = an_position
        self.impurity = impurity
        self.an_extinction = an_extinction

        self.det_obs_energy = det_obs_energy
        self.det_trans_channel = det_trans_channel
        self.det_pixel_size = det_pixel_size
        self.det_beam_width = det_beam_width

        if beam_position is None:
            self.beam_position = (0.5, 0.5)
        else:
            self.beam_position = beam_position
        if beam_distribution is None:
            if det_beam_width is None:
                raise Exception('det_beam_width has to be set,'
                                ' when the beam_distribution is not given!')
            self.beam_distribution = _beam_profile_fkt
        else:
            self.beam_distribution = beam_distribution
        self.m = m  #  rename to magnification !
        self.n_0 = n_0
        self.trans_telescope = trans_telescope

        # Attributes set to None at first:
        self.ph_per_px_on_axis = None


    def calc_ph_per_px_on_axis(self, a=1.6e7, b=1e11, c=13):
            """Rescales number of photons on axis per pixel for a new setup.

            When the distribution of the initial photons, after passing through
            the optical instruments, is not known, a gaussian beam with a specified
            number of photons per pixel at the beam center can be used. This number
            has to be calculated externally. This method allows simple rescaling to
            the new initial intensity and a different pixel size.

            Args:
                a (optional, default = 1.6e7): Calculation outcome (number
                    of photons per pixel on axis).
                b (optional, default = 1e11): Initial number of photons of the
                    initial calculation.
                c (optional, default = 13): Pixel size (side length) of the initial
                    calculation.
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
        px_x = math.ceil(((self.sim.ta_x_e - self.sim.ta_x_s)
                          / spatial_resolution) + 1)
        px_y = math.ceil(((self.sim.ta_y_e - self.sim.ta_y_s)
                          / spatial_resolution) + 1)

        self.det_shape = (px_y, px_x)


    def emulate_intensity(self, order=1):
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
                                     order=order,
                                     mode='reflect',
                                     anti_aliasing=False,
                                     # default is True, it probably
                                     # should be on.
                                     preserve_range=True,
                                     # default is False, not sure about it.
                                     )
        self.intensity_px = intensity

    def calc_beam_profile(self, position=None, *args, **kwargs):
        """Calculates a beam profile. Sets it to the 'beam_profile' attr."""

        if position is not None:
            self.cfg.beam_position = position
        # values should be calculated for a pixel center:
        x = np.arange(0, self.det_shape[0]) + 0.5
        y = np.arange(0, self.det_shape[1]) + 0.5



        x_0 = self.det_shape[0] * self.cfg.beam_position[0]
        y_0 = self.det_shape[1] * self.cfg.beam_position[1]


        # Backwards compatibility for the default distribution. When user
        # doesn't specify the distribution.
        if self.cfg.beam_distribution == _beam_profile_fkt:
            # c - scaling parameter - standard deviation
            c = np.sqrt(self.cfg.det_beam_width / self.cfg.det_pixel_size)
            args = (c,)
        # This default distribution is not normalized. It's equal to 1
        # at the beam center. Photons per pixel on axis are used as
        # the scaling parameter. (It has to be calculated separately,
        # for a given CLR system).
            scaling = self.cfg.ph_per_px_on_axis
        else:
            scaling = self.cfg.n_0

        # x[:, None], y [None, :] adds a second empty dimension.
        # Faster than np.meshgrid; non mixed expressions like x^2
        # are still calculated in 1D.
        beam_profile = self.cfg.beam_distribution(x[:, None], y[None, :],
                                                  x_0, y_0, *args, **kwargs)
        # scaling with transmission values:
        beam_profile = (scaling * self.sim.target.trans
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
            rotation(ndarray): The rephobtained faraday rotation.
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
