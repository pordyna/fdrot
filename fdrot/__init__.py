"""fdrot package

Provides tools for calculating the Faraday Rotation from PIC simulation
data. It handles both 2D and 3D inputs. It supports a time resolved
integration of the effect.
"""

from . import sim_sequence
from . import spliters
from .kernel3d import kernel3d

""" fdrot package __init__ file.
This file is a part of the fdrot package.

Authors: Paweł Ordyna
"""
