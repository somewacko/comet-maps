'''
__init__.py for sfm module.

'''

from ImageInstance import *
from StructureFromMotion import *

# Configurable parameters for SfM

SHOW_FEATURES       = False
SHOW_MATCHES        = False
SHOW_EPILINES       = False
SHOW_CORRESPONDENCE = False
SHOW_POINT_CLOUD    = False
SHOW_DEPTH_MAP      = False

SIFT_MATCHING_RATIO = 0.8
CORRESPONDENCE_SKIP = 16
WINDOW_RADIUS       = 10
FILTER_POINT_CLOUD  = True

