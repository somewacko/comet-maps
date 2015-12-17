'''
NavcamInfo.py

Holds information about the NAVCAM on the Rosetta spacecraft.

'''

import numpy as np

# ---- Known parameters (from http://issfd.org/ISSFD_2009/AOCSII/Lauer.pdf)

focal_length_mm = 0.1525
field_of_view   = 5.0 # In degrees
image_width = 1024.0

# ---- Estimated parameters

focal_length_px = (image_width/2.0) / np.tan(np.deg2rad(field_of_view/2.0))
offset = (image_width-1.0)/2.0 # Assumed principal point
skew = 0.0 # Assumed

intrinsic = np.array([
    [focal_length_px,            skew, offset],
    [              0, focal_length_px, offset],
    [              0,               0,      1]
])

