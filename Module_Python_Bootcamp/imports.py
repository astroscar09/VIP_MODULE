import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord

#defining the plotting styles to use throughout my plotting functions
plt.style.use('ggplot')
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 15