import numpy as np
from numpy.fft import fftfreq, fft, ifft
import matplotlib.pyplot as plt
import scipy.special
import scipy
import numba as nb
#import jax.numpy as jnp
#import jax
from functools import partial
#from jax.experimental import loops
from matplotlib.colors import Normalize
from tqdm import tqdm
from math import factorial
import math
#from google.colab import files
import scipy

# jax.config.update('jax_platform_name', 'cpu')
#jax.config.update('jax_enable_x64', True)

BC = 10 * np.pi
QMIN, QMAX = -BC, BC
N = 2**11
dQ = (QMAX - QMIN) / (N - 1)
hbar = 1.0 
T_kick = 0.05
k = 3.0
xf = 1.2
xb = 1.0
twopi = 2 * np.pi
step = 0
e2 = k * xf * (np.exp(-8 * xb * (2 * xf - xb)) / (1 - np.exp(-32 * xf * xb)))

# 量子BCH近似のそれぞれの項．（5次の項まで書かれている．）
def _bchHamitonianTerm(x, y, order):
    if order == 1:
        return x + y
    elif order == 2:
        return (x@y - y@x)/2
    elif order == 3:
        return x@x@y/12 - x@y@x/6 + x@y@y/12 + y@x@x/12 - y@x@y/6 + y@y@x/12
    elif order == 4:
        return x@x@y@y/24 - x@y@x@y/12 + y@x@y@x/12 - y@y@x@x/24
    elif order == 5:
        return (
            - x@x@x@x@y/720 + x@x@x@y@x/180 + x@x@x@y@y/180 - x@x@y@x@x/120 - x@x@y@x@y/120 - x@x@y@y@x/120 +
            x@x@y@y@y/180 + x@y@x@x@x/180 - x@y@x@x@y/120 + x@y@x@y@x/30 - x@y@x@y@y/120 - x@y@y@x@x/120 - x@y@y@x@y/120 +
            x@y@y@y@x/180 - x@y@y@y@y/720 - y@x@x@x@x/720 + y@x@x@x@y/180 - y@x@x@y@x/120 - y@x@x@y@y/120 - y@x@y@x@x/120 +
            y@x@y@x@y/30 - y@x@y@y@x/120 + y@x@y@y@y/180 + y@y@x@x@x/180 - y@y@x@x@y/120 - y@y@x@y@x/120 - y@y@x@y@y/120 +
            y@y@y@x@x/180 + y@y@y@x@y/180 - y@y@y@y@x/720
        )
    elif order == 6:
        return (
            (-1/60) * y @ x @ y @ x @ y @ x + (-1/240) * x @ x @ y @ x @ x @ y + (-1/240) * x @ x @ y @ x @ y @ y + (-1/240) * x @ x @ y @ y @ x @ y + (-1/240) * x @ y @ x @ x @ y @ y + (-1/240) * x @ y @ y @ x @ x @ y + (-1/240) * x @ y @ y @ x @ y @ y + (-1/360) * y @ x @ x @ x @ y @ x + (-1/360) * y @ x @ y @ x @ x @ x + (-1/360) * y @ x @ y @ y @ y @ x + (-1/360) * y @ y @ y @ x @ x @ x + (-1/360) * y @ y @ y @ x @ y @ x + (-1/1440) * x @ x @ x @ x @ y @ y + (-1/1440) * x @ x @ y @ y @ y @ y +
            (1/1440) * y @ y @ x @ x @ x @ x + (1/1440) * y @ y @ y @ y @ x @ x + (1/360) * x @ x @ x @ y @ x @ y + (1/360) * x @ x @ x @ y @ y @ y + (1/360) * x @ y @ x @ x @ x @ y + (1/360) * x @ y @ x @ y @ y @ y + (1/360) * x @ y @ y @ y @ x @ y +
            (1/240) * y @ x @ x @ y @ x @ x + (1/240) * y @ x @ x @ y @ y @ x + (1/240) * y @ x @ y @ y @ x @ x + (1/240) *
            y @ y @ x @ x @ y @ x + (1/240) * y @ y @ x @ y @ x @ x + (
                1/240) * y @ y @ x @ y @ y @ x + (1/60) * x @ y @ x @ y @ x @ y
        )
    elif order == 7:
        return ((-1/140) * x @ y @ x @ y @ x @ y @ x + (-1/140) * y @ x @ y @ x @ y @ x @ y + (-1/630) * x @ x @ x @ y @ x @ y @ x + (-1/630) * x @ y @ x @ x @ x @ y @ x + (-1/630) * x @ y @ x @ y @ x @ x @ x + (-1/630) * x @ y @ x @ y @ y @ y @ x + (-1/630) * x @ y @ y @ y @ x @ y @ x + (-1/630) * y @ x @ x @ x @ y @ x @ y + (-1/630) * y @ x @ y @ x @ x @ x @ y + (-1/630) * y @ x @ y @ x @ y @ y @ y + (-1/630) * y @ x @ y @ y @ y @ x @ y + (-1/630) * y @ y @ y @ x @ y @ x @ y + (-1/1120) * x @ x @ y @ x @ x @ y @ y + (-1/1120) * x @ x @ y @ y @ x @ x @ y + (-1/1120) * x @ x @ y @ y @ x @ y @ y + (-1/1120) * x @ y @ y @ x @ x @ y @ y + (-1/1120) * y @ x @ x @ y @ y @ x @ x + (-1/1120) * y @ y @ x @ x @ y @ x @ x + (-1/1120) * y @ y @ x @ x @ y @ y @ x + (-1/1120) * y @ y @ x @ y @ y @ x @ x + (-1/1512) * x @ x @ x @ y @ x @ x @ x + (-1/1512) * x @ x @ x @ y @ y @ y @ x + (-1/1512) * x @ y @ y @ y @ x @ x @ x + (-1/1512) * y @ x @ x @ x @ y @ y @ y + (-1/1512) * y @ y @ y @ x @ x @ x @ y + (-1/1512) * y @ y @ y @ x @ y @ y @ y + (-1/5040) * x @ x @ x @ x @ x @ y @ x + (-1/5040) * x @ x @ x @ x @ x @ y @ y + (-1/5040) * x @ x @ x @ y @ x @ x @ y + (-1/5040) * x @ x @ x @ y @ x @ y @ y + (-1/5040) * x @ x @ x @ y @ y @ x @ x + (-1/5040) * x @ x @ x @ y @ y @ x @ y + (-1/5040) * x @ x @ y @ x @ x @ x @ y + (-1/5040) * x @ x @ y @ x @ y @ y @ y + (-1/5040) * x @ x @ y @ y @ x @ x @ x + (-1/5040) * x @ x @ y @ y @ y @ x @ x + (-1/5040) * x @ x @ y @ y @ y @ x @ y + (-1/5040) * x @ x @ y @ y @ y @ y @ y + (-1/5040) * x @ y @ x @ x @ x @ x @ x + (-1/5040) * x @ y @ x @ x @ x @ y @ y + (-1/5040) * x @ y @ x @ x @ y @ y @ y + (-1/5040) * x @ y @ y @ x @ x @ x @ y + (-1/5040) * x @ y @ y @ x @ y @ y @ y + (-1/5040) * x @ y @ y @ y @ x @ x @ y + (-1/5040) * x @ y @ y @ y @ x @ y @ y + (-1/5040) * x @ y @ y @ y @ y @ y @ x + (-1/5040) * y @ x @ x @ x @ x @ x @ y + (-1/5040) * y @ x @ x @ x @ y @ x @ x + (-1/5040) * y @ x @ x @ x @ y @ y @ x + (-1/5040) * y @ x @ x @ y @ x @ x @ x + (-1/5040) * y @ x @ x @ y @ y @ y @ x + (-1/5040) * y @ x @ y @ y @ x @ x @ x + (-1/5040) * y @ x @ y @ y @ y @ x @ x + (-1/5040) * y @ x @ y @ y @ y @ y @ y + (-1/5040) * y @ y @ x @ x @ x @ x @ x + (-1/5040) * y @ y @ x @ x @ x @ y @ x + (-1/5040) * y @ y @ x @ x @ x @ y @ y + (-1/5040) * y @ y @ x @ x @ y @ y @ y + (-1/5040) * y @ y @ x @ y @ x @ x @ x + (-1/5040) * y @ y @ x @ y @ y @ y @ x + (-1/5040) * y @ y @ y @ x @ x @ y @ x + (-1/5040) * y @ y @ y @ x @ x @ y @ y + (-1/5040) * y @ y @ y @ x @ y @ x @ x + (-1/5040) * y @ y @ y @ x @ y @ y @ x + (-1/5040) * y @ y @ y @ y @ y @ x @ x + (-1/5040) * y @ y @ y @ y @ y @ x @ y + (1/30240) * x @ x @ x @ x @ x @ x @ y + (1/30240) * x @ y @ y @ y @ y @ y @ y + (1/30240) * y @ x @ x @ x @ x @ x @ x + (1/30240) * y @ y @ y @ y @ y @ y @ x + (1/3780) * x @ x @ x @ x @ y @ y @ y + (1/3780) * x @ x @ x @ y @ y @ y @ y + (1/3780) * y @ y @ y @ x @ x @ x @ x + (1/3780) * y @ y @ y @ y @ x @ x @ x + (1/2016) * x @ x @ x @ x @ y @ x @ x + (1/2016) * x @ x @ x @ x @ y @ x @ y + (1/2016) * x @ x @ x @ x @ y @ y @ x + (1/2016) * x @ x @ y @ x @ x @ x @ x + (1/2016) * x @ x @ y @ y @ y @ y @ x + (1/2016) * x @ y @ x @ x @ x @ x @ y + (1/2016) * x @ y @ x @ y @ y @ y @ y + (1/2016) * x @ y @ y @ x @ x @ x @ x + (1/2016) * x @ y @ y @ y @ y @ x @ x + (1/2016) * x @ y @ y @ y @ y @ x @ y + (1/2016) * y @ x @ x @ x @ x @ y @ x + (1/2016) * y @ x @ x @ x @ x @ y @ y + (1/2016) * y @ x @ x @ y @ y @ y @ y + (1/2016) * y @ x @ y @ x @ x @ x @ x + (1/2016) * y @ x @ y @ y @ y @ y @ x + (1/2016) * y @ y @ x @ x @ x @ x @ y + (1/2016) * y @ y @ x @ y @ y @ y @ y + (1/2016) * y @ y @ y @ y @ x @ x @ y + (1/2016) * y @ y @ y @ y @ x @ y @ x + (1/2016) * y @ y @ y @ y @ x @ y @ y + (1/840) * x @ x @ y @ x @ x @ y @ x + (1/840) * x @ x @ y @ x @ y @ x @ x + (1/840) * x @ x @ y @ x @ y @ x @ y + (1/840) * x @ x @ y @ x @ y @ y @ x + (1/840) * x @ x @ y @ y @ x @ y @ x + (1/840) * x @ y @ x @ x @ y @ x @ x + (1/840) * x @ y @ x @ x @ y @ x @ y + (1/840) * x @ y @ x @ x @ y @ y @ x + (1/840) * x @ y @ x @ y @ x @ x @ y + (1/840) * x @ y @ x @ y @ x @ y @ y + (1/840) * x @ y @ x @ y @ y @ x @ x + (1/840) * x @ y @ x @ y @ y @ x @ y + (1/840) * x @ y @ y @ x @ x @ y @ x + (1/840) * x @ y @ y @ x @ y @ x @ x + (1/840) * x @ y @ y @ x @ y @ x @ y + (1/840) * x @ y @ y @ x @ y @ y @ x + (1/840) * y @ x @ x @ y @ x @ x @ y + (1/840) * y @ x @ x @ y @ x @ y @ x + (1/840) * y @ x @ x @ y @ x @ y @ y + (1/840) * y @ x @ x @ y @ y @ x @ y + (1/840) * y @ x @ y @ x @ x @ y @ x + (1/840) * y @ x @ y @ x @ x @ y @ y + (1/840) * y @ x @ y @ x @ y @ x @ x + (1/840) * y @ x @ y @ x @ y @ y @ x + (1/840) * y @ x @ y @ y @ x @ x @ y + (1/840) * y @ x @ y @ y @ x @ y @ x + (1/840) * y @ x @ y @ y @ x @ y @ y + (1/840) * y @ y @ x @ x @ y @ x @ y + (1/840) * y @ y @ x @ y @ x @ x @ y + (1/840) * y @ y @ x @ y @ x @ y @ x + (1/840) * y @ y @ x @ y @ x @ y @ y + (1/840) * y @ y @ x @ y @ y @ x @ y)
        '''
        return (
            (-1/140) * x @ y @ x @ y @ x @ y @ x + (-1/140) * y @ x @ y @ x @ y @ x @ y + (-1/630) * x @ x @ x @ y @ x @ y @ x + (-1/630) * x @ y @ x @ x @ x @ y @ x + (-1/630) * x @ y @ x @ y @ x @ x @ x + (-1/630) * x @ y @ x @ y @ y @ y @ x + (-1/630) * x @ y @ y @ y @ x @ y @ x + (-1/630) * y @ x @ x @ x @ y @ x @ y + (-1/630) * y @ x @ y @ x @ x @ x @ y + (-1/630) * y @ x @ y @ x @ y @ y @ y + (-1/630) * y @ x @ y @ y @ y @ x @ y + (-1/630) * y @ y @ y @ x @ y @ x @ y + (-1/1120) * x @ x @ y @ x @ x @ y @ y + (-1/1120) * x @ x @ y @ y @ x @ x @ y + (-1/1120) * x @ x @ y @ y @ x @ y @ y + (-1/1120) * x @ y @ y @ x @ x @ y @ y + (-1/1120) * y @ x @ x @ y @ y @ x @ x + (-1/1120) * y @ y @ x @ x @ y @ x @ x + (-1/1120) * y @ y @ x @ x @ y @ y @ x + (-1/1120) * y @ y @ x @ y @ y @ x @ x + (-1/1512) * x @ x @ x @ y @ x @ x @ x + (-1/1512) * x @ x @ x @ y @ y @ y @ x + (-1/1512) * x @ y @ y @ y @ x @ x @ x + (-1/1512) * y @ x @ x @ x @ y @ y @ y + (-1/1512) * y @ y @ y @ x @ x @ x @ y + (-1/1512) * y @ y @ y @ x @ y @ y @ y + (-1/5040) * x @ x @ x @ x @ x @ y @ x + (-1/5040) * x @ x @ x @ x @ x @ y @ y + (-1/5040) * x @ x @ x @ y @ x @ x @ y + (-1/5040) * x @ x @ x @ y @ x @ y @ y + (-1/5040) * x @ x @ x @ y @ y @ x @ x + (-1/5040) * x @ x @ x @ y @ y @ x @ y + (-1/5040) * x @ x @ y @ x @ x @ x @ y + (-1/5040) * x @ x @ y @ x @ y @ y @ y + (-1/5040) * x @ x @ y @ y @ x @ x @ x + (-1/5040) * x @ x @ y @ y @ y @ x @ x + (-1/5040) * x @ x @ y @ y @ y @ x @ y + (-1/5040) * x @ x @ y @ y @ y @ y @ y + (-1/5040) * x @ y @ x @ x @ x @ x @ x + (-1/5040) * x @ y @ x @ x @ x @ y @ y + (-1/5040) * x @ y @ x @ x @ y @ y @ y + (-1/5040) * x @ y @ y @ x @ x @ x @ y + (-1/5040) * x @ y @ y @ x @ y @ y @ y + (-1/5040) * x @ y @ y @ y @ x @ x @ y + (-1/5040) * x @ y @ y @ y @ x @ y @ y + (-1/5040) * x @ y @ y @ y @ y @ y @ x + (-1/5040) * y @ x @ x @ x @ x @ x @ y + (-1/5040) * y @ x @ x @ x @ y @ x @ x + (-1/5040) * y @ x @ x @ x @ y @ y @ x + (-1/5040) * y @ x @ x @ y @ x @ x @ x + (-1/5040) * y @ x @ x @ y @ y @ y @ x + (-1/5040) * y @ x @ y @ y @ x @ x @ x + (-1/5040) * y @ x @ y @ y @ y @ x @ x + (-1/5040) * y @ x @ y @ y @ y @ y @ y + (-1/5040) * y @ y @ x @ x @ x @ x @ x + (-1/5040) * y @ y @ x @ x @ x @ y @ x + (-1/5040) * y @ y @ x @ x @ x @ y @ y + (-1/5040) * y @ y @ x @ x @ y @ y @ y + (-1/5040) * y @ y @ x @ y @ x @ x @ x + (-1/5040) * y @ y @ x @ y @ y @ y @ x + (-1/5040) * y @ y @ y @ x @ x @ y @ x + (-1/5040) * y @ y @ y @ x @ x @ y @ y +
            (-1/5040) * y @ y @ y @ x @ y @ x @ x + (-1/5040) * y @ y @ y @ x @ y @ y @ x + (-1/5040) * y @ y @ y @ y @ y @ x @ x + (-1/5040) * y @ y @ y @ y @ y @ x @ y + (1/30240) * x @ x @ x @ x @ x @ x @ y + (1/30240) * x @ y @ y @ y @ y @ y @ y + (1/30240) * y @ x @ x @ x @ x @ x @ x + (1/30240) * y @ y @ y @ y @ y @ y @ x + (1/3780) * x @ x @ x @ x @ y @ y @ y + (1/3780) * x @ x @ x @ y @ y @ y @ y + (1/3780) * y @ y @ y @ x @ x @ x @ x + (1/3780) * y @ y @ y @ y @ x @ x @ x + (1/2016) * x @ x @ x @ x @ y @ x @ x + (1/2016) * x @ x @ x @ x @ y @ x @ y + (1/2016) * x @ x @ x @ x @ y @ y @ x + (1/2016) * x @ x @ y @ x @ x @ x @ x + (1/2016) * x @ x @ y @ y @ y @ y @ x + (1/2016) * x @ y @ x @ x @ x @ x @ y + (1/2016) * x @ y @ x @ y @ y @ y @ y + (1/2016) * x @ y @ y @ x @ x @ x @ x + (1/2016) * x @ y @ y @ y @ y @ x @ x + (1/2016) * x @ y @ y @ y @ y @ x @ y + (1/2016) * y @ x @ x @ x @ x @ y @ x + (1/2016) * y @ x @ x @ x @ x @ y @ y + (1/2016) * y @ x @ x @ y @ y @ y @ y + (1/2016) * y @ x @ y @ x @ x @ x @ x + (1/2016) * y @ x @ y @ y @ y @ y @ x + (1/2016) * y @ y @ x @ x @ x @ x @ y + (1/2016) * y @ y @ x @ y @ y @ y @ y + (1/2016) * y @ y @ y @ y @ x @ x @ y + (1/2016) * y @ y @ y @ y @ x @ y @ x + (1/2016) *
            y @ y @ y @ y @ x @ y @ y + (1/840) * x @ x @ y @ x @ x @ y @ x + (1/840) * x @ x @ y @ x @ y @ x @ x + (1/840) * x @ x @ y @ x @ y @ x @ y + (1/840) * x @ x @ y @ x @ y @ y @ x + (1/840) * x @ x @ y @ y @ x @ y @ x + (1/840) * x @ y @ x @ x @ y @ x @ x + (1/840) * x @ y @ x @ x @ y @ x @ y + (1/840) * x @ y @ x @ x @ y @ y @ x + (1/840) * x @ y @ x @ y @ x @ x @ y + (1/840) * x @ y @ x @ y @ x @ y @ y + (1/840) * x @ y @ x @ y @ y @ x @ x + (1/840) * x @ y @ x @ y @ y @ x @ y + (1/840) * x @ y @ y @ x @ x @ y @ x + (1/840) * x @ y @ y @ x @ y @ x @ x + (1/840) * x @ y @ y @ x @ y @ x @ y + (1/840) * x @ y @ y @ x @ y @ y @ x + (
                1/840) * y @ x @ x @ y @ x @ x @ y + (1/840) * y @ x @ x @ y @ x @ y @ x + (1/840) * y @ x @ x @ y @ x @ y @ y + (1/840) * y @ x @ x @ y @ y @ x @ y + (1/840) * y @ x @ y @ x @ x @ y @ x + (1/840) * y @ x @ y @ x @ x @ y @ y + (1/840) * y @ x @ y @ x @ y @ x @ x + (1/840) * y @ x @ y @ x @ y @ y @ x + (1/840) * y @ x @ y @ y @ x @ x @ y + (1/840) * y @ x @ y @ y @ x @ y @ x + (1/840) * y @ x @ y @ y @ x @ y @ y + (1/840) * y @ y @ x @ x @ y @ x @ y + (1/840) * y @ y @ x @ y @ x @ x @ y + (1/840) * y @ y @ x @ y @ x @ y @ x + (1/840) * y @ y @ x @ y @ x @ y @ y + (1/840) * y @ y @ x @ y @ y @ x @ y
        ) '''
    elif order == 8:
        return ((-1/280) * x @ y @ x @ y @ x @ y @ x @ y + (-1/1260) * x @ x @ x @ y @ x @ y @ x @ y + (-1/1260) * x @ y @ x @ x @ x @ y @ x @ y + (-1/1260) * x @ y @ x @ y @ x @ x @ x @ y + (-1/1260) * x @ y @ x @ y @ x @ y @ y @ y + (-1/1260) * x @ y @ x @ y @ y @ y @ x @ y + (-1/1260) * x @ y @ y @ y @ x @ y @ x @ y + (-1/1680) * y @ x @ x @ y @ x @ x @ y @ x + (-1/1680) * y @ x @ x @ y @ x @ y @ x @ x + (-1/1680) * y @ x @ x @ y @ x @ y @ y @ x + (-1/1680) * y @ x @ x @ y @ y @ x @ y @ x + (-1/1680) * y @ x @ y @ x @ x @ y @ x @ x + (-1/1680) * y @ x @ y @ x @ x @ y @ y @ x + (-1/1680) * y @ x @ y @ x @ y @ y @ x @ x + (-1/1680) * y @ x @ y @ y @ x @ x @ y @ x + (-1/1680) * y @ x @ y @ y @ x @ y @ x @ x + (-1/1680) * y @ x @ y @ y @ x @ y @ y @ x + (-1/1680) * y @ y @ x @ x @ y @ x @ y @ x + (-1/1680) * y @ y @ x @ y @ x @ x @ y @ x + (-1/1680) * y @ y @ x @ y @ x @ y @ x @ x + (-1/1680) * y @ y @ x @ y @ x @ y @ y @ x + (-1/1680) * y @ y @ x @ y @ y @ x @ y @ x + (-1/2240) * x @ x @ y @ y @ x @ x @ y @ y + (-1/3024) * x @ x @ x @ y @ x @ x @ x @ y + (-1/3024) * x @ x @ x @ y @ x @ y @ y @ y + (-1/3024) * x @ x @ x @ y @ y @ y @ x @ y + (-1/3024) * x @ y @ x @ x @ x @ y @ y @ y + (-1/3024) * x @ y @ y @ y @ x @ x @ x @ y + (-1/3024) * x @ y @ y @ y @ x @ y @ y @ y + (-1/4032) * y @ x @ x @ x @ x @ y @ x @ x + (-1/4032) * y @ x @ x @ x @ x @ y @ y @ x + (-1/4032) * y @ x @ x @ y @ x @ x @ x @ x + (-1/4032) * y @ x @ x @ y @ y @ y @ y @ x + (-1/4032) * y @ x @ y @ y @ x @ x @ x @ x + (-1/4032) * y @ x @ y @ y @ y @ y @ x @ x + (-1/4032) * y @ y @ x @ x @ x @ x @ y @ x + (-1/4032) * y @ y @ x @ y @ x @ x @ x @ x + (-1/4032) * y @ y @ x @ y @ y @ y @ y @ x + (-1/4032) * y @ y @ y @ y @ x @ x @ y @ x + (-1/4032) * y @ y @ y @ y @ x @ y @ x @ x + (-1/4032) * y @ y @ y @ y @ x @ y @ y @ x + (-23/120960) * y @ y @ y @ y @ x @ x @ x @ x + (-1/10080) * x @ x @ x @ x @ x @ y @ x @ y + (-1/10080) * x @ x @ x @ x @ x @ y @ y @ y + (-1/10080) * x @ x @ x @ y @ x @ x @ y @ y + (-1/10080) * x @ x @ x @ y @ y @ x @ x @ y + (-1/10080) * x @ x @ x @ y @ y @ x @ y @ y + (-1/10080) * x @ x @ x @ y @ y @ y @ y @ y + (-1/10080) * x @ x @ y @ x @ x @ x @ y @ y + (-1/10080) * x @ x @ y @ x @ x @ y @ y @ y + (-1/10080) * x @ x @ y @ y @ x @ x @ x @ y + (-1/10080) * x @ x @ y @ y @ x @ y @ y @ y + (-1/10080) * x @ x @ y @ y @ y @ x @ x @ y + (-1/10080) * x @ x @ y @ y @ y @ x @ y @ y + (-1/10080) * x @ y @ x @ x @ x @ x @ x @ y + (-1/10080) * x @ y @ x @ y @ y @ y @ y @ y + (-1/10080) * x @ y @ y @ x @ x @ x @ y @ y + (-1/10080) * x @ y @ y @ x @ x @ y @ y @ y + (-1/10080) * x @ y @ y @ y @ x @ x @ y @ y + (-1/10080) * x @ y @ y @ y @ y @ y @ x @ y + (-1/60480) * y @ y @ x @ x @ x @ x @ x @ x + (-1/60480) * y @ y @ y @ y @ y @ y @ x @ x + (1/60480) * x @ x @ x @ x @ x @ x @ y @ y + (1/60480) * x @ x @ y @ y @ y @ y @ y @ y + (1/10080) * y @ x @ x @ x @ x @ x @ y @ x + (1/10080) * y @ x @ x @ x @ y @ y @ x @ x + (1/10080) * y @ x @ x @ y @ y @ x @ x @ x + (1/10080) * y @ x @ x @ y @ y @ y @ x @ x + (1/10080) * y @ x @ y @ x @ x @ x @ x @ x + (1/10080) * y @ x @ y @ y @ y @ y @ y @ x + (1/10080) * y @ y @ x @ x @ x @ y @ x @ x + (1/10080) * y @ y @ x @ x @ x @ y @ y @ x + (1/10080) * y @ y @ x @ x @ y @ x @ x @ x + (1/10080) * y @ y @ x @ x @ y @ y @ y @ x + (1/10080) * y @ y @ x @ y @ y @ x @ x @ x + (1/10080) * y @ y @ x @ y @ y @ y @ x @ x + (1/10080) * y @ y @ y @ x @ x @ x @ x @ x + (1/10080) * y @ y @ y @ x @ x @ y @ x @ x + (1/10080) * y @ y @ y @ x @ x @ y @ y @ x + (1/10080) * y @ y @ y @ x @ y @ y @ x @ x + (1/10080) * y @ y @ y @ y @ y @ x @ x @ x + (1/10080) * y @ y @ y @ y @ y @ x @ y @ x + (23/120960) * x @ x @ x @ x @ y @ y @ y @ y + (1/4032) * x @ x @ x @ x @ y @ x @ x @ y + (1/4032) * x @ x @ x @ x @ y @ x @ y @ y + (1/4032) * x @ x @ x @ x @ y @ y @ x @ y + (1/4032) * x @ x @ y @ x @ x @ x @ x @ y + (1/4032) * x @ x @ y @ x @ y @ y @ y @ y + (1/4032) * x @ x @ y @ y @ y @ y @ x @ y + (1/4032) * x @ y @ x @ x @ x @ x @ y @ y + (1/4032) * x @ y @ x @ x @ y @ y @ y @ y + (1/4032) * x @ y @ y @ x @ x @ x @ x @ y + (1/4032) * x @ y @ y @ x @ y @ y @ y @ y + (1/4032) * x @ y @ y @ y @ y @ x @ x @ y + (1/4032) * x @ y @ y @ y @ y @ x @ y @ y + (1/3024) * y @ x @ x @ x @ y @ x @ x @ x + (1/3024) * y @ x @ x @ x @ y @ y @ y @ x + (1/3024) * y @ x @ y @ y @ y @ x @ x @ x + (1/3024) * y @ y @ y @ x @ x @ x @ y @ x + (1/3024) * y @ y @ y @ x @ y @ x @ x @ x + (1/3024) * y @ y @ y @ x @ y @ y @ y @ x + (1/2240) * y @ y @ x @ x @ y @ y @ x @ x + (1/1680) * x @ x @ y @ x @ x @ y @ x @ y + (1/1680) * x @ x @ y @ x @ y @ x @ x @ y + (1/1680) * x @ x @ y @ x @ y @ x @ y @ y + (1/1680) * x @ x @ y @ x @ y @ y @ x @ y + (1/1680) * x @ x @ y @ y @ x @ y @ x @ y + (1/1680) * x @ y @ x @ x @ y @ x @ x @ y + (1/1680) * x @ y @ x @ x @ y @ x @ y @ y + (1/1680) * x @ y @ x @ x @ y @ y @ x @ y + (1/1680) * x @ y @ x @ y @ x @ x @ y @ y + (1/1680) * x @ y @ x @ y @ y @ x @ x @ y + (1/1680) * x @ y @ x @ y @ y @ x @ y @ y + (1/1680) * x @ y @ y @ x @ x @ y @ x @ y + (1/1680) * x @ y @ y @ x @ y @ x @ x @ y + (1/1680) * x @ y @ y @ x @ y @ x @ y @ y + (1/1680) * x @ y @ y @ x @ y @ y @ x @ y + (1/1260) * y @ x @ x @ x @ y @ x @ y @ x + (1/1260) * y @ x @ y @ x @ x @ x @ y @ x + (1/1260) * y @ x @ y @ x @ y @ x @ x @ x + (1/1260) * y @ x @ y @ x @ y @ y @ y @ x + (1/1260) * y @ x @ y @ y @ y @ x @ y @ x + (1/1260) * y @ y @ y @ x @ y @ x @ y @ x + (1/280) * y @ x @ y @ x @ y @ x @ y @ x)
    else:
        raise ValueError("order > 8 does not implement")




def bchHamiltonian(matT, matV, order):
    s = -1.j / hbar * T_kick
    x = matV
    y = matT
    out = sum([s ** (i - 1) * _bchHamitonianTerm(x, y, i)
               for i in range(1, order + 1)])
    return out

    '''
ハミルトニアンの量子BCH近似式をorderの項までの書き下し．
H = V + K + T/(i*hbar) * 1/2 * [V, K] + T^2/(i*habr)^2 * [V - K, [V , K]] + ...
'''


# 位置と運動量の定義


def get_qp(n=N):
    q = np.linspace(-BC, BC, n)
    nu = fftfreq(n, d=dQ)  # nuは周波数
    p = 2 * np.pi * nu * hbar  # 周波数から運動量に変換
    return q, p


# ポテンシャルの定義
#def V(q):
#    return .5 * q ** 2 - 2 * np.cos(q) - np.sqrt(np.pi) * scipy.special.erf(q) / 2

def V(q):
    return - (k/16) * np.exp(-8 * (np.array(q) ** 2)) - ((e2/(2 * (8 ** 0.5))) * np.sqrt(np.pi)) * (scipy.special.erf( (8 ** 0.5)* ( np.array(q) - xb)) - scipy.special.erf((8 ** 0.5) *( np.array(q) + xb)))

lamb =1.2
def V(q):
    return q ** 2 /2 - 2 * np.cos(q/lamb)
# 運動エネルギーの定義
def T(p):
    return 0.5 * p ** 2


# 規格化
@ nb.njit
def renormalize(psi):
    prob = np.real(np.conj(psi) * psi)  # 波動関数から確率へ変換
    renorm = np.sqrt((prob.sum() * dQ) + 1e-10)
    out = psi / renorm
    return out


# 初期コヒーレント状態
def init_state(q):
    d = q #- 2
    psi = np.power(np.pi, -1/4) * np.exp(-.5 * d ** 2 / hbar)
    psi = psi.astype(np.complex_)
    return renormalize(psi)


# TVオーダーの行列計算
def TVevolve(psi, dt, q, p):
    psi = np.exp(-V(q) * dt * 1.j / hbar) * psi
    psi = np.exp(-T(p) * dt * 1.j / hbar) * fft(psi)
    return ifft(psi)

# TVオーダーの行列計算
def TVevolvewithabsorb(psi, dt, q, p):
    psi = np.exp(-V(q) * dt * 1.j / hbar) * psi
    psi = np.exp(-T(p) * dt * 1.j / hbar) * fft(psi)
    psi = ifft(psi)
    psi = tanh_abs2(q, 1.2, 10) * psi
    print(psi)
    psi = tanh_abs(q, -1.2, 10) * psi
    print(psi)
    return psi


def tanh_abs(x,x_c,beta):
    xx = beta*((x-x_c))
    return (np.tanh(xx)+1)/2

def tanh_abs2(x,x_c,beta):
    xx = beta*(-(x-x_c))
    return (np.tanh(xx)+1)/2

def exp_abs(x, x1,x2,alpha,beta):
    x1 = (x - x1)
    x2 = (x - x2)
    theta0= (1 - np.tanh(x1*beta))/2
    theta1= (1 + np.tanh(x2*beta))/2
    w = (x1**2*theta0 +x2**2*theta1)
    p = np.exp(-w/alpha)
    return p

# VTオーダーの行列計算
def VTevolve(psi, dt, q, p):
    psi = jnp.exp(+T(p) * dt * 1.j / hbar) * fft(psi)
    psi = jnp.exp(+V(q) * dt * 1.j / hbar) * ifft(psi)
    return psi



# ポテンシャルの行列表現．対角に並べただけ．
def matrix_V():
    q, p = get_qp()
    return np.diag(V(q))


# 運動エネルギーの行列表現．


def matrix_T():
    q, p = get_qp()
    matT = np.zeros((N, N), dtype=np.complex_)
    for j in range(N):
        psi_j = np.zeros(N, dtype=np.complex_)
        psi_j[j] = 1.0
        pvec = fft(psi_j)
        pvec = T(p) * pvec  # WHY HANADA?
        matT[j] = ifft(pvec)
    return matT.T


# ユニタリ行列の対角化
def unitary_eig(matrix):
    R, V = scipy.linalg.schur(matrix, output="complex")
    return R.diagonal(), V.T


# BCH波動関数の計算
@ nb.njit
def bch_evolve(evals, evecs, phi0, n):
    phi = np.zeros(N, dtype=nb.complex128)
    for j in range(N):
        u_j = np.exp(-1.j * evals[j] / hbar * n)
        psi_j = evecs[j]
        phi += u_j * np.dot(np.conj(psi_j), phi0) * \
            psi_j  # u^n_j <psi_j|phi_0>|psi_j>
    return renormalize(phi)

def main3():
    n_steps = 0
    dt = T_kick
    q, p = get_qp()
    phi0 = init_state(q).astype(np.complex_)
    matT = matrix_T()
    matV = matrix_V()
    #H = bchHamiltonian(matT, matV, 3)
    H = matT + matV
    #U = np.exp(-i * H / hbar)bb
    bch_evals1, bch_evecs1 = np.linalg.eigh(H)
    bch_evecs1 = bch_evecs1.T
    h = np.arange(0,N,1)
    print(bch_evals1,h)
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_xlim(-BC, BC)
    ax.set_ylim(-BC, BC)
    ax.set_title(r'$N = {}, \tau = {}, BC = {}, grid = {}$'.format(
        n_steps, T_kick, QMAX - QMIN, N),fontsize = 20)
    phi0 = np.array(phi0)
    #bch_prob1 = psi * np.conj(psi)
    np.max(phi0)
    bch_evals1 = np.array(bch_evals1)
    bch_evecs1 = np.array(bch_evecs1)
    a = np.linspace(-BC,BC, 10 ** 3.5)
    b = np.linspace(-BC,BC, 10 ** 3.5)
    A,B = np.meshgrid(a,b)
    z = T(A) + V(B)
    #bch_psi1 = bch_evolve(bch_evals1, bch_evecs1, bch_evecs1[0], n_steps * T_kick)
    for i in range(0,500):
        psi = bch_evecs1[i]
        density  = psi * np.conj(psi)
        print( np.abs(q[np.argmax(density)]),h[i],bch_evals1[i])
        #if i %5 == 0:
        #    cntr = ax.contour(A,B,z,levels = [bch_evals1[i]],color = "blue")
        #    ax.clabel(cntr)
        if i == 125 or i == 251 or i == 375:
            cntr = ax.contour(A,B,z,levels = [bch_evals1[i]],color = "red")
            ax.clabel(cntr)
    ax.set_xlabel(r"$q$",fontsize = 25)
    ax.set_ylabel(r"$p$",fontsize = 25)
    plt.tick_params(labelsize = 20)
    phase = np.loadtxt("phase_shearless_12.txt")
    plt.plot(phase[0],phase[1],",k")
    #ax.set_ylim(0.000,0.013)
    plt.show()
    #print(bch_evecs1.shape)
    #for i in range(n_steps):#時間発展
    #    psi = TVevolvewithabsorb(psi, dt, q, p)
        #qp = map(q,p)
    #X,Y = np.meshgrid(q,p)
    #density = bch_evecs1[1] * np.conj(bch_evecs1[1] ) 
    #density = np.array(density,dtype = float)
    #plt.pcolormesh(X, Y, density ,cmap = 'hsv')
    #plt.show()

    #exit()
    ###############################
    
    ##############################以下では波動関数を求める。
    bch_prob1 = psi * np.conj(psi)
    #bch_prob1 = bch_psi1 * np.conj( bch_psi1 )
    #plt.plot( q, bch_prob1 )
    #plt.xlim(-0.5 ,0.5)
    #plt.semilogy()
    #plt.show()
    #plt.savefig(
        #f"Cla_bch_{n_steps}_tau{T_kick}_Q{QMAX - QMIN}_grid{N}.png", dpi=300)
    # plt.close()
if __name__ == '__main__':
    main3()