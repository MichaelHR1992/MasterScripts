import numpy as np

def deredshift(x, z):
    return x/(1+z)

def wl_to_velocity(wl, rest_wl, n):
    c = 300000
    v = (wl-rest_wl[n])/rest_wl[n]*c
    return v

def rest_frame_epoch(epochs, z):
    LF = np.sqrt(1-z)
    return epochs*LF