import numpy as np
from iminuit import Minuit
from astropy.io import fits

def sub_data(xlim, array):
    indeces = np.array([], dtype=int)
    for i in range(len(array)):
        if array[i, 0] < xlim[0]:
            indeces = np.append(i, indeces)
        if array[i, 0] > xlim[1]:
            indeces = np.append(i, indeces)
    array = np.delete(array, indeces, axis=0)
    return array

def normalize_with_range(data, range):
    # have changed this from
    #     data[:, 1] = data[:, 1] / scaling
    #     return data
    # as it normalizes the input and not just the output
    # might cause problems
    data_used_to_normalize = sub_data(range, data)
    scaling = np.trapz(data_used_to_normalize[:, 1], x=data_used_to_normalize[:, 0])
    return data[:, 1] / scaling

def normalize_with_range2(data, range):
    data_used_to_normalize = sub_data(range, data)
    scaling = np.trapz(data_used_to_normalize[:, 1], x=data_used_to_normalize[:, 0])
    data[:, 1] = data[:, 1] / scaling
    if data.shape[1] > 2:
        data[:, 2] = data[:, 2] / scaling
    return data

def genfromtxtandfits(fileName):
    if fileName.find('.dat') >= 0:
        data = np.genfromtxt(fileName)
    if fileName.find(".fits") >= 0:
        hdul = fits.open(fileName)
        data = np.transpose(hdul[0].data)
    return data


def gaussian_with_background(x, a, mu, std, slope, offset):
    return a * np.exp(-(x-mu)**2 / (2*std**2)) + slope * x + offset

def gauss_function(x, a, x0, sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))

def avg_of_values_with_gauss_err(vals, std_deviations):
    mu0 = np.mean(vals)
    variance = std_deviations**2
    def chi2(mu):
        return sum((vals-mu)**2/variance)
    m = Minuit(chi2, mu=mu0, errordef=1)
    m.migrad()
    return [m.np_values()[0], m.np_errors()[0]]

def digitized_data_converter(digitized_data):
    ret = np.zeros((len(digitized_data), 4))
    for i in range(int(len(digitized_data) / 2)):
        # filling x-coordinates
        ret[i, 0] = digitized_data[i*2, 0]
        # filling y-coordinates
        ret[i, 1] = digitized_data[i*2, 1]
        # filling x-err
        ret[i, 2] = abs(digitized_data[i*2, 0] - digitized_data[i*2 + 1, 0])
        # filling y-err
        ret[i, 3] = abs(digitized_data[i*2, 1] - digitized_data[1 + i*2, 1])
    return ret



