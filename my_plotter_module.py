import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import math
from scipy.signal import argrelextrema
from general_scripts import my_astro_formulas as maf
from general_scripts import my_data_handling as mdh
from scipy.signal import savgol_filter

# Use to plot spectroscopy with an offset (spacing) inbetween each spectrum at each epoch
# an array with strings of the epoch can be passed to the function
# a smooth overlay made with binned data points can be shown on the plot if smooth_overlay=True
# DATA MUST BE AN ARRAY WITH A TABLE IN EACH "data[i]" and DATA MUST BE PREPROCESSED so that it does not
# contain negative flux values (also not 0) and NaN values!
def multi_spec_plot(data, fig_xlim, spacing=1, epochs=None, z=0, smooth_overlay=True, plot_raw=False):
    """ Have changed it from plt.figure(), plt.savefig() and plt.show() being part of the script
     to not including these to able to change figure settings and so on just like plt.plot()
     So dont include nameofImage as it has been removed!!!"""

    # plotting flux (log scale) vs wavelength
    if z != 0:
        for i in range(len(data)):
            data[i][:, 0] = maf.deredshift(data[i][:, 0]*10000, z)
    for i in range(len(data)):
        print("iteration: ", i)
        log10_data = np.log10(data[i][:, 1])
        logsum = sum(log10_data)
        mean_val = logsum / len(data[i])
        print("     mean val: ", mean_val)
        offset = spacing * (len(data) - i - 1) - mean_val
        print("     offset: ", offset)
        print(max(data[i][:, 0]))
        clr = 'darkgray'
        if plot_raw == True:
            plt.plot(data[i][:, 0], log10_data + offset, clr, zorder=2)


        # plotting epochs at the right spot if epochs != None
        if epochs != None:
            assert (len(epochs) == len(data))
            max_wl = np.amax(data[i][:, 0])
            first_index_of_tail = round(len(data[i][:, 1]) / 1.2)
            mean_val_tail = sum(log10_data[first_index_of_tail:]) / (len(data[i]) - first_index_of_tail)
            clr2 = 'tomato'
            if i % 2 == 0: clr2 = 'sandybrown'
            plt.text(max_wl + 1/80*max_wl, spacing * (len(data) - i - 1) - (mean_val - mean_val_tail), epochs[i], color=clr2, zorder=10)

        # smoothening of the data with bins of width "bin_width" if smooth_overlay is True
        if smooth_overlay:
            bin_width = 20
            first_wl = data[i][0, 0]
            last_wl = data[i][-1, 0]
            n_bins = math.ceil((last_wl - first_wl)/bin_width)
            print('number of bins', n_bins, 'first wavelength', first_wl, 'last wavelength', last_wl)
            bins = []
            for n in range(n_bins):
                bins.append(first_wl+1/2*bin_width+n*bin_width)

            bin_flux = [log10_data[0]]
            b_index = 0
            n_pr_bin = 1
            for j in range(len(data[i])):
                if data[i][j, 0] < (bins[b_index]-1/2*bin_width + bin_width):
                    bin_flux[b_index] = bin_flux[b_index] + log10_data[j]
                    n_pr_bin += 1
                else:
                    bin_flux[-1] = bin_flux[-1]/n_pr_bin
                    bin_flux.append(log10_data[j])
                    b_index += 1
                    n_pr_bin = 1
            # little workaround; if time find out why you need to remove the last element!!!
            bins = np.delete(bins, -1)
            bin_flux = np.delete(bin_flux, -1)
            while len(bins) > len(bin_flux):
                bins = np.delete(bins, -1)
            clr3 = 'tomato'
            if i % 2 == 0: clr3 = 'sandybrown'
            plt.plot(bins, bin_flux + offset, clr3, zorder=3)



    # other figure specifications
    plt.xlabel("Restframe wavelength (Å)", fontsize=16)
    plt.ylabel("$log_{10}(F_{\lambda}$) + offset", fontsize=16)
    axes = plt.gca()
    axes.tick_params(axis='both', labelsize=14)
    axes.set_xlim(fig_xlim)
    axes.set_ylim([-1*spacing, spacing * (len(data)+1)-1])
    # plt.yticks([])


def print_array(array):
    for i in range(len(array)):
        print(array[i])


def print_2d_array(array):
    dim = array.shape
    for i in range(dim[0]):
        print()
        for j in range(dim[1]):
            print(array[i, j], '    ',end='')
        if(array[i, 1] == 0):
            print()
            print()
    print()


def set_data_range(array, xrange):
    for i in range(len(array)):
        indexes = []
        for j in range(len(array[i])):
            if array[i][j, 0] < xrange[0]:
                indexes.append(j)
            if array[i][j, 0] > xrange[1]:
                indexes.append(j)
        array[i] = np.delete(array[i], indexes, axis=0)

def pre_processing(array):

    # what to do about unordered data points?

    for i in range(len(array)):
        #print("Spectrum: ", i + 1)
        #print("     length of data before: ", len(array[i]))
        indexes_nan = []
        for j in range(len(array[i])):
            if math.isnan(array[i][j, 1]):
                indexes_nan.append(j)
        array[i] = np.delete(array[i], indexes_nan, axis=0)

        indexes_neg = []
        for j in range(len(array[i])):
            if array[i][j, 1] <= 0:
                indexes_neg.append(j)
        array[i] = np.delete(array[i], indexes_neg, axis=0)

        indexes_noisy = []
        indexes_after_noisy = []
        tol = 1.6
        log10_array = np.log10(array[i][:, 1])
        for j in range(round(len(array[i])/3)):
            mean_15 = sum(log10_array[-j-1-15:-j-1])/15
            if abs(log10_array[-j-1]-mean_15) > tol:
                indexes_noisy.append(len(array[i])-j-1)
        if len(indexes_noisy) > 0:
            min_index = np.amin(indexes_noisy)
            number_of_noisy_points_in_tail = len(array[i])-min_index
            for k in range(number_of_noisy_points_in_tail):
                indexes_after_noisy.append(min_index+k)
        array[i] = np.delete(array[i], indexes_after_noisy, axis=0)


        #print("     noisy data poits filtered out: ", len(indexes_noisy))
        #print("     neg data points filtered out: ", len(indexes_neg))
        #print("     nan data points filtered out: ", len(indexes_nan))
        #print("     length of data after: ", len(array[i]))

    return array


def fill_EW(data):

    def find_points(extrema_start, extrema_end):
        y_diff = np.longdouble(data[extrema_start, 1]-data[extrema_end, 1])
        x_distance = data[extrema_end, 0]-data[extrema_start, 0]
        ys = np.zeros(extrema_end-extrema_start)
        ys[0] = data[extrema_start, 1]
        for i in range(extrema_start, extrema_end-1):
            fraction = (data[i+1, 0]-data[i, 0])/x_distance
            ys[i + 1 - extrema_start] = ys[i + 1 - extrema_start - 1] - fraction * y_diff
        xs = data[extrema_start:extrema_end, 0]
        return ys, xs

    tuple_ex = argrelextrema(data[:, 1], np.greater, order=40)
    extrema_indices = tuple_ex[0]
    print(tuple_ex)

    ys_Fe, xs_Fe = find_points(388, extrema_indices[3])
    ys_S, xs_S = find_points(extrema_indices[4], extrema_indices[6])
    ys_Si2, xs_Si2 = find_points(extrema_indices[7], extrema_indices[8])
    ys_Si, xs_Si = find_points(extrema_indices[8], extrema_indices[9])
    ys_Ca, xs_Ca = find_points(extrema_indices[13], extrema_indices[15])

    plt.plot(data[:, 0], data[:, 1])
    plt.fill_between(xs_Fe, data[388:extrema_indices[3], 1], ys_Fe, alpha=0.2, color=(0, 0, 0.5), label='Fe II: $\lambda$5083.4')
    plt.fill_between(xs_S, data[extrema_indices[4]:extrema_indices[6], 1], ys_S, color=(0.1, 0.9, 0.3), alpha=0.2, label='S II: $\lambda$5449.4 and $\lambda$5623.1')
    plt.fill_between(xs_Si2, data[extrema_indices[7]:extrema_indices[8], 1], ys_Si2, color=(0.8, 0.9, 0.0), alpha=0.2, label='Si II: $\lambda$5971.8')
    plt.fill_between(xs_Si, data[extrema_indices[8]:extrema_indices[9], 1], ys_Si, color=(0.9, 0.6, 0.0), alpha=0.2, label='Si II: $\lambda$6355.2')
    plt.fill_between(xs_Ca, data[extrema_indices[13]:extrema_indices[15], 1], ys_Ca, color=(1, 0, 0), alpha=0.2, label='Ca IRT: $\lambda$8578.5')
    plt.xlabel('Rest wavelength (Å)', fontsize=18)
    plt.ylabel('F$_\lambda$ (erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$)', fontsize=18)
    plt.legend(fontsize=17)

def nan_counts(array):
    counts = 0
    for i in range(len(array)):
        if math.isnan(array[i]):
            counts += 1
    return counts


def inf_counts(array):
    counts = 0
    for i in range(len(array)):
        if math.isinf(abs(array[i])):
            counts += 1
    return counts


def neg_counts(array):
    counts = 0
    for i in range(len(array)):
        if array[i] <= 0:
            counts += 1
    return counts

def plot_box(xy_min, xy_max, **kwargs):
    plt.fill_between([xy_min[0], xy_max[0]], xy_min[1], xy_max[1], **kwargs)

def plot_on_top(object_array, epoch, x_range, epoch_offset=1, small_offset=0.1, y_offset_text=1, plotLog=False):

    colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5']
    # First we need to iterate over the objects in the array
    # (index 0 is could be changed with 1 as both sub-arrays need the same length)
    for i in range(len(object_array[0])):
        # Next we iterate over number of supernova objects
        for j in range(len(object_array)):
            object_array[j][i] = mdh.sub_data(x_range, object_array[j][i])
            object_array[j][i][:, 1] = savgol_filter(object_array[j][i][:, 1], 15, 1)
            if plotLog==True:
                object_array[j][i][:, 1] = np.log(object_array[j][i][:, 1])
            plt.plot(object_array[j][i][:, 0], object_array[j][i][:, 1] + epoch_offset * (len(object_array[0]) - i) + small_offset*(len(object_array) - j), colors[j])
            plt.text(object_array[j][i][-1, 0] + y_offset_text, object_array[j][i][-1, 1] + epoch_offset * (len(object_array[0]) - i) + small_offset*(len(object_array) - j), epoch[j][i], color=colors[j], fontsize=12)
