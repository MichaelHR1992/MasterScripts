import numpy as np
from snpy.utils.deredden import *
import matplotlib.pyplot as plt
from general_scripts import my_astro_formulas as maf
from general_scripts import my_data_handling as mdh
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter


def extinction_law(xs, E_BV, offset):
    a, b = ccm(xs)
    R_V = 3.1
    A_lam = E_BV * R_V * (a + b / R_V)
    return A_lam + offset

def extinction_law_no_Rv(xs, E_BV, R_V, offset):
    a, b = ccm(xs)
    A_lam = E_BV * R_V * (a + b / R_V)
    return A_lam + offset


# setting key parameters
n_data = 10; range_of_interest = [3700, 7600]; t_Bmax = 56648.64; z_sn13gy = 0.01402; z_sn11fe = 0.000804

'''loading spectra of and ordering epochs: 
only color matched spectra with RMS above 0.2 are used.'''
sn13gy_filenames = ['spectra_colmatch/56633_5.dat', 'spectra_colmatch/56634_5.dat', 'spectra/56640_0.dat', 'spectra_colmatch/56646_5.dat', 'spectra_colmatch/56647_7.dat', 'spectra_colmatch/56651_8.dat', 'spectra/56657_2.dat', 'spectra_colmatch/56665_3.dat', 'spectra/56671_0.dat', 'spectra/56697_9.dat']
MJDs = np.array([56633.5, 56634.5, 56640.0, 56646.5, 56647.7, 56651.8, 56657.3, 56665.3, 56671.0, 56697.9])
epochs_sn13gy_float = ((MJDs - t_Bmax)/(1 + z_sn13gy))
epochs_sn13gy = [str(i.round(decimals=1)) for i in epochs_sn13gy_float]
print(epochs_sn13gy)
# loading spectra of 11fe and ordering epochs
sn11fe_filenames = ['sn2011fe/colmatch/M01600.dat', 'sn2011fe/colmatch/M01375.dat', 'sn2011fe/colmatch/M00875.dat', 'sn2011fe/colmatch/M00263.dat', 'sn2011fe/colmatch/M00076.dat', 'sn2011fe/colmatch/P00300.dat', 'sn2011fe/colmatch/P00922.dat', 'sn2011fe/colmatch/P01722.dat', 'sn2011fe/colmatch/P02221.dat', 'sn2011fe/colmatch/P04118.dat']
epochs_sn11fe = ['-15.3', '-13.8', '-8.8', '-2.6', '-0.8', '3.0', '9.2', '17.2', '22.2', '41.2']
sn13gy_data = np.empty(len(sn13gy_filenames), dtype=object)
sn11fe_data = np.empty(len(sn11fe_filenames), dtype=object)
for i in range(n_data):
    print(i)
    sn13gy_data[i] = np.loadtxt(sn13gy_filenames[i])
    sn11fe_data[i] = np.loadtxt(sn11fe_filenames[i])

# initializing arrays
ratios = np.zeros((100, 1000))
ratios_means = np.zeros((n_data, 1000))
ratios_errs = np.zeros((n_data, 1000))
A_from_ratios = np.zeros((n_data, 1000))
A_err_from_ratios = np.zeros((n_data, 1000))
xs = np.linspace(range_of_interest[0] + 200, range_of_interest[1] - 200, 1000)
sn13gy_interp = np.zeros(1000)
sn11fe_interp = np.zeros(1000)

'''Below a MC rutine is used to fnd errors'''
EBV_hosts = np.zeros(n_data); var = np.zeros(n_data)
offset = np.zeros(n_data)
plt.figure(figsize=(7, 9))
for i in range(n_data):
    sn13gy = sn13gy_data[i]
    sn11fe = sn11fe_data[i]
    # unreddening due to MW extinction
    sn13gy[:, 1], a, b = unred(sn13gy[:, 0], sn13gy[:, 1], 0.048)
    sn11fe[:, 1], a, b = unred(sn11fe[:, 0], sn11fe[:, 1], 0.0077)
    # deredshifting
    sn13gy[:, 0] = maf.deredshift(sn13gy[:, 0], z_sn13gy)
    sn11fe[:, 0] = maf.deredshift(sn11fe[:, 0], z_sn11fe)
    # normalizing
    sn13gy = mdh.normalize_with_range2(sn13gy, [3800, 7000])
    sn11fe = mdh.normalize_with_range2(sn11fe, [3800, 7000])
    # smoothening
    sn13gy[:, 1] = savgol_filter(sn13gy[:, 1], 11, polyorder=1)
    sn11fe[:, 1] = savgol_filter(sn11fe[:, 1], 11, polyorder=1)
    # cutting data in x-dimension
    sn13gy = mdh.sub_data(range_of_interest, sn13gy)
    sn11fe = mdh.sub_data(range_of_interest, sn11fe)
    # doing montecarlo for errorestimation
    std = np.zeros(1000)
    for j in range(100):
        # make iterpolation for finding ratio
        sn13gy_interp = interp1d(sn13gy[:, 0]/(1 + np.random.normal(0, 0.001402)), sn13gy[:, 1]) # + np.random.normal(0, sn13gy_noise))
        # some 11fe data do not have error other has measured variance
        sn11fe_interp = interp1d(sn11fe[:, 0], sn11fe[:, 1])
        # finding ratio (note to represent extinction we must have that A_ratio below is positive. Fot it to be positive the denominator must be negative.
        ratios[j, :] = sn13gy_interp(xs)/sn11fe_interp(xs)

    # finding mean flux radius
    ratios_means[i, :] = np.mean(ratios, axis=0)
    ratios_errs[i, :] = np.std(ratios, axis=0)
    A_from_ratios[i, :] = -2.5*np.log(ratios_means[i, :])
    A_err_from_ratios[i, :] = -2.5 * np.log(ratios_errs[i, :])
    # finding best fit value for EBV_host
    popt, pconv = curve_fit(extinction_law_no_Rv, xs, A_from_ratios[i, :], sigma=A_err_from_ratios[i, :], bounds=([-1, 3.09, -np.inf], [2, 3.1, np.inf]))
    EBV_hosts[i] = popt[0]
    # plotting data:
    plt.plot(xs, A_from_ratios[i, :] + (n_data-i)*1.2, color='C0')
    plt.plot(xs, extinction_law_no_Rv(xs, popt[0], popt[1], popt[2] + (n_data-i)*1.2), color='k', linestyle='dashed')
    plt.text(range_of_interest[1] - 180, extinction_law_no_Rv(max(xs), popt[0], popt[1], popt[2]) + (n_data - i) * 1.2 - 0.2, epochs_sn11fe[i], color='C0')
    plt.text(range_of_interest[1] - 180, extinction_law_no_Rv(max(xs), popt[0], popt[1], popt[2]) + (n_data - i) * 1.2 + 0.08, epochs_sn13gy[i], color='C1')
    plt.text(range_of_interest[1] - 700, extinction_law_no_Rv(max(xs), popt[0], popt[1], popt[2]) + (n_data - i) * 1.2 - 0.5, '{}'.format(round(popt[0], 2)))

# changing things about figure
plt.xlabel('Restwavelength ($\AA$)', fontsize=14)
plt.ylabel('-2.5log$_{10}$($F_{13gy}$/$F_{11fe}$) + offset (mag)', fontsize=14)
plt.xlim([range_of_interest[0] + 150, range_of_interest [1] + 150])
plt.ylim([0, 14.4])
plt.legend(['A$_\lambda$ from ratio', 'Best fit CCM+O law with R$_V$ = 3.1'])
plt.savefig('SN2011fe_extinction_law.pdf', bbox_inches='tight')

EBV_host = np.mean(EBV_hosts); EBV_host_err = np.std(EBV_hosts)
print('The best fit value for EBV_host is: {} \u00B1 {}'.format(EBV_host, EBV_host_err))
print('All values found are: ', [EBV_hosts])

plt.show()

