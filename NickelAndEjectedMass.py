import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import statistics as stat


def sub_data(xlim, array):
    indeces = np.array([], dtype=int)
    for i in range(len(array)):
        if array[i, 0] < xlim[0]:
            indeces = np.append(i, indeces)
        if array[i, 0] > xlim[1]:
            indeces = np.append(i, indeces)
    array = np.delete(array, indeces, axis=0)
    return array


def Ni_mass(lum, t_R, alpha):
    def epsilon(t_R):
        m_u = 1.66054 * 1e-27  # transformed to kg
        tau_Co = 111.3 * 24 * 60 * 60  # in seconds
        tau_Ni = 8.80 * 24 * 60 * 60  # in seconds
        Q_Co = 3.73 * 1.602 * 10 ** (-6)  # in ergs
        Q_Ni = 1.75 * 1.602 * 10 ** (-6)  # in ergs
        epsilon1 = 1 / (56 * m_u) / (tau_Co - tau_Ni) * (Q_Ni * (tau_Co / tau_Ni - 1) - Q_Co) * np.exp(-t_R / tau_Ni)
        epsilon2 = 1 / (56 * m_u) / (tau_Co - tau_Ni) * Q_Co * np.exp(-t_R / tau_Co)
        return epsilon1 + epsilon2
    solar_mass = 1.9891 * 1e30
    lum_from_1solarmass = alpha * epsilon(t_R) * solar_mass
    return lum/lum_from_1solarmass



def solar_mass_to_atoms(solar_masses, Z):
    sun_mass = 1.988435 * 10 ** 30  # kg
    amu = 1.6605391 * 10 ** (-27)  # kg
    return solar_masses * sun_mass / (Z * amu)


def ejected_mass(CS_opacity, q, v_efolding, t_0):
    sun_mass = 1.988435*10**30 # kg
    ejected_mass = 8*np.pi / (CS_opacity * q) * v_efolding**2 * t_0**2
    return ejected_mass/sun_mass

# set parameters
t_rise = 19.1 # time from first light to bol max
x_lim = [60, 120] # late phase range that must be fitted (days relative to bol max)
CS_opacity = 0.025 # compton scattering opacity
q = 1/3 # distribution of Fe-peak elements
v_ef = 3000 # e-folding velocity

lum_data = np.load('bol_data/SED_MUV_UVOIR_IR.npy')

# finding nickel mass using Arnetts rule
max_lum = max(lum_data[:, 1])
max_lum_err = lum_data[np.argmax(lum_data[:, 1]), 2]

nickel_masses = np.zeros(500)
for i in range(500):
    nickel_masses[i] = Ni_mass(max_lum + np.random.normal(0, max_lum_err), (19.10 + np.random.normal(0, 0.13))*24*60*60, 1.2 + np.random.normal(0, 0.2))

nickel_mass = np.mean(nickel_masses)
nickel_mass_std = np.std(nickel_masses)

def energy_deposition2(t, t0):
    Ni_mass = solar_mass_to_atoms(nickel_mass, 56)
    lt_Ni = 8.8
    lt_Co = 111.3
    Q_Co_e = 0.12 * 1.602 * 10 ** (-6) # in erg
    Q_Co_gamma = 3.61 * 1.602 * 10 ** (-6)
    Q_Ni = 1.75 * 1.602 * 10 ** (-6)  # in ergs
    E_dep_1st_term = Ni_mass * (Q_Ni / lt_Ni) * np.exp(-t / lt_Ni)
    E_dep_2nd_term = Ni_mass * (Q_Co_e + Q_Co_gamma * (1 - np.exp(-t0**2/t**2))) / (lt_Co - lt_Ni) * (np.exp(-t / lt_Co) - np.exp(-t / lt_Ni))
    E_dep = E_dep_1st_term + E_dep_2nd_term
    return E_dep / (24*60*60) # converted to erg pr second

# finding t0 and ejected mass using Jeffery's idea
fid_times = np.zeros(500)
Ni_masses = np.zeros(500)
late_lum_data = sub_data(x_lim, lum_data)

for i in range(500):
    popt, pcov = curve_fit(energy_deposition2, late_lum_data[:, 0] + t_rise, late_lum_data[:, 1] + np.random.normal(0, late_lum_data[:, 2]), p0=[10], bounds=([1], [50]))
    fid_times[i] = popt[0]

# finding ejected mass and error via MC simulation
fid_time = np.mean(fid_times)
fid_time_err = np.std(fid_times)
M_ej = np.array([ejected_mass(CS_opacity * 1000 / 100000 ** 2, q, v_ef, fid_time * 24 * 60 * 60),
                 (ejected_mass(CS_opacity * 1000 / 100000 ** 2, q, v_ef, (fid_time + fid_time_err) * 24 * 60 * 60)- ejected_mass(CS_opacity * 1000 / 100000 ** 2, q, v_ef, (fid_time - fid_time_err) * 24 * 60 * 60))/2])


# printing values
print('Max magnitude: ', max_lum)
print('Ni mass from Arnetts rule (in solar masses): {} \u00B1 {} '.format(nickel_mass, nickel_mass_std))
print('Fiducial time from fit: {} \u00B1 {}'.format(fid_time, fid_time_err))
print('Total ejected mass: {} \u00B1 {}'.format(M_ej[0], M_ej[1]))
print('Nickel mass from Dhawan relation: {}'.format(Ni_mass(1.3*10**43, 19.10*24*60*60, 1.2)))


# plotting data and fit
fig = plt.figure(figsize=(10, 6))
plt.plot(lum_data[:, 0], np.log10(lum_data[:, 1]))
plt.fill_between(lum_data[:, 0], np.log10(lum_data[:, 1] - lum_data[:, 2]), np.log10(lum_data[:, 1] + lum_data[:, 2]), alpha=0.2)
plt.plot(lum_data[:, 0], np.log10(energy_deposition2(lum_data[:, 0]+t_rise, fid_time)), color='k', linestyle='dashed', label='Deposition function')
plt.xlim([-16, 100])
plt.ylim([41.5, 43.5])
plt.xlabel('Restframe days since $t_{bol, max}$', fontsize=17)
plt.ylabel('log$_{10}$(Luminosity (erg s$^{-1}$))', fontsize=17)
plt.legend(['SED: full range', 'SED: UVOIR'], fontsize=14)
plt.vlines([0], 41.5, 43.2, linestyles='dashed')
ax = plt.gca()
ax.tick_params(axis='x', labelsize=17)
ax.tick_params(axis='y', labelsize=17)
ax2 = fig.add_axes([0.6, 0.4, 0.29, 0.345])
ax2.plot(lum_data[:, 0], np.log(lum_data[:, 1]))
ax2.fill_between(lum_data[:, 0], np.log(lum_data[:, 1] - lum_data[:, 2]), np.log(lum_data[:, 1] + lum_data[:, 2]), alpha=0.2)
#ax2.set_yticks([0, 0.5e43, 1e43, 1.5e43])
ax2.set_xticks([0, 150, 300])
plt.savefig('bolometric_LC_SED.pdf', bbox_inches='tight')

plt.figure()
plt.plot(lum_data[:, 0], np.log10(energy_deposition2(lum_data[:, 0], fid_time)), color='k', linestyle='dashed', label='Deposition function')

plt.show()