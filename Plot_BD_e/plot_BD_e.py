import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from astropy import constants as c
from astropy import units as u
import matplotlib as mpl
mpl.rcParams['xtick.direction'] = 'in'

in_f = 'previous_BD.txt'
data = np.loadtxt(in_f, dtype=str)
data = np.array(data[:, 1:], dtype=float)
bds = {}
bds['m2'] = data[:, 0]
bds['m2_err'] = data[:, [2,1]]
bds['r2'] = data[:, 3]
bds['r2_err'] = data[:, [5,4]]
bds['P'] = data[:, 6]
bds['e'] = data[:, 7]
bds['e_err'] = data[:, [9,8]]

bds['e_err'][bds['e_err'] == -999] = 0

mthis = [90.5, 69.9, 77.5]
mthis_err = [3.7, 2.3, 3.2]
rthis = [1.22, 1.05, 1.02]
rthis_err = [[0.06, 0.04, 0.04],[0.06, 0.04, 0.04]]
ethis = [0.041, 0.01, 0.]
ethis_err = [[0.019, 0.005, 0.],[0.024, 0.006, 0.035]]
bdthis = ['TOI-1608b', 'TOI-2336b', 'TOI-2521b']
Pthis = np.array([2.47274, 7.711977, 5.563062])

plt.rcParams['figure.figsize'] = [16, 12]
plt.errorbar(bds['m2'], bds['e'], xerr=bds['m2_err'].T, yerr=bds['e_err'].T, fmt='.', c='black', ms=1)
plt.errorbar(mthis, ethis, xerr=mthis_err, yerr=ethis_err, fmt='.', ms=1, c='red', zorder=5)

norm = colors.LogNorm(np.min(bds['P']), np.max(bds['P']))
plt.scatter(bds['m2'], bds['e'], s=200*bds['r2'], zorder=10, c=bds['P'], norm=norm)
cb = plt.colorbar()
plt.scatter(mthis, ethis, s=200*np.array(rthis), zorder=15, c=Pthis, linewidth=3, edgecolors='red', norm=norm)

# Note CWW-89Ab
plt.scatter([39.2], [0.189], s=800, facecolor='None', edgecolors='orange', linewidth=4)
plt.annotate('CWW 89Ab', xy=(10, 0.25), xytext=(10, 0.23), fontsize=30, c='orange', zorder=10)

# norm = colors.PowerNorm(gamma=0.5)
# plt.scatter(bds['m2'], bds['e'], c=bds['r2'], zorder=10, norm=norm)


# plt.annotate(bdthis[0], xy=(mthis[0], ethis[0]), xytext=(mthis[0]+2, ethis[0]+0.02), fontsize=30, c='red', zorder=10)
# plt.annotate(bdthis[1], xy=(mthis[1], ethis[1]), xytext=(mthis[1]-25, ethis[1]+0.02), fontsize=30, c='red', zorder=10)
# plt.annotate(bdthis[2], xy=(mthis[2], ethis[2]), xytext=(mthis[2], ethis[2]), fontsize=30, c='red', zorder=10)

plt.axvline(42.5, ls='dashed', c='black', lw=3)

cb.ax.tick_params(labelsize=25, direction='in', width=2, length=12)
cb.ax.tick_params(labelsize=25, direction='in', axis='both', which='minor', width=2, length=6)
cb.set_label('Period (days)', fontsize=35)
plt.ylim(-0.01, 0.8)
plt.tick_params(labelsize=25, direction='in', axis='both', width=2, length=12)
plt.xlabel('Mass ($M_J$)', fontsize=35)
plt.ylabel('Eccentricity', fontsize=35)

plt.savefig('BD_m_e.png')
