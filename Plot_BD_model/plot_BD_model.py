import numpy as np
import matplotlib.pyplot as plt
from astropy import constants as c
from astropy import units as u
from scipy.interpolate import interp1d
import glob

plt.rcParams['figure.figsize'] = [16, 12]
ages = [0.1, 0.5, 1, 5, 10]

fl = glob.glob('phillips2020/*.txt')
first_time = True
for f in fl:
    if first_time:
        data = np.loadtxt(f)
        first_time = False
    else:
        data = np.vstack((data, np.loadtxt(f)))

for age in ages:
    data_age = data[:, 1][np.argmin(np.abs(data[:, 1]-age))]
    arg = (data[:, 1] == data_age)
    ms = data[:, 0][arg]
    rs = data[:, 4][arg]
    sort_arg = np.argsort(ms)
    ms = ms[sort_arg]
    rs = rs[sort_arg]

    iso_f = f'{age}Gyr.txt'
    iso = np.loadtxt(iso_f, comments = '!')
    iso = iso[np.argsort(iso[:, 0])]
    iso_arg = iso[:, 0] > ms[-1]
    ms = np.hstack((ms, iso[:, 0][iso_arg]))
    rs = np.hstack((rs, iso[:, 4][iso_arg]))

    f = interp1d(ms*u.Msun.to('Mjupiter'), rs*u.Rsun.to('Rjupiter'), kind='cubic')
    x = np.linspace(10, 140, 1000)
    plt.plot(x, f(x), label=f'{age} Gyr', lw=3)

bd_f = 'previous_BD.txt'
bds = np.loadtxt(bd_f, dtype=str)
ms = np.array(bds[:, 1],dtype=float)
ms_err = np.transpose(np.array(bds[:, [3,2]],dtype=float))
rs = np.array(bds[:, 4],dtype=float)
rs_err = np.transpose(np.array(bds[:, [6,5]],dtype=float))
plt.errorbar(ms, rs, xerr=ms_err, yerr=rs_err, fmt='.', c='black', alpha=0.5, ms=20)

bdthis = ['TOI-1608b', 'TOI-2336b', 'TOI-2521b']
mthis = [90.5, 69.9, 77.5]
mthis_err = [3.7, 2.3, 3.2]
rthis = [1.22, 1.05, 1.02]
rthis_err = [[0.06, 0.04, 0.04],[0.06, 0.04, 0.04]]
plt.errorbar(mthis, rthis, xerr=mthis_err, yerr=rthis_err, fmt='.', ms=30, c='red')
plt.annotate(bdthis[0], xy=(mthis[0], rthis[0]), xytext=(mthis[0]+2, rthis[0]+0.05), fontsize=40, c='red')
plt.annotate(bdthis[1], xy=(mthis[1], rthis[1]), xytext=(mthis[1]-20, rthis[1]+0.05), fontsize=40, c='red')
plt.annotate(bdthis[2], xy=(mthis[2], rthis[2]), xytext=(mthis[2]+2, rthis[2]-0.1), fontsize=40, c='red')


plt.xlim(20,140)
plt.ylim(0.5, 2.0)
plt.tick_params(labelsize=25, direction='in', axis='both', width=2, length=12)
plt.xlabel('Mass ($M_J$)', fontsize=35)
plt.ylabel('Radius ($R_J$)', fontsize=35)
plt.legend(fontsize=25)

plt.savefig('BD_iso.png')
