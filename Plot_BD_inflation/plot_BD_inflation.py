import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from astropy import constants as c
from astropy import units as u
import emcee
import corner
import matplotlib as mpl
mpl.rcParams['xtick.direction'] = 'in'

bdthis = ['TOI-1608b', 'TOI-2336b', 'TOI-2521b']
Fthis = [3124, 863, 787]
Fthis_err = [[282, 89, 153],[271, 86, 124]]
mthis = [90.5, 69.9, 77.5]
mthis_err = [3.7, 2.3, 3.2]
rthis = [1.22, 1.05, 1.02]
rthis_err = [[0.06, 0.04, 0.04],[0.06, 0.04, 0.04]]
Fearth = (1*u.Lsun/((1*u.au)**2 * np.pi * 4))

data = np.loadtxt('PS_2022.05.29_05.24.22.csv', delimiter=',', dtype=str)
data = data[data[:, -4] != '']
data = data[data[:, -3] != '']
data = data[data[:, -2] != '']
data = data[data[:, 8] != '']
rps = np.array(data[:, 8], dtype=float)
data = data[(rps < 2) & (rps > 0.6)]
rps, rperr1, rperr2 = np.array(data[:, 8], dtype=float), np.array(data[:, 9], dtype=float), np.array(data[:, 10], dtype=float)
data = data[(rperr1 < 0.5*rps) & (-rperr2 < 0.5*rps)]
Fs, Ferr1, Ferr2 = np.array(data[:, -4], dtype=float), np.array(data[:, -3], dtype=float), np.array(data[:, -2], dtype=float)
data = data[(Ferr1 < 0.5*Fs) & (-Ferr2 < 0.5*Fs)]
rps, rperr1, rperr2 = np.array(data[:, 8], dtype=float), np.array(data[:, 9], dtype=float), np.array(data[:, 10], dtype=float)
Fs, Ferr1, Ferr2 = np.array(data[:, -4], dtype=float), np.array(data[:, -3], dtype=float), np.array(data[:, -2], dtype=float)

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
bds['m1'] = data[:, 10]
bds['m1_err'] = data[:, [12,11]]
bds['r1'] = data[:, 13]
bds['r1_err'] = data[:, [15,14]]
bds['T'] = data[:, 16]
bds['T_err'] = data[:, [18,17]]
print(len(bds['m2']))

# arg = bds['m2'] < 100

arg = (np.max(bds['m2_err'], axis=1) / bds['m2'] < 0.3) & (np.max(bds['r2_err'], axis=1) / bds['r2'] < 0.3) \
      & (bds['r2'] < 3.0) & (bds['m1'] != -999) & (bds['r1'] != -999) & (bds['T'] != -999)
# arg = np.max(bds['r2_err'], axis=1) / bds['r2']
# print(arg)
# print(np.max(bds['r2_err'], axis=1))
for key in bds.keys():
    bds[key] = bds[key][arg]

print(len(bds['m2']))

def flux(m1, m2, r1, P, e, T):
    a = (c.G * (m1*u.Msun + m2*u.Mjupiter) * (P*u.day)**2 / 4 / (np.pi)**2) ** (1/3)
    f = c.sigma_sb * (T*u.K)**4 * (r1*u.Rsun)**2 / a**2 * (1-e**2)**(-1/2)
    return f.cgs

Fearth = (1*u.Lsun/((1*u.au)**2 * np.pi * 4))

plt.rcParams['figure.figsize'] = [16, 12]
bds['flux'] = flux(bds['m1'], bds['m2'], bds['r1'], bds['P'], bds['e'], bds['T']) / (Fearth.cgs)
m1l, m1u = bds['m1']-bds['m1_err'][:, 0], bds['m1']+bds['m1_err'][:, 1]
m2l, m2u = bds['m2']-bds['m2_err'][:, 0], bds['m2']+bds['m2_err'][:, 1]
r1l, r1u = bds['r1']-bds['r1_err'][:, 0], bds['r1']+bds['r1_err'][:, 1]
Tl, Tu = bds['T']-bds['T_err'][:, 0], bds['T']+bds['T_err'][:, 1]
f_lerr = bds['flux'] - (flux(m1u, m2u, r1l, bds['P'], bds['e'], Tl) / (Fearth.cgs))
f_uerr = (flux(m1l, m2l, r1u, bds['P'], bds['e'], Tu) / (Fearth.cgs)) - bds['flux']


# ln(R) = alpha*ln(M) + beta*ln(F) + ln(C)

def log_likelihood(theta, x1, x2, y, yerr):
    a, b, c = theta
    model = a*x1 + b*x2 + np.log(c)
    return -0.5 * np.sum((y - model) ** 2 / (yerr**2) + np.log(yerr**2))

def log_prior(theta):
    a, b, c = theta
    if c <= 0.0:
        return -np.inf
    if -5.0 < a < 5 and -5.0 < b < 5 and -5.0 < np.log(c) < 5.0:
        return 0.0
    return -np.inf

def log_probability(theta, x1, x2, y, yerr):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, x1, x2, y, yerr)

ys = np.log(bds['r2'])
yerr = np.empty_like(ys)
for i in range(len(yerr)):
    err = bds['r2_err'][i] / bds['r2'][i]
    err = np.max(err)
    yerr[i] = err

x1s = np.log(bds['m2'])
x2s = np.log(bds['flux'])

ys = np.hstack((ys, np.log(rthis)))
yerr = np.hstack((yerr, (np.max(rthis_err, axis=0)/rthis)))
x1s = np.hstack((x1s, np.log(mthis)))
x2s = np.hstack((x2s, np.log(Fthis)))


# low mass star
arg = (np.exp(x1s) > 80)
ys_high, yerr_high, x1s_high, x2s_high = ys[arg], yerr[arg], x1s[arg], x2s[arg]
print(len(ys_high))

np.random.seed(3)
pos = np.array([0., 0., 1]) + np.random.randn(32, 3)
pos[:, 2] = np.exp(pos[:, 2])
nwalkers, ndim = pos.shape
sampler = emcee.EnsembleSampler(
    nwalkers, ndim, log_probability, args=(x1s_high, x2s_high, ys_high, yerr_high)
)
sampler.run_mcmc(pos, 10000, progress=True)

print('low mass stars')
flat_samples_high = sampler.get_chain(discard=500, flat=True)
mcmc = []
for i in range(ndim):
    result = np.percentile(flat_samples_high[:, i], [16, 50, 84])
    mcmc.append(result[1])
    print(result[1], result[0]-result[1], result[2]-result[1])

mcmc_alpha_high, mcmc_beta_high, mcmc_C_high = mcmc


# brown dwarfs
arg = (np.exp(x1s) <= 80)
ys_low, yerr_low, x1s_low, x2s_low = ys[arg], yerr[arg], x1s[arg], x2s[arg]
print(len(ys_low))

np.random.seed(3)
pos = np.array([0., 0., 1]) + np.random.randn(32, 3)
pos[:, 2] = np.exp(pos[:, 2])
nwalkers, ndim = pos.shape
sampler = emcee.EnsembleSampler(
    nwalkers, ndim, log_probability, args=(x1s_low, x2s_low, ys_low, yerr_low)
)
sampler.run_mcmc(pos, 10000, progress=True)

print('brown dwarf')
flat_samples_low = sampler.get_chain(discard=500, flat=True)
mcmc = []
for i in range(ndim):
    result = np.percentile(flat_samples_low[:, i], [16, 50, 84])
    mcmc.append(result[1])
    print(result[1], result[0]-result[1], result[2]-result[1])

mcmc_alpha_low, mcmc_beta_low, mcmc_C_low = mcmc


# plot BDs and low mass stars
r2_err = np.transpose(bds['r2_err'])
norm = plt.Normalize(np.min(bds['m2']), np.max(bds['m2']))
plt.errorbar(bds['flux'], bds['r2'], xerr=[f_lerr, f_uerr], yerr = r2_err, fmt='.', c='black', alpha=0.5)
plt.scatter(bds['flux'], bds['r2'], c=bds['m2'], s=120, zorder=10, norm=norm)#, label='Published transiting companions')
# plt.scatter(0.0001, 0.1, c=80)

# color bar
cb = plt.colorbar()
cb.ax.tick_params(labelsize=25, width=2, length=9, direction='in')

# hot Jupiters
# plt.errorbar(Fs, rps, xerr=[-Ferr2, Ferr1], yerr=[-rperr2, rperr1], fmt='.', c='black', alpha=0.1)
plt.scatter(Fs, rps, c='black', alpha=0.3, s=40)


# this work BDs
plt.errorbar(Fthis, rthis, xerr=Fthis_err, yerr=rthis_err, fmt='.', ms=10)
# plt.scatter(Fthis, rthis, c=mthis, s=160, zorder=10, norm=norm, edgecolor='red', linewidth=3)
plt.scatter(Fthis, rthis, c=mthis, s=120, zorder=10, norm=norm)
# plt.annotate(bdthis[0], xy=(Fthis[0], rthis[0]), xytext=(Fthis[0]-1500, rthis[0]+0.05), fontsize=30, c='red', zorder=10)
# plt.annotate(bdthis[1], xy=(Fthis[1], rthis[1]), xytext=(Fthis[1]-750, rthis[1]+0.05), fontsize=30, c='red', zorder=10)
# plt.annotate(bdthis[2], xy=(Fthis[2], rthis[2]), xytext=(Fthis[2]+2, rthis[2]-0.07), fontsize=30, c='red', zorder=10)


# fit models
fs = np.linspace(0, 4, 1000)
fs = 10**fs
ms = [30, 60, 90, 120]
for m in [30, 60]:
    plt.plot(fs, mcmc_C_low * (m**mcmc_alpha_low) * (fs**mcmc_beta_low), label=f'M={m} $M_J$', lw=3)
for m in [90, 120]:
    plt.plot(fs, mcmc_C_high * (m**mcmc_alpha_high) * (fs**mcmc_beta_high), label=f'M={m} $M_J$', lw=3)

# Weiss model
plt.plot(fs, 2.5*0.089214178 * ((fs*1361166.5)**0.094) * ((1*317.82841)**(-0.039)), ls='dashed', c='black', lw=3)
# plt.plot(fs, 2.5*0.089214178 * ((fs*1361166.5)**0.094) * ((1*317.82841)**(-0.039)), ls='dashed', c='black', lw=3, label='M=1 $M_J$')



plt.xscale('log')
plt.xlim(0.8, 2e4)
plt.ylim(0.55, 2.05)
plt.tick_params(labelsize=25, direction='in', axis='both', width=2, length=12)
plt.tick_params(labelsize=25, direction='in', axis='both', which='minor', width=2, length=6)
plt.xlabel('Incident Flux ($F_{\oplus}$)', fontsize=35)
plt.ylabel('Radius ($R_J$)', fontsize=35)
cb.set_label('Mass ($M_J$)', fontsize=35)
# plt.legend(fontsize=25)
plt.savefig('BD_inflation.png')
plt.clf()


all_labels = [r'$\alpha$', r'$\beta$', 'C']
fig = corner.corner(
    flat_samples_low, labels = all_labels, label_kwargs=dict(fontsize=20)
)
for ax in fig.get_axes():
    ax.tick_params(labelsize=10, direction='in', axis='both', width=1.5, length=9)
fig.savefig('mcmc_post_BD.png')
plt.clf()

fig = corner.corner(
    flat_samples_high, labels = all_labels, label_kwargs=dict(fontsize=20)
)
for ax in fig.get_axes():
    ax.tick_params(labelsize=10, direction='in', axis='both', width=1.5, length=9)
fig.savefig('mcmc_post_stars.png')

