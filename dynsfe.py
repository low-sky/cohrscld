import numpy as np
from astropy.table import Table
import scipy.stats as ss
import matplotlib.pyplot as plt
import astropy.units as u
import astropy.constants as con
import sys
import matplotlib as mpl
t = Table.read('/mnt/bigdata/erosolow/cohrs_work/output_catalog_withsfr.fits')

if sys.platform == 'darwin':
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['font.serif'] = 'Times New Roman'

elif sys.platform == 'linux2':
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['font.serif'] = 'FreeSerif'

apix_sr = 8.46159e-10

mlum = t['mlum_msun']
cloud_irlum = (t['ir_luminosity'] - t['bg_lum']) * 6e11 * apix_sr
sfeobs = 2e-10 * cloud_irlum / t['mlum_msun'] * 1e6
R0 = 8.5e3
rgal = (R0**2 + t['distance']**2 -
        2 * R0 * t['distance'] * np.cos(t['x_coor'] * np.pi / 180))**0.5

kappa = 2**0.5 * 220 * u.km/u.s / (rgal * u.pc)

surfdens = t['mlum_msun'] * u.M_sun / (np.pi * t['radius']**2 * u.pc**2)
k = 1.
gamma = 10.
sgfrac = (np.pi * (1-k/3)/(1-2*k/5) *
                          con.G * surfdens / gamma**2 /
                          kappa**2 /
                          t['radius'] / u.pc)**(0.5*(3-k))

sfedyn = 0.046 / u.Myr * sgfrac.to(u.dimensionless_unscaled)

sfedyn = sfedyn.to(1 / u.Myr).value

x = sfedyn
y = sfeobs
x0 = 1e-5
x1 = 1e-1
y0 = 1e-5
y1 = 1e-1
fig, (ax1) = plt.subplots(1)
fig.set_size_inches(5, 4)
idx = mlum > 1e3
val, edges, _ = ss.binned_statistic(np.log10(x[idx]),
                                    np.log10(y[idx]),
                                    statistic=np.nanmedian, bins=10)

histdata, xedge, yedge = np.histogram2d(np.log10(x[idx]),
                                        np.log10(y[idx]),
                                        bins=[np.linspace(np.log10(x0),
                                                          np.log10(x1),31),
                                              np.linspace(np.log10(y0),
                                                          np.log10(y1),31)])

ax1.scatter(x[idx], y[idx], edgecolor='k',
            facecolor='none', zorder=-99)
ax1.plot(1e1**(0.5 * (edges[1:] + edges[0:-1])), 1e1**val, color='green', lw=3)
ax1.set_xscale('log')
ax1.set_yscale('log')


ax1.set_xlabel(r'$\epsilon_{\mathrm{dyn}} (\mathrm{Myr}^{-1})$', size=16)
ax1.set_ylabel(r'$\dot{M}_\star / M_c \ (\mathrm{Myr}^{-1})$ ',
               size=16)
# ax1.set_xlim(1e1**])
ax1.set_xlim([x0, x1])
ax1.set_ylim([y0, y1])

#histdata[histdata < 3] = np.nan
#cax = ax1.imshow(histdata.T, extent=(x0, x1, y0, y1),
#                 origin='lower',
#                 interpolation='nearest',
#                 cmap='inferno', vmin=2, aspect='auto')

ax1.grid()

#cb = fig.colorbar(cax)
#cb.set_label(r'Number')
fig.tight_layout()
plt.savefig('sfe_dyn.png', dpi=300)
plt.close(fig)
plt.clf()
