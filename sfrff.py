import numpy as np
from astropy.table import Table
import scipy.stats as ss
import matplotlib.pyplot as plt
import astropy.units as u
import astropy.constants as con
# t = Table.read('/Users/erik/code/cohrscld/output_catalog_withsfr.fits')
t = Table.read('/mnt/ephem/ephem/erosolow/cohrs/cohrs_withir.fits')
apix_sr = 8.46159e-10
distance = t['lars1_distance_pc'].data
mlum = t['mlum_msun']
cloud_irlum = ((t['flux_tir'] - t['bg_tir']) * distance**2 *
               3.086e18**2*np.pi*4 / 3.84e33) * apix_sr # Lsun
sfr = cloud_irlum * 2e-10  #solar masses per year

sfe = cloud_irlum / mlum
R0 = 8.5e3
rgal = (R0**2 + distance**2 -
        2 * R0 * distance * np.cos(t['x_coor'] * np.pi / 180))**0.5

rho = t['mlum_msun'].data * u.M_sun / (4 * np.pi *
                                       (t['radius'].data * u.pc)**3 / 3)


alpha = ((5 * t['sigv_kms']**2 * (u.km / u.s)**2 * t['radius'] * u.pc) *
         (con.G * t['mlum_msun'] * u.M_sun)**(-1)).to(u.dimensionless_unscaled)
Mach = t['sigv_kms'] / 0.2
tff = (((3 * np.pi / (32 * con.G * rho))**0.5).to(u.Myr)).value
area = np.pi * (t['radius'] * u.pc)**2
x = 0.014 * (alpha / 1.3)**(-0.68) * (Mach / 100)**(-0.32) *\
    t['mlum_msun'] / tff / area.value
y = 2e-10 * cloud_irlum * 1e6 / area.value

fig, (ax1) = plt.subplots(1)
fig.set_size_inches(5, 4)
idx = mlum > 1e2
xmin = -5
xmax = 1
ymin = -5
ymax = 1
val, edges, _ = ss.binned_statistic(np.log10(x[idx]),
                                    np.log10(y[idx]),
                                    statistic=np.nanmedian, bins=10)

histdata, xedge, yedge = np.histogram2d(np.log10(x[idx]),
                                        np.log10(y[idx]),
                                        range=[[xmin, xmax], [ymin, ymax]],
                                        bins=40)
ax1.scatter(x[idx], y[idx], edgecolor='k',
            facecolor='none', zorder=-99)
ax1.plot(1e1**(0.5 * (edges[1:] + edges[0:-1])), 1e1**val, color='green', lw=3)
ax1.set_xscale('log')
ax1.set_yscale('log')

ax1.set_xlabel(r'$\mathrm{SFR_{ff}} \Sigma_{\mathrm{CO}} ' +
               r't_{\mathrm{ff}}^{-1}\ (M_{\odot}\ ' +
               r'\mathrm{Myr}^{-1}\ \mathrm{pc}^{-2})$', size=16)
ax1.set_ylabel(r'$\dot{\Sigma}_{\star} (M_{\odot}\ ' +
               r'\mathrm{Myr}^{-1}\ \mathrm{pc}^{-2})$',
               size=16)
# ax1.set_xlim(1e1**])
ax1.set_ylim([1e1**ymin, 1e1**ymax])
ax1.set_xlim([1e1**xmin, 1e1**xmax])
histdata[histdata < 3] = np.nan

ax1.grid()

cax = ax1.imshow(histdata.T, extent=(1e1**xmin, 1e1**xmax,
                                     1e1**ymin, 1e1**ymax), origin='lower',
                 interpolation='nearest', cmap='inferno', vmin=2, aspect='auto')
cb = fig.colorbar(cax)
cb.set_label(r'Number')
fig.tight_layout()
plt.savefig('sfe_virparam.png', dpi=300)
plt.close(fig)
plt.clf()
