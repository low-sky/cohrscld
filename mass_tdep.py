import numpy as np
from astropy.table import Table
import scipy.stats as ss
import matplotlib.pyplot as plt
import astropy.units as u
import astropy.constants as con

cohrs_dir = '/mnt/ephem/ephem/erosolow/cohrs/'
t = Table.read(cohrs_dir + 'cohrs_withir_withsfr.fits')

rho = t['mlum_msun'].data * u.M_sun / (4 * np.pi *
                                       (t['radius_pc'].data * u.pc)**3 / 3)

alpha = ((5 * t['sigv_kms']**2 * (u.km / u.s)**2
          * t['radius_pc'].data * u.pc) *
         (con.G * t['mlum_msun'].data * u.M_sun)**(-1)).to(
             u.dimensionless_unscaled)
filt = (np.isfinite(alpha))
filt = alpha < 4
t = t[filt]
alpha = alpha[filt]
rho = rho[filt]
Mach = t['sigv_kms'] / 0.2
tff = (((3 * np.pi / (32 * con.G * rho))**0.5).to(u.Myr)).value

area = t['area_exact_as'] / 206265**2 * t['bgps_distance_pc']**2
# area = np.pi * (t['radius_pc'].data * u.pc)**2
sfe = 1e6 * (t['sfr_70um']/ area)
x = 0.014 * (alpha / 1.3)**(-0.68) * (Mach / 100)**(-0.32) *\
    t['mlum_msun'].data / tff / area#.value
#x = (0.014 * t['mlum_msun'].data / tff / area)#.value

x = (0.026 * (alpha / 1.3)**(-0.3) * (Mach / 100)**(0.8) *
     t['mlum_msun'].data / tff / area)

y = t['mlum_msun'] / t['sfr_70um']
x = t['mlum_msun']

x = x#.value

y = y#.value
fig, (ax1) = plt.subplots(1)
fig.set_size_inches(5, 4)
idx = t['mlum_msun'].data > 1e2
xmin = 2
xmax = 6
ymin = 7
ymax = 12
val, edges, _ = ss.binned_statistic(np.log10(x[idx]),
                                    np.log10(y[idx]),
                                    statistic=np.nanmedian, bins=10)

histdata, xedge, yedge = np.histogram2d(np.log10(x[idx]),
                                        np.log10(y[idx]),
                                        range=[[xmin, xmax], [ymin, ymax]],
                                        bins=40)
ax1.scatter(x[idx], y[idx], edgecolor='k',
            facecolor='none', zorder=-99)
ax1.plot(1e1**(0.5 * (edges[1:] + edges[0:-1])), 1e1**val, color='red', lw=3)
ax1.set_xscale('log')
ax1.set_yscale('log')

ax1.set_xlabel(r'$M_{\mathrm{lum}}\ (M_{\odot})$', size=16)
ax1.set_ylabel(r'$\tau_{\mathrm{dep}}\ (\mathrm{yr})$', size=16)
# ax1.set_xlim(1e1**])
ax1.set_ylim([1e1**ymin, 1e1**ymax])
ax1.set_xlim([1e1**xmin, 1e1**xmax])

histdata[histdata < 3] = np.nan

ax1.grid()
print(xmin, xmax)
cax = ax1.pcolormesh(1e1**xedge, 1e1**yedge,
                     np.ma.fix_invalid(histdata.T), vmin=1, vmax=np.nanmax(histdata))
# cax = ax1.imshow(histdata.T, extent=[1e1**xmin, 1e1**xmax,
#                                      1e1**ymin, 1e1**ymax], origin='lower',
#                  interpolation='nearest', cmap='inferno', vmin=2, aspect='auto')
# ax1.plot([1e1**xmin, 1e1**xmax],[1e1**ymin, 1e1**ymax], alpha=0.5, lw=3)
cb = fig.colorbar(cax)
cb.set_label(r'Number')
fig.tight_layout()
plt.savefig('mass_tdep.png', dpi=300)
plt.close(fig)
plt.clf()
