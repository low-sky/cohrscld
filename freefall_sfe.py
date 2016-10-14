import numpy as np
from astropy.table import Table
import scipy.stats as ss
import matplotlib.pyplot as plt
import astropy.units as u
import astropy.constants as con
t = Table.read('/Users/erik/code/cohrscld/output_catalog_withsfr.fits')
apix_sr = 8.46159e-10

mlum = t['mlum_msun']
cloud_irlum = (t['ir_luminosity'] - t['bg_lum']) * 6e11 * apix_sr
sfe = cloud_irlum / mlum
R0 = 8.5e3
rgal = (R0**2 + t['distance']**2 -
        2 * R0 * t['distance'] * np.cos(t['x_coor'] * np.pi / 180))**0.5

rho = t['mlum_msun'].data * u.M_sun / (4 * np.pi *
                                       (t['radius'].data * u.pc)**3 / 3)

x = (((3 * np.pi / (32 * con.G * rho))**0.5).to(u.Myr)).value
y = sfe.data

fig, (ax1) = plt.subplots(1)
fig.set_size_inches(5, 4)
idx = mlum > 3e3
val, edges, _ = ss.binned_statistic(np.log10(x[idx]),
                                    np.log10(y[idx]),
                                    statistic=np.nanmedian, bins=10)

histdata, xedge, yedge = np.histogram2d(np.log10(x[idx]),
                                        np.log10(y[idx]),
                                        range=[[0, 2], [-2, 2]],
                                        bins=40)
ax1.scatter(x[idx], y[idx], edgecolor='k',
            facecolor='none', zorder=-99)
ax1.plot(1e1**(0.5 * (edges[1:] + edges[0:-1])), 1e1**val, color='green', lw=3)
ax1.set_xscale('log')
ax1.set_yscale('log')

ax1.set_xlabel(r'$t_{\mathrm{ff}}$' +
               r' (Myr)', size=16)
ax1.set_ylabel(r'$L_{\mathrm{IR}}/M_{\mathrm{CO}}\ (L_{\odot}/M_{\odot})$',
               size=16)
# ax1.set_xlim(1e1**])
ax1.set_ylim([1e-2, 1e2])
ax1.set_xlim([1e0, 1e2])
histdata[histdata < 3] = np.nan

ax1.grid()

cax = ax1.imshow(histdata.T, extent=(1, 1e2, 1e-2, 1e2), origin='lower',
                 interpolation='nearest', cmap='inferno', vmin=2, aspect='auto')
cb = fig.colorbar(cax)
cb.set_label(r'Number')
fig.tight_layout()
plt.savefig('sfe_tff.png', dpi=300)
plt.close(fig)
plt.clf()
