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


alpha = ((5 * t['sigv_kms']**2 * (u.km / u.s)**2 * t['radius'] * u.pc) *
         (con.G * t['mlum_msun'] * u.M_sun)**(-1)).to(u.dimensionless_unscaled)
Mach = t['sigv_kms'] / 0.2
tff = (((3 * np.pi / (32 * con.G * rho))**0.5).to(u.Myr)).value
area = np.pi * (t['radius'] * u.pc)**2
x = 0.014 * (alpha / 1.3)**(-0.68) * (Mach / 100)**(-0.32) *\
    t['mlum_msun'] / tff / area.value
y = 2e-10 * cloud_irlum * 1e6 / area.value

epsff = np.log10(2e-10 * cloud_irlum * 1e6 / (t['mlum_msun'] / tff))
idx = (t['mlum_msun'] > 3e3) * (np.isfinite(epsff))
epsff = epsff[idx]
mlum = t['mlum_msun'][idx]


fig, (ax1) = plt.subplots(1)
fig.set_size_inches(5, 4)

xmin = -5
xmax = 1
ymin = -5
ymax = 1
ax1.hist(epsff,bins=30)

fig.tight_layout()
plt.savefig('eps_ff.png', dpi=300)
plt.close(fig)
plt.clf()
