import numpy as np
from astropy.table import Table
import scipy.stats as ss
import matplotlib.pyplot as plt
import astropy.units as u
import astropy.constants as con
import emcee

def lp(p, sfeff, sdff, vp, mach):
    lp = -np.nansum(np.abs(np.log(sfeff) - (np.log(sdff)
                                            + p[0]
                                            + p[1] * np.log(vp/1.3)
                                            + p[2] * np.log(mach/100))
    )**1)
    return(lp)

cohrs_dir = '/mnt/ephem/ephem/erosolow/cohrs/'
t = Table.read(cohrs_dir + 'cohrs_withir_withsfr.fits')
t = t[t['n_pixel']>1e3]
#t = t[t['mlum_msun']>1e3]

rho = t['mlum_msun'].data * u.M_sun / (4 * np.pi *
                                       (t['radius_pc'].data * u.pc)**3 / 3)

alpha = ((5 * t['sigv_kms']**2 * (u.km / u.s)**2
          * t['radius_pc'].data * u.pc) *
         (con.G * t['mlum_msun'].data * u.M_sun)**(-1)).to(
             u.dimensionless_unscaled)
Mach = t['sigv_kms'] / 0.2
tff = (((3 * np.pi / (32 * con.G * rho))**0.5).to(u.Myr)).value

area = t['area_exact_as'] / 206265**2 * t['bgps_distance_pc']**2
# area = np.pi * (t['radius_pc'].data * u.pc)**2
sfe = 1e6 * (t['sfr_70um']/ area)
sdff = t['mlum_msun'].data / tff / area


epsff_obs = sfe / (t['mlum_msun'].data / tff / area)
leps = np.log10(epsff_obs)
leps = leps[np.isfinite(leps)]
fig, ax = plt.subplots(1,1)
fig.set_size_inches(4,3)
ax.hist(leps,bins=40)
ax.set_xlabel(r'$\log_{10}(\mathrm{SFE}_{\mathrm{ff}})$')
ax.set_ylabel(r'$N$')
plt.tight_layout()
plt.savefig('sfe_obs.png')

