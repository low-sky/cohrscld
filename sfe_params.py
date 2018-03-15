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

ndim = 3
nwalkers = ndim * 10
p0 = np.zeros((nwalkers, ndim))
p0[:, 0] = np.random.randn(nwalkers) * 0.1 + np.log(0.014)
p0[:, 1] = np.random.randn(nwalkers) * 0.1 -0.68
p0[:, 2] = np.random.randn(nwalkers) * 0.1 -0.32
sampler = emcee.EnsembleSampler(nwalkers,
                                ndim, lp,
                                args=[sfe, sdff, alpha, Mach])

pos, prob, state = sampler.run_mcmc(p0, 200)
sampler.reset()
pos, prob, state = sampler.run_mcmc(pos, 1000, thin=10)

epsff = np.exp(np.median(sampler.flatchain[:,0])
               + np.median(sampler.flatchain[:,1]) * np.log(alpha/1.3)
               + np.median(sampler.flatchain[:,2]) * np.log(Mach/100))
#eps = np.median(epsff)
sfe_obs = t['mlum_msun'].data / tff / area * epsff

