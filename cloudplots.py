import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman']
mpl.rcParams['font.size'] = 14
mpl.rc('xtick', labelsize=14)
mpl.rc('ytick', labelsize=14)
t = Table.read('cohrs_sfecat2.fits')
mlum = t['mlum_msun']

R0 = 8.5e3
rgal = (R0**2+t['distance']**2-2*R0*t['distance']*np.cos(t['x_coor']*np.pi/180))**0.5

bins = 5
edges = np.linspace(0, 100, bins+1)
plt.clf()
fig = plt.figure(figsize=(5, 4))
ax = fig.add_subplot(111, aspect='equal')
for pmin, pmax in zip(edges[0:-1], edges[1:]):
    rmin = np.percentile(rgal, pmin)
    rmax = np.percentile(rgal, pmax)
    idx = (rgal >= rmin) * (rgal < rmax)
    m = np.sort(mlum[idx])
    n = np.sum(idx)
    plt.loglog(m, np.linspace(n, 1, n),
               label='{0:2.1f} kpc - {1:2.1f} kpc'.format(rmin / 1e3,
                                                          rmax / 1e3),
               drawstyle='steps')
    plt.xlim([1e1, 2e6])
plt.xlabel(r'Mass ($M_\odot$)')
plt.ylabel(r'$N(>M)$')
plt.legend(loc='lower left', fontsize=12)
plt.grid()
plt.tight_layout()

plt.savefig('mass_spec.png',dpi=300)


edges = np.array([15,25,35,45,55])
plt.clf()
fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(111,aspect='equal')
for lmin,lmax in zip(edges[0:-1],edges[1:]):
    idx = (t['x_coor']>=lmin)*(t['x_coor']<lmax)
    m = np.sort(mlum[idx])
    n = np.sum(idx)
    plt.loglog(m,np.linspace(n,1,n),
               label=r'{0:2.0f}$^\circ$ < $\ell$ < {1:2.0f}$^\circ$'.format(lmin,lmax),
               drawstyle='steps')
plt.xlim([1e1,1e7])
plt.xlabel(r'Mass ($M_\odot$)',size=16)
plt.ylabel(r'$N(>M)$',size=16)
plt.legend()
plt.tight_layout()
plt.savefig('mass_angle_spec.png', dpi=300)




