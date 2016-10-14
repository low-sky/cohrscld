import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table
import powerlaw
import astropy.constants as con
import astropy.units as u

mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman']
mpl.rcParams['font.size'] = 14
mpl.rc('xtick', labelsize=14)
mpl.rc('ytick', labelsize=14)
t = Table.read('cohrs_sfecat2.fits')
mlum = t['mlum_msun']

complim = 1e3
R0 = 8.5e3
rgal = (R0**2 + t['distance']**2 - 2 * R0 * t['distance'] *
        np.cos(t['x_coor'] * np.pi / 180))**0.5

fit = powerlaw.Fit(mlum, xmin=complim)
#R, p = fit.distribution_compare('power_law', 'truncated_power_law')


fig = plt.figure(figsize=(5, 4))
ax = fig.add_subplot(111)
ax.axis([1e-1, 3e6, 0.8, 2e4])
msort = np.sort(mlum)
mtrun = 5e5
ntrun = 20
ccdf = np.linspace(mlum.size, 1.0, mlum.size)
ccdfbreak = np.max(ccdf[msort >= complim])
ccdf_pl = ccdfbreak * (msort / complim)**(-fit.alpha + 1)
ccdf_tpl = ntrun * ((msort / mtrun)**(-fit.alpha + 1) - 1)
# ccdf_tpl = ccdfbreak * (msort / complim)**(
#     -fit.alpha + 1) * \
#     np.exp(-fit.truncated_power_law.parameter2 * msort)
ax.scatter(np.sort(mlum), ccdf, 10, marker='o',
           edgecolor='black', facecolor='grey', alpha=0.6)
ax.plot(np.sort(mlum), ccdf_pl, linestyle='--', label='Power Law')
ax.plot(np.sort(mlum), ccdf_tpl, label='Trunc. Power Law')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel(r'$M_{\mathrm{CO}}~(M_{\odot})$')
ax.set_ylabel(r'$N(>M_{\mathrm{CO}})$')
ax.legend(loc='lower left', fontsize=12)
ax.grid()
plt.tight_layout()
plt.savefig('MassSpec.png', dpi=300)
plt.close(fig)
plt.clf()

rgal = np.linspace(1,10,101)
vrot = 200 * u.km / u.s * np.ones_like(rgal)
surfdens = ((1.4 * rgal - 0.6) * (rgal < 4) +\
                (5) * (rgal >= 4) * (rgal < 8.5) +\
                ((-1.12) + (6.12 * rgal / 8.5)) * (rgal >= 8.5)) +\
                4.5 * np.exp(-(rgal - 4.85)**2 / 7.045) * (rgal < 6.97) +\
                1.4 * np.exp(-rgal / 2.89) * (rgal >= 6.97) /\
                np.exp(-8.5 / 2.89)
surfdens = 1.2 * 1.4 * surfdens * u.M_sun / u.pc**2
sigmav = 5 * u.km / u.s
M_J = (np.pi * sigmav**4 / (4 * con.G**2 * surfdens)).to(u.M_sun)
v = 220 * u.km / u.s
r = rgal * u.kpc
kappasq = 2 * v**2 / r**2  # v / r * np.gradient(v, r)
M_T = (4 * np.pi**5 * con.G**2 * surfdens**3 / kappasq**2).to(u.M_sun)

fig2 = plt.figure(figsize=(5, 4))
ax = fig2.add_subplot(111)
ax.axis([1, 10, 1e4, 1e8])
ax.semilogy(rgal, M_J, label=r'$M_{\mathrm{Jeans}}$')
ax.semilogy(rgal, M_T, label=r'$M_{\mathrm{Toomre}}$',
            linestyle='--')
ax.set_xlabel(r'$R_{\mathrm{gal}}$ (kpc)')
ax.set_ylabel('Characterisitic Mass')
ax.fill_between(rgal, M_J.value, M_T.value, where=(M_T > M_J),
                facecolor='grey', alpha=0.5)
ax.legend(loc='lower right')
fig2.tight_layout()
fig2.savefig('CharMass.png', dpi=300)
plt.close(fig2)
plt.clf()
# bins = 5
# edges = np.linspace(0, 100, bins + 1)
# plt.clf()
# fig = plt.figure(figsize=(5, 5))
# ax = fig.add_subplot(111, aspect='equal')
# for pmin, pmax in zip(edges[0:-1], edges[1:]):
#     rmin = np.percentile(rgal, pmin)
#     rmax = np.percentile(rgal, pmax)
#     idx = (rgal >= rmin) * (rgal < rmax)
#     m = np.sort(mlum[idx])
#     n = np.sum(idx)
#     plt.loglog(m, np.linspace(n, 1, n),
#                label='{0:2.1f} kpc - {1:2.1f} kpc'.format(rmin / 1e3,
#                                                           rmax / 1e3),
#                drawstyle='steps')
#     plt.xlim([1e1, 1e7])
# plt.xlabel(r'Mass ($M_\odot$)', size=16)
# plt.ylabel(r'$N(>M)$', size=16)
# plt.legend()
# plt.tight_layout()
# plt.savefig('mass_spec.png',dpi=300)


# edges = np.array([15,25,35,45,55])
# plt.clf()
# fig = plt.figure(figsize=(5,5))
# ax = fig.add_subplot(111,aspect='equal')
# for lmin,lmax in zip(edges[0:-1],edges[1:]):
#     idx = (t['x_coor']>=lmin)*(t['x_coor']<lmax)
#     m = np.sort(mlum[idx])
#     n = np.sum(idx)
#     plt.loglog(m,np.linspace(n,1,n),
#                label=r'{0:2.0f}$^\circ$ < $\ell$ < {1:2.0f}$^\circ$'.format(lmin,lmax),
#                drawstyle='steps')
# plt.xlim([1e1,1e7])
# plt.xlabel(r'Mass ($M_\odot$)',size=16)
# plt.ylabel(r'$N(>M)$',size=16)
# plt.legend()
# plt.tight_layout()
# plt.savefig('mass_angle_spec.png', dpi=300)




