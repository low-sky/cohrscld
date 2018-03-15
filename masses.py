from astropy.table import Table
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import corner
from astropy.stats import median_absolute_deviation as mad
t = Table.read('/mnt/ephem/ephem/erosolow/cohrs/cohrs_withlines.fits')

fig = plt.figure(figsize=(4.5,3.5))
ax = fig.add_subplot(111)
idx = np.isfinite(np.log10((t['13co10'])/t['12co32']))
t2 = t[idx]
corner.hist2d(np.log10(t[idx]['flux']),
              np.log10((t[idx]['13co10'])/t[idx]['12co32']),
              ax=ax)

# ax.scatter(t['flux'], t['13co10']/t['12co32'], edgecolor='k',
#            facecolor='gray', alpha=0.2)
ax.set_ylim(-3,0)
ax.set_xlabel(r'$\log_{10}[F_{\mathrm{CO(3-2)}}/(\mathrm{K\ km\ s^{-1}\ pc^{2}})]$')
ax.set_ylabel(r'$\log_{10}[F_{\mathrm{{}^{13}CO(1-0)}}/F_{\mathrm{CO(3-2)}}]$')
plt.tight_layout()
#ax.set_xscale("log", nonposx='clip')
#ax.set_yscale("log", nonposy='clip')

idx2 = t2['flux']>1e3
print(np.median(t2[idx2]['13co10']/t2[idx2]['12co32']))
print(mad(np.log(t2[idx2]['13co10']/t2[idx2]['12co32'])))
count = idx2.sum()
print('Number of clouds: {0}'.format(count))


plt.savefig('R13.pdf')
# sns.jointplot(np.log10(t[idx]['flux']),
#               np.log10((t[idx]['13co10'])/t[idx]['12co32']),
#               kind="hex")
# plt.ylim(-3,1)

