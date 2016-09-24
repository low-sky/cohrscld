import numpy as np
import seaborn as sns
from astropy.table import Table, join
import scipy.stats as ss
import matplotlib.pyplot as plt
#t1=Table.read('output_catalog.fits')
#t2 = Table.read('output_catalog2.fits')
#t = t1

#t = Table.read('output_catalog_withsfr.fits')

t = Table.read('output_catalog.fits')
t2 = Table.read('cohrs_ultimatecatalog3.fits')

for tag in ('x_coor','y_coor','v_coor'):
    t[tag]=t2[tag]


apix_sr = 8.46159e-10

mlum = t['mlum_msun']
cloud_irlum = (t['ir_luminosity']-t['bg_lum'])*6e11*apix_sr
sfe = cloud_irlum/mlum
R0 = 8.5e3
rgal = (R0**2+t['distance']**2-2*R0*t['distance']*np.cos(t['x_coor']*np.pi/180))**0.5
# idx = (rgal>3e3)*(rgal<7e3)
# val,edges,_ = ss.binned_statistic(rgal[idx]/1e3,sfe[idx],statistic=np.nanmedian,bins=20)
# plt.semilogy(rgal/1e3,sfe,'ro')
# plt.plot(0.5*(edges[1:]+edges[0:-1]),val)
# plt.xlabel(r'$R_{\mathrm{gal}}$ (kpc)',size=20)
# plt.ylabel(r'$L_{\mathrm{IR}}/M_{\mathrm{CO}}\ (L_{\odot}/M_{\odot})$')
# plt.xlim(3,7)
# plt.savefig('sfe_rgal.pdf')
# plt.clf()

fig, (ax1) = plt.subplots(1)
fig.set_size_inches(5,4)
idx = mlum>1e2
val,edges,_ = ss.binned_statistic(np.log10(mlum[idx]),np.log10(sfe[idx]),statistic=np.nanmedian,bins=10)

histdata,xedge,yedge = np.histogram2d(np.log10(mlum[idx]),np.log10(sfe[idx]),
                                      range=[[2,6],[-2,2]],bins=40)
#sns.heatmap(histdata.T,mask=histdata.T<1,ax=ax1)


ax1.scatter(mlum[idx],sfe[idx],edgecolor='k',facecolor='none',zorder=-99)
ax1.plot(1e1**(0.5*(edges[1:]+edges[0:-1])),1e1**val)
ax1.set_xscale('log')
ax1.set_yscale('log')

ax1.set_xlabel(r'$M_{\mathrm{CO}}\ (M_{\odot})$',size=16)
ax1.set_ylabel(r'$L_{\mathrm{IR}}/M_{\mathrm{CO}}\ (L_{\odot}/M_{\odot})$',
               size=16)
ax1.set_xlim([1e2,1e6])
ax1.set_ylim([1e-2,1e2])

histdata[histdata<3]=np.nan

cax = ax1.imshow(histdata.T,extent=(1e2,1e6,1e-2,1e2),origin='lower',
                 interpolation='nearest',cmap='inferno',vmin=2)
cb = fig.colorbar(cax)
cb.set_label(r'Number')
fig.tight_layout()
plt.savefig('sfe_gmc.pdf')
plt.close(fig)
plt.clf()
