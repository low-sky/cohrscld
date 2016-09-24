import numpy as np
from astropy.table import Table, join
import scipy.stats as ss
import matplotlib.pyplot as plt
#t1=Table.read('output_catalog.fits')
#t2 = Table.read('output_catalog2.fits')
#t = t1

t = Table.read('output_catalog_withsfr.fits')

mlum = t['flux']*t['distance']**2/109949271.0*2.5
order1 = np.argsort(mlum)
order2 = np.argsort(t['mlum_msun'])
sfe = 500*(t['ir_luminosity']-t['bg_lum'])/mlum
R0 = 8.5e3
rgal = (R0**2+t['distance']**2-2*R0*t['distance']*np.cos(t['x_coor']*np.pi/180))**0.5
idx = (rgal>3e3)*(rgal<7e3)
val,edges,_ = ss.binned_statistic(rgal[idx]/1e3,sfe[idx],statistic=np.nanmedian,bins=20)
plt.semilogy(rgal/1e3,sfe,'ro')
plt.plot(0.5*(edges[1:]+edges[0:-1]),val,color='r',lw=3,alpha=0.6)
plt.xlabel(r'$R_{\mathrm{gal}}$ (kpc)',size=20)
plt.ylabel(r'$L_{\mathrm{IR}}/M_{\mathrm{CO}}\ (L_{\odot}/M_{\odot})$')
plt.xlim(3,7)
plt.clf()


fig = plt.figure(figsize=(5,4))
mlum *= 4
idx = mlum>1e2
histdata,xax,yax = np.histogram2d(np.log10(sfe[idx]),np.log10(mlum[idx]),bins=[np.linspace(-2,2,21),np.linspace(2,6.5,24)])
histdata[histdata < 10]=np.nan
plt.plot(mlum[idx],sfe[idx],'k.',zorder=-99)
plt.imshow(histdata,extent=[1e2,3e6,1e-2,1e2],interpolation='Nearest',cmap='inferno',aspect='auto',origin='lower',vmin=1)
plt.xscale('log')
plt.yscale('log')
# plt.loglog(mlum,sfe,'r.')
val,edges,_ = ss.binned_statistic(np.log10(mlum[idx]),np.log10(sfe[idx]),statistic=np.nanmedian,bins=10)


plt.plot(1e1**(0.5*(edges[1:]+edges[0:-1])),1e1**val)
plt.xlabel(r'$M_{\mathrm{CO}}\ (M_{\odot})$',size=16)
plt.ylabel(r'$L_{\mathrm{IR}}/M_{\mathrm{CO}}\ (L_{\odot}/M_{\odot})$',size=16)
plt.xlim([1e2,5e6])
plt.ylim([1e-2,1e2])
plt.tight_layout()
plt.savefig('sfe_gmc_binned.png',dpi=300)
