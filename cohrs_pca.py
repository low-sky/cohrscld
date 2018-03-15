from turbustat.statistics import pca, vca
from astropy.table import Table
import numpy as np
from astropy.table import Table, Column
import scipy.stats as ss
import matplotlib.pyplot as plt
import astropy.units as u
import astropy.constants as con
import os
from astropy.io import fits
from astropy import wcs
from spectral_cube import SpectralCube

datadir = '/mnt/ephem/ephem/erosolow/cohrs/'
t = Table.read(datadir + 'cohrs_withir_withsfr.fits')
t = t[t['n_pixel']>1e5]
#t = t[t['mlum_msun']>1e5]
def calc_structure_fcn(catalog,
                       bootiter=0,doPlot=False):
    
    cat = catalog
    if 'sf_offset' not in cat.keys():
        keylist = ['sf_offset','sf_index','sf_ngood',
                   'sf_index_err','sf_offset_err', 'vca_index',
                   'vca_index_err']
        for thiskey in keylist:
            c = Column(np.zeros(len(cat))+np.nan,name=thiskey)
            cat.add_column(c)
        
    current_open_file = ''
    for cloud in cat:
        root = cloud['orig_file'].split('_')
        orig_file = 'COHRS_{0}_{1}.fits'.format(root[-2], root[-1])
        asgn_file = cloud['orig_file'] + '_fasgn.fits'
        if os.path.isfile(datadir+'COHRS_tiles/'+orig_file):
            if current_open_file != datadir+'COHRS_tiles/'+orig_file:
                hdu = fits.open(datadir+'COHRS_tiles/'+orig_file,memmap=False)
                cat.write('cohrs_structurefunc.fits',overwrite=True)
                w = wcs.WCS(hdu[0].header)
                hdr2 = w.to_header()
                hdr2['BMAJ'] = 15./3600
                hdr2['BMIN'] = 15./3600
                hdr2['BPA'] = 0.
                co = SpectralCube(hdu[0].data,w,header=hdr2)
                hdu = fits.open(datadir+'ASGN/'+asgn_file,memmap=False)
                w = wcs.WCS(hdu[0].header)
                hdr2 = w.to_header()
                hdr2['BMAJ'] = 15./3600
                hdr2['BMIN'] = 15./3600
                hdr2['BPA'] = 0.
                asgn = SpectralCube(hdu[0].data,w,header=hdr2)

#                masked_co = co.with_mask(asgn>0*u.dimensionless_unscaled)
#                moment = masked_co.moment(0)
                
                current_open_file = datadir+'COHRS_tiles/'+orig_file

            print(cloud['cloud_id'])
            mask = (asgn == cloud['cloud_id']*u.dimensionless_unscaled)
            subcube = co.with_mask(mask)
            subcube = subcube.subcube_from_mask(mask)
#            subcube.moment0().quicklook()
#            plt.show()
            if subcube.shape[0] > 10:
                # zeros = np.zeros_like(subcube) # np.random.randn(*subcube.shape)*0.5
                # concatdata = np.vstack([subcube.filled_data[:],zeros])
                # hdr2 = subcube.wcs.to_header()
                # hdr2['BMAJ'] = 15./3600
                # hdr2['BMIN'] = 15./3600
                # hdr2['BPA'] = 0.
                # newcube = SpectralCube(concatdata,subcube.wcs,header=hdr2)
                # if True:
                try:
                    pcaobj = pca.PCA(subcube,
                                     distance=cloud['bgps_distance_pc']*u.pc)
                    vcaobj = vca.VCA(subcube,
                                     distance=cloud['bgps_distance_pc']*u.pc)
                    vcaobj.run(verbose=False, high_cut = 0.3/u.pix)
                    pcaobj.run(verbose=True,mean_sub=True,
                               min_eigval=0.9,eigen_cut_method='proportion')
                    cloud['sf_index']= pcaobj.index
                    cloud['sf_offset'] = pcaobj.intercept.value
                    cloud['sf_ngood'] = np.min([np.isfinite(pcaobj.spatial_width).sum(),
                                                np.isfinite(pcaobj.spectral_width).sum()])
                    
                    cloud['sf_index_err'] = (pcaobj.index_error_range[1]-pcaobj.index_error_range[0])*0.5
                    cloud['sf_offset_err'] = ((pcaobj.intercept_error_range[1]-pcaobj.intercept_error_range[0])*0.5).value
                    cloud['vca_index'] = vcaobj.slope
                    cloud['vca_index_err'] = vcaobj.slope_err
                    print('{0} +/- {1}'.format(cloud['sf_index'],
                                               cloud['sf_index_err']),
                          cloud['sf_offset'])
                    # import pdb; pdb.set_trace()
                # else:
                except:
                    pass
        
    cat.write('output_catalog2.fits',overwrite=True)
