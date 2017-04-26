from spectral_cube import SpectralCube
import numpy as np
import astropy.units as u
cohrsdir = '/mnt/bigdata/erosolow/cohrs/'

def select_cloud(idxarray, cloudcat):
    for idx in idxarray:
        entry = cloudcat[idx]
        asgn = SpectralCube.read(cohrsdir+'FINALASGNS/'+
                                 entry['orig_file']+
                                 '_fasgn.fits')

        data = SpectralCube.read(cohrsdir+'DATA/'+
                                 entry['orig_file']+
                                 '.fits')
        mask = (asgn == entry['_idx'] *
                u.dimensionless_unscaled)
        cube = data.with_mask(mask)
        cube = cube.minimal_subcube()
        cube.write('cohrscld_{0}'.format(entry['_idx'])+'.fits',
                   overwrite=True)


