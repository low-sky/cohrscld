from astropy.table import Table,Column
import numpy as np
from spectral_cube import SpectralCube
import astropy.io.fits as fits
import os, subprocess

def tileall():
    catalog = Table.read('cohrs_fastphyscatalog_bessel.fits')
    tilecube(catalog, infile='COHRS_all.fits',
             outdir = 'COHRS_tiles', root='COHRS')
    tilecube(catalog, infile='CHIPS_13CO_all.fits',
             outdir = 'CHIPS_13CO_tiles', root='CHIMPS_13CO')
    tilecube(catalog, infile='CHIPS_C18O_all.fits',
             outdir = 'CHIPS_C18O_tiles', root='CHIMPS_C18O')
    tilecube(catalog,
             infile='GRS_13CO_all.fits',
             outdir = 'GRS_tiles',
             root='GRS_13CO')

def tilecube(catalog, infile='COHRS_all.fits', outdir='COHRS_tiles',
             root=None):
    uniqfiles = np.unique(catalog['orig_file'])
    s = SpectralCube.read(infile)
    if not os.path.exists(outdir):
        os.mkdir(outdir)
        
    for thisfile in uniqfiles:
        print "Now processing {0}".format(thisfile)
        xstart = int((thisfile.split('_'))[2])
        xend = int((thisfile.split('_'))[3])
        subcube = s[:,:,xstart:xend]
        if not root:
            root = (infile.split('_'))[0]
        subcube.write(outdir+'/'+root+'_{0}_{1}.fits'.format(xstart,
                                                             xend),
                      overwrite=True)
