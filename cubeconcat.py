import numpy as np
from spectral_cube import SpectralCube
import astropy.units as u
import glob
import montage_wrapper as montage
import shutil
import os
from astropy.io import fits

workdir = '/mnt/work/erosolow/mosaic_raw/'
outdir = '/mnt/work/erosolow/mosaic_out/'
planesdir = '/mnt/work/erosolow/grs_planes/'

cubedir = '/mnt/bigdata/erosolow/surveys/grs/'
splitdir  ='/mnt/bigdata/erosolow/cohrs/GRS_SPLITS/'

def runall():
    cubesplit()
    planemontage()
    writeplanes()

def cubesplit():
    # Change this to match the pattern of whatever cubes you have.
    flist = glob.glob(cubedir + 'grs-*-cube.fits')
    spaxis = np.linspace(-30, 160, 191) * u.km / u.s
    for thisfile in flist: 
        s = SpectralCube.read(thisfile, hdu=0)
        # If the cubes aren't on a common spectral axis, reproject.
        s2 = s.spectral_interpolate(spaxis, fill_value=np.nan)
        shapevec = s2.shape
        for i in np.arange(shapevec[0]):
            plane = s2[i, :, :]
            vel = spaxis[i].value
            root = thisfile.split('/')[-1]
           # import pdb; pdb.set_trace()
            plane.write(splitdir +
                        root.replace(
                            '.fits', '_{0}.fits'.format(vel)),
                        overwrite=True)

            def planemontage():
#    hdr = fits.getheader('INTEG/COHRS_RELEASE1_FULL_INTEG.fit')
#    try:
#        hdr.tofile('template.hdr')
#    except:
#        pass
    spaxis = np.linspace(-30, 160, 191) * u.km / u.s

    for v in spaxis:
        flist = glob.glob(splitdir + '*cube_{0}.fit*'.format(v.value))
        shutil.rmtree(workdir, True)
        shutil.rmtree(outdir, True)
        os.mkdir(workdir)
        for thisfile in flist:
            shutil.copy(thisfile, workdir)
        montage.mosaic(workdir,outdir,header='template.hdr',
                       mpi=True, n_proc=8, exact_size=True)
        os.rename(outdir + 'mosaic.fits',
                  planesdir + 'GRSPLANE_{0}'.format(v.value)+'.fits')
        shutil.rmtree(outdir, True)
        
def writeplanes(save_name='/mnt/work/erosolow/GRS_13CO_all.fits'):
    spatial_template = fits.open('INTEG/COHRS_RELEASE1_FULL_INTEG.fit')
    spectral_template = SpectralCube.read('reprojected.fits')

    # Smoosh astrometry components together
    spatial_header = spatial_template[0].header
    spectral_header = spectral_template.header

    new_header = spatial_header.copy()
    new_header["NAXIS"] = 3
    for keyword in ['NAXIS3', 'CRVAL3', 'CDELT3','CRPIX3','CUNIT3']:
        new_header[keyword] = spectral_header[keyword]
    new_header['BMAJ'] = 14./3600
    new_header['BMIN'] = 14./3600
    new_header['BPA'] = 0.00
    
    if os.path.exists(save_name):
        raise Exception("The file name {} already "
                        "exists".format(save_name))

    # Open a file and start filling this with planes.
    output_fits = fits.StreamingHDU(save_name, new_header)
    # Again, set  up a common vel axis and spin out
    vel = np.linspace(-30, 160, 191)
    for v in vel:
        output_fits.write(fits.getdata(planesdir +
                                       'GRSPLANE_{0}'.format(v) +
                          '.fits'))

    output_fits.close()


