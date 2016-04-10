from astropy.table import Table,Column
import montage_wrapper as montage
import matplotlib.pyplot as plt
import numpy as np
from spectral_cube import SpectralCube
import astropy.io.fits as fits
import astropy.wcs as wcs
import astropy.units as u
import os, subprocess
import cloudpca

datadir = '/Users/erik/astro/cohrs/'
outdir = '/Users/erik/astro/cohrs/RESULTS/'
datadir = '/home/erosolow/fastdata/cohrs/'
outdir = '/home/erosolow/fastdata/RESULTS/'


def idenfity_overlaps(fitsfiles, search_directory):
    """
    Identifies overlaps between a list of fits files and a survey given in the directory.
    """
    raise NotImplementedError("But probably should be")

def inventory(search = '70to500'):
    import glob
    cofiles = glob.glob(datadir+'COHRS/*fits')
    cofoot_dict = {}
    for thisfile in cofiles:
        shortname = (thisfile.split('/'))[-1]
        hdr = fits.getheader(thisfile)
        w = wcs.WCS(thisfile)
        footprint = w.celestial.calc_footprint()
        cofoot_dict[shortname] = footprint

    higalfiles = glob.glob(datadir+'HIGAL/*'+search+'*fits')
    higalfoot_dict = {}
    for thisfile in higalfiles:
        shortname = (thisfile.split('/'))[-1]
        hdr = fits.getheader(thisfile)
        w = wcs.WCS(thisfile)
        footprint = w.celestial.calc_footprint()
        higalfoot_dict[shortname] = footprint
    return(higalfoot_dict,cofoot_dict)

def identify_higal_overlaps(cofile, COFootprint = None, HigalFootprint = None):
    import matplotlib.path as path
    if COFootprint is None or HigalFootprint is None:
        HigalFootprint, COFootprint = inventory()
    if '/' in cofile:
        shortname = (cofile.split('/'))[-1]
    else:
        shortname = cofile
    co_vertices = COFootprint[shortname]
    overlapping_files = []
    for higal_file in  HigalFootprint:
        higal_vertices = HigalFootprint[higal_file]
        boundary = path.Path(higal_vertices,closed=True)
        if np.any(boundary.contains_points(co_vertices)):
            overlapping_files.append(higal_file)
    return(overlapping_files)

def match_higal_to_cohrs(output='HIGAL_MATCHED',search='70to500'):
    higaldict, codict = inventory(search=search)
    for cofile in codict:
        matching = identify_higal_overlaps(cofile, COFootprint=codict, 
                                           HigalFootprint = higaldict)
        if matching:
            subprocess.call('rm -rf montagetmp tmp.fits '+
                            'montage_output '+
                            'template_header.hdr',shell=True)
            os.mkdir('montagetmp')
            for thisfile in matching:
                os.symlink(datadir+'/HIGAL/'+thisfile,'montagetmp/'+thisfile)
            s = SpectralCube.read(datadir+'COHRS/'+cofile)
            thisslice = s[0,:,:]
            thisslice.write('tmp.fits')
            hdr = fits.getheader('tmp.fits')
            hdr.totextfile('template_header.hdr')
            montage.mosaic('montagetmp','montage_output',
                           header='template_header.hdr',exact_size=True)
            new_higal_name = cofile.replace('cohrs','higal_xmatch')
            os.rename('montage_output/mosaic.fits', datadir+output+'/'+new_higal_name)
            subprocess.call('rm -rf montagetmp tmp.fits '+
                            'montage_output '+
                            'template_header.hdr',shell=True)

def calc_irlum(catalog = 'cohrs_ultimatecatalog4p.fits', refresh=False):
    cat = Table.read(catalog)
    current_open_file = ''
    if 'ir_luminosity' not in cat.keys():
        IRlum = Column(np.zeros(len(cat))+np.nan,name='ir_luminosity')
        IRflux = Column(np.zeros(len(cat))+np.nan,name='ir_flux')
        IRfluxshort = Column(np.zeros(len(cat))+np.nan,name='ir_flux_short')
        IRlumshort = Column(np.zeros(len(cat))+np.nan,name='ir_lum_short')
        cat.add_column(IRlum)
        cat.add_column(IRflux)
        cat.add_column(IRfluxshort)
        cat.add_column(IRlumshort)

    for cloud in cat:
        if np.isnan(cloud['ir_luminosity']) or refresh:
            orig_file = cloud['orig_file']+'.fits'
            asgn_file = cloud['orig_file']+'_fasgn.fits'
            higal_file = orig_file.replace('cohrs','higal_xmatch')
            if os.path.isfile(datadir+'COHRS/'+orig_file) and \
                    os.path.isfile(datadir+'HIGAL_MATCHED/'+higal_file):
                if current_open_file != datadir+'COHRS/'+orig_file:
                    co = SpectralCube.read(datadir+'COHRS/'+orig_file)
                    irfull= fits.open(datadir+'HIGAL_MATCHED/'+higal_file)
                    irlong = fits.open(datadir+'HIGAL_MATCHED2/'+higal_file)
                    irmap = (irfull[0].data-irlong[0].data)
                    irmap2 = irfull[0].data
                    asgn = SpectralCube.read(datadir+'ASSIGNMENTS/'+asgn_file)
                    masked_co = co.with_mask(asgn>0*u.dimensionless_unscaled)
                    moment = masked_co.moment(0)
                    current_open_file = datadir+'COHRS/'+orig_file
                    cat.write('output_catalog.fits',overwrite=True)
                mask = (asgn == cloud['_idx']*u.dimensionless_unscaled)
                cloud_cube = co.with_mask(mask)
                cloud_moment = cloud_cube.moment(0)
                fraction = cloud_moment.value/moment.value
                # I am sitcking a 6e11 in here as the frequency of 
                # the 500 microns
                ir_flux = np.nansum(fraction*irmap)/6e11
                ir_lum = ir_flux * cloud['distance']**2*\
                         3.086e18**2*np.pi*4/3.84e33
                print(cloud['_idx'],ir_flux,ir_lum)
                cloud['ir_flux'] = ir_flux
                cloud['ir_luminosity'] = ir_lum
                ir_flux2 = np.nansum(fraction*irmap2)/6e11
                ir_lum2 = ir_flux2 * cloud['distance']**2*\
                         3.086e18**2*np.pi*4/3.84e33
                cloud['ir_flux_short'] = ir_flux2
                cloud['ir_lum_short'] = ir_lum2
                print(cloud['_idx'],ir_flux,ir_lum,ir_lum2)
    return(cat)

def calc_structure_fcn(catalog='cohrs_ultimatecatalog4p.fits'):
    cat = Table.read(catalog)
    if 'sf_offset' not in cat.keys():
        sf_offset = Column(np.zeros(len(cat))+np.nan,name='sf_offset')
        sf_index = Column(np.zeros(len(cat))+np.nan,name='sf_index')
        sfgood = Column(np.zeros(len(cat))+np.nan,name='sf_ngood')
        cat.add_column(sf_offset)
        cat.add_column(sf_index)
        cat.add_column(sfgood)
        current_open_file = ''
    for cloud in cat:
        orig_file = cloud['orig_file']+'.fits'
        asgn_file = cloud['orig_file']+'_fasgn.fits'
        if os.path.isfile(datadir+'COHRS/'+orig_file):
            if current_open_file != datadir+'COHRS/'+orig_file:
                co = SpectralCube.read(datadir+'COHRS/'+orig_file)
                asgn = SpectralCube.read(datadir+'ASSIGNMENTS/'+asgn_file)
#                masked_co = co.with_mask(asgn>0*u.dimensionless_unscaled)
#                moment = masked_co.moment(0)
                current_open_file = datadir+'COHRS/'+orig_file
                cat.write('output_catalog2.fits',overwrite=True)
            print(cloud['_idx'])
            mask = (asgn == cloud['_idx']*u.dimensionless_unscaled)
            subcube = co.subcube_from_mask(mask)
            if subcube.shape[0] > 15:
                nchan = subcube.shape[0]
                nscale = np.min([nchan/2,10])
                
                r, dv = cloudpca.structure_function(subcube,
                                                    meanCorrection=True,
                                                    nScales=nscale,
                                                    noiseScales=nscale/2)
                idx = np.isfinite(r) * np.isfinite(dv)
                n_good = np.sum(idx)
                p = np.polyfit(np.log10(r[idx])+
                               np.log10(2.91e-5*cloud['distance']),
                               np.log10(dv[idx]),1)
                if doPlot:
                    plt.clf()
                    x = np.log10(r[idx])+\
                        np.log10(2.91e-5*cloud['distance'])
                    plt.plot(x,np.log10(dv[idx]),'ro')
                    plt.plot(x,p[0]*x+p[1],alpha=0.5)
                    plt.show()
                cloud['sf_index']= p[0]
                cloud['sf_offset'] = p[1]
                cloud['sf_ngood'] = n_good
            print(cloud['sf_index'],cloud['sf_offset'])
    return(cat)
