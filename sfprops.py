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
import skimage.morphology as skm
from scipy.optimize import least_squares as lsq
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

def myplane(p,x,y,z):
    return(p[0]+p[1]*x+p[2]*y+p[3]*x*y-z)

def myline(p,x,y):
    return(p[0]*x+p[1]-y)

def calc_irlum(catalog = 'cohrs_ultimatecatalog5.fits', refresh=False):
    cat = Table.read(catalog)
    current_open_file = ''
    if 'ir_luminosity' not in cat.keys():
        keylist = ['ir_luminosity','ir_flux','ir_flux_short',
                'ir_lum_short','bg_flux','bg_lum']
        for thiskey in keylist:
            c = Column(np.zeros(len(cat))+np.nan,name=thiskey)
            cat.add_column(c)

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
                fraction = (cloud_moment.value/moment.value)
                planemask = skm.binary_closing(fraction > 0,selem=skm.disk(3))
                fraction = np.nanmean(fraction)
                rind = (skm.binary_dilation(planemask,selem=skm.disk(6))-\
                        skm.binary_dilation(planemask,selem=skm.disk(3)))*\
                    np.isfinite(irmap2)
                if np.any(rind):
                    rindvals = irmap2[rind]
                    clipval = 4*np.percentile(rindvals,15.87)-\
                              3*(np.percentile(rindvals,2.28))
                    rind *= irmap2 <= clipval
                    yv,xv = np.where(rind)
                    x0,y0 = np.median(xv),np.median(yv)
                    dataz = np.c_[np.ones(xv.size), 
                                  xv-x0, 
                                  yv-y0]
                    try:

                        lsqcoeffs,_,_,_ = np.linalg.lstsq(dataz,irmap2[rind])
                        outputs = lsq(myplane,np.r_[lsqcoeffs,0],
                                     args=(xv-x0,
                                           yv-y0,
                                           irmap2[rind]),
                                     loss = 'soft_l1')
                        coeffs = outputs.x
                        yhit,xhit = np.where(planemask)
                        bg = coeffs[0]+coeffs[1]*(xhit-x0)+\
                             coeffs[2]*(yhit-y0)+coeffs[3]*(yhit-y0)*(xhit-x0)

                        # I am sitcking a 6e11 in here as the frequency of 
                        # the 500 microns
                        bgavg = np.sum(fraction*bg)/6e11
                        bglum = bgavg*cloud['distance']**2*\
                                3.086e18**2*np.pi*4/3.84e33

                    except ValueError:
                        pass

                    ir_flux = np.nansum(fraction*(irmap[planemask]))/6e11
                    ir_lum = ir_flux * cloud['distance']**2*\
                             3.086e18**2*np.pi*4/3.84e33
                    ir_flux2 = np.nansum(fraction*irmap2[planemask])/6e11
                    ir_lum2 = ir_flux2 * cloud['distance']**2*\
                             3.086e18**2*np.pi*4/3.84e33
                    cloud['ir_flux'] = ir_flux2
                    cloud['ir_luminosity'] = ir_lum2
                    cloud['ir_flux_short'] = ir_flux
                    cloud['ir_lum_short'] = ir_lum
                    cloud['bg_flux'] = bgavg
                    cloud['bg_lum'] = bglum
                    print(cloud['_idx'],ir_flux,ir_lum,ir_lum2,bglum)
    #                if cloud['volume_pc2_kms']>1e2:
    #                    import pdb; pdb.set_trace()

    return(cat)

def calc_structure_fcn(catalog='cohrs_ultimatecatalog4p.fits',bootiter=0,doPlot=False):
    cat = Table.read(catalog)
    if 'sf_offset' not in cat.keys():
        keylist = ['sf_offset','sf_index','sf_ngood',
                   'sf_index_err','sf_offset_err']
        for thiskey in keylist:
            c = Column(np.zeros(len(cat))+np.nan,name=thiskey)
            cat.add_column(c)
        
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
            try:
                if subcube.shape[0] > 15:
                    nchan = subcube.shape[0]
                    nscale = np.min([nchan/2,10])

                    r, dv = cloudpca.structure_function(subcube,
                                                        meanCorrection=True,
                                                        nScales=nscale,
                                                        noiseScales=nscale/2)
                    idx = np.isfinite(r) * np.isfinite(dv)
                    n_good = np.sum(idx)
                    if n_good >3:
                        p = np.polyfit(np.log10(r[idx])+
                                       np.log10(2.91e-5*cloud['distance']),
                                       np.log10(dv[idx]),1)

                        pboot = np.zeros((2,bootiter))
                        if bootiter>0:
                            indices= (np.where(idx))[0]
                            length = len(indices)
                            for ctr in np.arange(bootiter):
                                bootidx = np.random.choice(indices,length,True)
                                pboot[:,ctr] = np.polyfit(np.log10(r[bootidx])+
                                                          np.log10(2.91e-5*
                                                                   cloud['distance']),
                                                          np.log10(dv[bootidx]),1)


                            cloud['sf_index_err']=0.5*(\
                                                   np.percentile(pboot[0,:],84.13)-\
                                                   np.percentile(pboot[0,:],15.87))
                            cloud['sf_offset_err']=0.5*(\
                                                   np.percentile(pboot[1,:],84.13)-\
                                                   np.percentile(pboot[1,:],15.87))
                        if doPlot:
                            plt.clf()
                            x = np.log10(r[idx])+\
                                np.log10(2.91e-5*cloud['distance'])
                            plt.plot(x,np.log10(dv[idx]),'ro')
                            plt.plot(x,p[0]*x+p[1],alpha=0.5)
                            plt.plot(x,probust.x[0]*x+probust.x[1],
                                     alpha=0.5,linestyle='--')
                            plt.show()
                        cloud['sf_index']= p[0]
                        cloud['sf_offset'] = p[1]
                        cloud['sf_ngood'] = n_good
                    print('{0} +/- {1}'.format(cloud['sf_index'],
                                               cloud['sf_index_err']),
                          cloud['sf_offset'])
            except:
                pass
    return(cat)
