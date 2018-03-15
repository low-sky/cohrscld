from astropy.table import Table,Column
from astropy.utils.console import ProgressBar
import montage_wrapper as montage
import matplotlib.pyplot as plt
import numpy as np
from spectral_cube import SpectralCube
import astropy.io.fits as fits
import astropy.wcs as wcs
import astropy.units as u
import os, subprocess
# from turbustat.statistics import pca
import skimage.morphology as skm
from scipy.optimize import least_squares as lsq
# from memory_profiler import profile
from radio_beam import Beam
import glob
#datadir = '/Users/erik/astro/cohrs/'
#outdir = '/Users/erik/astro/cohrs/RESULTS/'

datadir = '/mnt/ephem/ephem/erosolow/cohrs/'
outdir = '/mnt/bigdata/erosolow/cohrs/results/'


def myplane(p,x,y,z):
    return(p[0]+p[1]*x+p[2]*y+p[3]*x*y-z)

def myline(p,x,y):
    return(p[0]*x+p[1]-y)

def sparse_mask(obj, asgn, previous_file='', fill_data=None):
    if obj['orig_file'] != previous_file:
        print "Pulling subcube for {0}".format(obj['orig_file'])
        ofile = obj['orig_file']
        xstart = ofile.split('_')[2]
        xend = ofile.split('_')[3]
        thisfile=glob.glob('ASGN/*_{0}_{1}*fits'.format(xstart, xend))
        if not thisfile:
            subcube = asgn[:, :, int(xstart):int(xend)]
            subcube.write('ASGN/{0}_fasgn.fits'.format(obj['orig_file']),
                                                       overwrite=True)
        else:
            subcube = SpectralCube.read(thisfile[0])
        fill_data = (subcube.filled_data[:].value).astype(np.int)
        previous_file = obj['orig_file']
    mask = (fill_data == obj['cloud_id'])
    zcld, ycld, xcld = np.where(mask)
    return previous_file, fill_data, zcld, ycld, xcld

def line_flux2(catalog, line_name='13co10',
               asgn=datadir + 'COHRS_all_asgn.fits',
               cubefile=datadir + 'GRS_13CO_all.fits'):

    flux = Column(np.zeros(len(catalog)),name=line_name)

    asgn = SpectralCube.read(asgn)
    linefile = SpectralCube.read(cubefile)

    previous_file=''
    fill_data=None
    previous_cube_file=''
    for idx, obj in enumerate(catalog):
        if obj['orig_file'] != previous_cube_file:
            print "Pulling line subcube for {0}".format(obj['orig_file'])
            subx1 = obj['orig_file'].split('_')[2]
            subx2 = obj['orig_file'].split('_')[3]
            subcube = linefile[:, :, int(subx1):int(subx2)]
            fill_cube_data = (subcube.filled_data[:].value)
            previous_cube_file = obj['orig_file']
        
        outtuple = sparse_mask(obj, asgn,
                               previous_file=previous_file,
                               fill_data=fill_data)
        previous_file, fill_data, zcld, ycld, xcld = outtuple
        if len(xcld)>0:
            flux[idx] = np.nansum(fill_cube_data[zcld, ycld, xcld])
    catalog.add_column(flux)
    return catalog

def background(image, mask, fraction):
    rind =  (((skm.binary_dilation(mask,selem=skm.disk(6)))
              ^ skm.binary_dilation(mask,selem=skm.disk(3)))
             * np.isfinite(image))
    
    if np.any(rind):
        rindvals = image[rind]
        clipval = 4*np.percentile(rindvals,15.87)-\
                  3*(np.percentile(rindvals,2.28))
        rind *= image <= clipval
        yv,xv = np.where(rind)
        x0,y0 = np.median(xv),np.median(yv)
        dataz = np.c_[np.ones(xv.size), 
                      xv-x0, 
                      yv-y0]
        try:
            lsqcoeffs, _, _, _ = np.linalg.lstsq(dataz,image[rind])
            outputs = lsq(myplane, np.r_[lsqcoeffs,0],
                        args=(xv-x0,
                              yv-y0,
                              image[rind]),
                          loss = 'soft_l1')
            coeffs = outputs.x
            yhit, xhit = np.where(mask)
            bg = (coeffs[0]
                  + coeffs[1] * (xhit-x0)
                  + coeffs[2] * (yhit-y0)
                  + coeffs[3] * (yhit-y0) * (xhit-x0))


            # fracvals = 1 - np.nan_to_num(fraction[yhit, xhit])
            bgavg = np.nansum(bg) # * fraction[yhit, xhit])
            # import pdb; pdb.set_trace()
            return(bgavg)
        except ValueError:
            return(np.nan)

def line_flux(catalog,
              asgn=datadir + 'COHRS_all_asgn.fits'):

    thco10 = Column(np.zeros(len(catalog)), name='13co10')
    thco32 = Column(np.zeros(len(catalog)), name='13co32')
    c18o32 = Column(np.zeros(len(catalog)), name='c18o32')
    twco32 = Column(np.zeros(len(catalog)), name='12co32')
    asgn = SpectralCube.read(asgn)

    previous_file=''
    fill_data=None
    previous_file=''
    
    for idx, obj in enumerate(ProgressBar(catalog)):
        outtuple = sparse_mask(obj, asgn,
                               previous_file=previous_file,
                               fill_data=fill_data)
        
        if obj['orig_file'] != previous_file:
            print "Pulling img tiles for {0}".format(obj['orig_file'])
            subx1 = obj['orig_file'].split('_')[2]
            subx2 = obj['orig_file'].split('_')[3]
            co32cube = (SpectralCube.read(
            './COHRS_tiles/COHRS_{0}_{1}.fits'.format(
                subx1, subx2))).filled_data[:].value
            thco32cube = (SpectralCube.read(
            './CHIPS_13CO_tiles/CHIMPS_13CO_{0}_{1}.fits'.format(
                subx1, subx2))).filled_data[:].value
            c18o32cube = (SpectralCube.read(
            './CHIPS_C18O_tiles/CHIMPS_C18O_{0}_{1}.fits'.format(
                subx1, subx2))).filled_data[:].value
            grscube = (SpectralCube.read(
                './GRS_tiles/GRS_13CO_{0}_{1}.fits'.format(
                    subx1, subx2))).filled_data[:].value

        previous_file, fill_data, z, y, x = outtuple
        thco10[idx] = np.nansum(grscube[z, y, x])
        thco32[idx] = np.nansum(thco32cube[z, y, x])
        c18o32[idx] = np.nansum(c18o32cube[z, y, x])
        twco32[idx] = np.nansum(co32cube[z, y, x])
    catalog.add_columns([thco10, thco32, c18o32, twco32])
    return catalog
    
        

def img_flux(catalog, 
             asgn=datadir + 'COHRS_all_asgn.fits'):

    flux_mips = Column(np.zeros(len(catalog)), name='mips24um')
    flux_tir = Column(np.zeros(len(catalog)), name='flux_tir')
    flux_fir = Column(np.zeros(len(catalog)), name='flux_fir')
    flux_70um = Column(np.zeros(len(catalog)), name='flux_70um')

    flux_frac_mips = Column(np.zeros(len(catalog)), name='mips24um_frac')
    flux_frac_tir = Column(np.zeros(len(catalog)), name='flux_tir_frac')
    flux_frac_fir = Column(np.zeros(len(catalog)), name='flux_fir_frac')
    flux_frac_70um = Column(np.zeros(len(catalog)), name='flux_70um_frac')

    bg_mips = Column(np.zeros(len(catalog)), name='bg_24um')
    bg_tir = Column(np.zeros(len(catalog)), name='bg_tir')
    bg_fir = Column(np.zeros(len(catalog)), name='bg_fir')
    bg_70um = Column(np.zeros(len(catalog)), name='bg_70um')
    co_all = SpectralCube.read('COHRS_all.fits')

    asgn = SpectralCube.read(asgn)
    mips = fits.open('mips_24um.fits')
    tir = fits.open('higal_70to500.fits')
    fir = fits.open('higal_160to500.fits')
    higal70 = fits.open('higal_70um.fits')

    previous_file=''
    fill_data=None
    previous_img_file=''
    for idx, obj in enumerate(ProgressBar(catalog)):
        outtuple = sparse_mask(obj, asgn,
                               previous_file=previous_file,
                               fill_data=fill_data)
        previous_file, fill_data, zcld, ycld, xcld = outtuple

        if obj['orig_file'] != previous_img_file:
            print "Pulling img tiles for {0}".format(obj['orig_file'])
            subx1 = obj['orig_file'].split('_')[2]
            subx2 = obj['orig_file'].split('_')[3]
            filelist = glob.glob('./COHRS_tiles/'
                                 + 'COHRS_{0}_{1}.fits'.format(subx1,
                                                               subx2))
            if not filelist:
                subcube = co_all[:, :, int(subx1):int(subx2)]
                subcube.write('COHRS_tiles/' 
                              + 'COHRS_{0}_{1}.fits'.format(subx1,
                                                            subx2),
                              overwrite=True)
                
            co = SpectralCube.read('./COHRS_tiles/COHRS_{0}_{1}.fits'.format(subx1, subx2))
            co_sub = co.filled_data[:].value

            tir_sub = tir[0].data[:, int(subx1):int(subx2)]
            fir_sub = fir[0].data[:, int(subx1):int(subx2)]
            mips_sub = mips[0].data[:, int(subx1):int(subx2)]
            higal70_sub = higal70[0].data[:, int(subx1):int(subx2)]
            
            previous_img_file = obj['orig_file']
            moment = np.zeros_like(mips_sub)
            zall, yall, xall = np.where(fill_data >= 0)
            for z,y,x in zip(zall, yall, xall):
                moment[y, x] += co_sub[z, y, x]
            
        if len(xcld)>0:
            mask = np.zeros_like(tir_sub, dtype=np.bool)
            mask[ycld, xcld] = True
            cloud_moment = np.zeros_like(mips_sub)
            for z,y,x in zip(zcld, ycld, xcld):
                cloud_moment[y, x] += co_sub[z, y, x]

            fraction = (cloud_moment/moment)
            mask = skm.binary_closing(fraction > 0,selem=skm.disk(3))
            
            bg_tir[idx] = (background(tir_sub, mask, fraction))
            bg_fir[idx] = (background(fir_sub, mask, fraction))
            bg_mips[idx] = (background(mips_sub, mask, fraction))
            bg_70um[idx] = (background(higal70_sub, mask, fraction))
            # print bg_tir[idx], bg_fir[idx], bg_mips[idx], bg_70um[idx]


            flux_tir[idx] = np.nansum(tir_sub[mask])
            flux_fir[idx] = np.nansum(fir_sub[mask])
            flux_mips[idx] = np.nansum(mips_sub[mask])
            flux_70um[idx] = np.nansum(higal70_sub[mask])

            flux_frac_tir[idx] = np.nansum(tir_sub[mask] * fraction[mask])
            flux_frac_fir[idx] = np.nansum(fir_sub[mask] * fraction[mask])
            flux_frac_mips[idx] = np.nansum(mips_sub[mask] * fraction[mask])
            flux_frac_70um[idx] = np.nansum(higal70_sub[mask] * fraction[mask])
    catalog.add_columns([flux_mips, flux_tir, flux_fir, flux_70um,
                         flux_frac_mips, flux_frac_tir, flux_frac_fir, flux_frac_70um,
                         bg_mips, bg_tir, bg_fir, bg_70um])
    return catalog


    
def minimum_bbox(catalog, asgn=datadir+'COHRS_all_asgn.fits'):
    """
    This calculates bounding boxes in WCS for clouds given and
    assignment cube and a catalog
    """

    s = SpectralCube.read(asgn)
    cubeshape = s.shape
    lmax = Column(np.zeros(len(catalog)),name='l_max')
    bmax = Column(np.zeros(len(catalog)),name='b_max')
    vmax = Column(np.zeros(len(catalog)),name='v_max')
    lmin = Column(np.zeros(len(catalog)),name='l_min')
    bmin = Column(np.zeros(len(catalog)),name='b_min')
    vmin = Column(np.zeros(len(catalog)),name='v_min')

    xmax = Column(np.zeros(len(catalog),dtype=np.int),name='x_max')
    ymax = Column(np.zeros(len(catalog),dtype=np.int),name='y_max')
    zmax = Column(np.zeros(len(catalog),dtype=np.int),name='z_max')
    xmin = Column(np.zeros(len(catalog),dtype=np.int),name='x_min')
    ymin = Column(np.zeros(len(catalog),dtype=np.int),name='y_min')
    zmin = Column(np.zeros(len(catalog),dtype=np.int),name='z_min')

    previous_file = ''
    for idx, obj in enumerate(catalog):

        if obj['orig_file'] != previous_file:
            print "Pulling subcube for {0}".format(obj['orig_file'])
            subx1 = obj['orig_file'].split('_')[2]
            subx2 = obj['orig_file'].split('_')[3]
            subcube = s[:, :, int(subx1):int(subx2)]
            fill_data = (subcube.filled_data[:].value).astype(np.int)
            previous_file = obj['orig_file']
            
        mask = (fill_data == obj['cloud_id'])
        zcld, ycld, xcld = np.where(mask)
        if len(xcld)>0:
            lmin_elt, bmin_elt, vmin_elt = subcube.wcs.wcs_pix2world(np.min(xcld),
                                                                     np.min(ycld),
                                                                     np.min(zcld), 0)
            lmax_elt, bmax_elt, vmax_elt = subcube.wcs.wcs_pix2world(np.max(xcld),
                                                                     np.max(ycld),
                                                                     np.max(zcld), 0)
            
            xmin_elt, ymin_elt, zmin_elt = s.wcs.wcs_world2pix(lmin_elt, bmin_elt, vmin_elt, 0)
            xmax_elt, ymax_elt, zmax_elt = s.wcs.wcs_world2pix(lmax_elt, bmax_elt, vmax_elt, 0)
        
            lmax[idx] = lmin_elt
            bmax[idx] = bmax_elt
            vmax[idx] = vmax_elt
            lmin[idx] = lmax_elt
            bmin[idx] = bmin_elt
            vmin[idx] = vmin_elt

            xmax[idx] = xmax_elt
            ymax[idx] = ymax_elt
            zmax[idx] = zmax_elt
            xmin[idx] = xmin_elt
            ymin[idx] = ymin_elt
            zmin[idx] = zmin_elt
        
    catalog.add_columns([lmax, bmax, vmax, lmin, bmin, vmin])
    return catalog



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
            hdu = fits.open(datadir+'COHRS/'+cofile,memmap=False)
            w = wcs.WCS(hdu[0].header)
            hdr2 = w.to_header()
            hdr2['BMAJ'] = 15./3600
            hdr2['BMIN'] = 15./3600
            hdr2['BPA'] = 0.
            s = SpectralCube(hdu[0].data,w,header=hdr2)
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



def calc_irlum(catalog = 'cohrs_finalcatalog_clumpflux_medaxis.fits',
               refresh=False):
    ctrr = 0
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
                    hdu = fits.open(datadir+'COHRS/'+orig_file,memmap=False)
                    w = wcs.WCS(hdu[0].header)
                    hdr2 = w.to_header()
                    hdr2['BMAJ'] = 15./3600
                    hdr2['BMIN'] = 15./3600
                    hdr2['BPA'] = 0.
                    co = SpectralCube(hdu[0].data,w,header=hdr2)
                    irfull= fits.open(datadir+'HIGAL_MATCHED/'+higal_file,memmap=False)
                    irlong = fits.open(datadir+'HIGAL_MATCHED2/'+higal_file,memmap=False)
                    irmap = (irfull[0].data-irlong[0].data)
                    irmap2 = irfull[0].data
                    asgn = fits.getdata(datadir+'ASSIGNMENTS/'+asgn_file,memmap=False)
                    masked_co = co.with_mask(asgn>0*u.dimensionless_unscaled)
                    moment = masked_co.moment(0)
                    current_open_file = datadir+'COHRS/'+orig_file
                    cat.write('output_catalog.fits',overwrite=True)
#                mask = np.zeros(co.shape,dtype=np.bool)
#                mask[asgn == cloud['cloud_id']]=True
                cloud_cube = co.with_mask(asgn == cloud['cloud_id'])
                cloud_moment = cloud_cube.moment(0)
                cloud_cube = 0.0
                fraction = (cloud_moment.value/moment.value)
                planemask = skm.binary_closing(fraction > 0,selem=skm.disk(3))
                fraction = np.nanmean(fraction)
                rind = (skm.binary_dilation(planemask,selem=skm.disk(6)) ^\
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
                        import pdb; pdb.set_trace()


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
                    print(cloud['cloud_id'],ir_flux,ir_lum,ir_lum2,bglum)
                    cat.write('output_catalog.fits',overwrite=True)
                    ctrr+=1
                    if ctrr==20:
                        return(cat)
    #                if cloud['volume_pc2_kms']>1e2:
    #                    import pdb; pdb.set_trace()
    
    return(cat)

def calc_structure_fcn(catalog='cohrs_finalcatalog_clumpflux_medaxis.fits',
                       bootiter=0,doPlot=False):
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
                hdu = fits.open(datadir+'COHRS/'+orig_file,memmap=False)
                cat.write('cohrs_structurefunc.fits',overwrite=True)
                w = wcs.WCS(hdu[0].header)
                hdr2 = w.to_header()
                hdr2['BMAJ'] = 15./3600
                hdr2['BMIN'] = 15./3600
                hdr2['BPA'] = 0.
                co = SpectralCube(hdu[0].data,w,header=hdr2)
                hdu = fits.open(datadir+'ASSIGNMENTS/'+asgn_file,memmap=False)
                w = wcs.WCS(hdu[0].header)
                hdr2 = w.to_header()
                hdr2['BMAJ'] = 15./3600
                hdr2['BMIN'] = 15./3600
                hdr2['BPA'] = 0.
                asgn = SpectralCube(hdu[0].data,w,header=hdr2)

#                masked_co = co.with_mask(asgn>0*u.dimensionless_unscaled)
#                moment = masked_co.moment(0)
                current_open_file = datadir+'COHRS/'+orig_file
                cat.write('output_catalog2.fits',overwrite=True)
            print(cloud['cloud_id'])
            mask = (asgn == cloud['cloud_id']*u.dimensionless_unscaled)
            
            subcube = co.subcube_from_mask(mask)
            
            if subcube.shape[0] > 15:
                # zeros = np.zeros_like(subcube) # np.random.randn(*subcube.shape)*0.5
                # concatdata = np.vstack([subcube.filled_data[:],zeros])
                # hdr2 = subcube.wcs.to_header()
                # hdr2['BMAJ'] = 15./3600
                # hdr2['BMIN'] = 15./3600
                # hdr2['BPA'] = 0.
                # newcube = SpectralCube(concatdata,subcube.wcs,header=hdr2)
                pcaobj = pca.PCA(subcube,distance=cloud['distance']*u.pc)

                try:
                    pcaobj.run(min_eigval=0.25,verbose=False,mean_sub=True)
                    cloud['sf_index']= pcaobj.index
                    cloud['sf_offset'] = pcaobj.intercept.value
                    cloud['sf_ngood'] = np.min([np.isfinite(pcaobj.spatial_width).sum(),
                                                np.isfinite(pcaobj.spectral_width).sum()])

                    cloud['sf_index_err'] = (pcaobj.index_error_range[1]-pcaobj.index_error_range[0])*0.5
                    cloud['sf_offset_err'] = ((pcaobj.intercept_error_range[1]-pcaobj.intercept_error_range[0])*0.5).value

                    print('{0} +/- {1}'.format(cloud['sf_index'],
                                               cloud['sf_index_err']),
                          cloud['sf_offset'])
                except:
                    pass
        

                    # r, dv = cloudpca.structure_function(subcube,
                    #                                     meanCorrection=True,
                    #                                     nScales=nscale,
                    #                                     noiseScales=nscale/2)
                    # idx = np.isfinite(r) * np.isfinite(dv)
                    # n_good = np.sum(idx)
                    # if n_good >3:
                    #     p = np.polyfit(np.log10(r[idx])+
                    #                    np.log10(2.91e-5*cloud['distance']),
                    #                    np.log10(dv[idx]),1)

                    #     pboot = np.zeros((2,bootiter))
                    #     if bootiter>0:
                    #         indices= (np.where(idx))[0]
                    #         length = len(indices)
                    #         for ctr in np.arange(bootiter):
                    #             bootidx = np.random.choice(indices,length,True)
                    #             pboot[:,ctr] = np.polyfit(np.log10(r[bootidx])+
                    #                                       np.log10(2.91e-5*
                    #                                                cloud['distance']),
                    #                                       np.log10(dv[bootidx]),1)


                        #     cloud['sf_index_err']=0.5*(\
                        #                            np.percentile(pboot[0,:],84.13)-\
                        #                            np.percentile(pboot[0,:],15.87))
                        #     cloud['sf_offset_err']=0.5*(\
                        #                            np.percentile(pboot[1,:],84.13)-\
                        #                            np.percentile(pboot[1,:],15.87))
                        # if doPlot:
                        #     plt.clf()
                        #     x = np.log10(r[idx])+\
                        #         np.log10(2.91e-5*cloud['distance'])
                        #     plt.plot(x,np.log10(dv[idx]),'ro')
                        #     plt.plot(x,p[0]*x+p[1],alpha=0.5)
                        #     plt.plot(x,probust.x[0]*x+probust.x[1],
                        #              alpha=0.5,linestyle='--')
                        #     plt.show()
#            except:
#                pass
    return(cat)
