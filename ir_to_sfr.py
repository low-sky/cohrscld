from astropy.table import Table, Column
import astropy.units as u
import astropy.constants as con
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss

cohrsdir = '/mnt/ephem/ephem/erosolow/cohrs/'
t = Table.read('cohrs_catalog_withir.fits')

hdr = fits.getheader('higal_70um.fits')
srpix = ((hdr['CDELT2'] * u.deg)**2).to(u.sr)

#HiGAL 70 um
hdu = fits.open('/mnt/bigdata/erosolow/surveys/higal/'
                + 'HiGAL_70um_masked/l24_blue_img_wgls_masked.fits')
bunit = 1*u.Jy/(hdu[0].header['CDELT2']*u.deg)**2
scalefac = bunit.to(u.MJy/u.sr).value

# This applies the scale factor in the original data
higal70 = (t['flux_70um'] - t['bg_70um']) * scalefac

nu70 = 2.99492758e8 / (70e-6)

sfr70 = (1e1**(-43.23 - 17) # Kennicutt and Evans and MJy defn
         * higal70 * nu70  # nuFnu
         * 4 * np.pi * t['bgps_distance_pc']**2 * 3.086e18**2 # 4 pi d^2
         * srpix.value)

#MIPS 24um (originally in MJy/sr)

nu24 = 2.99492758e8 / (23.68e-6)
mips24 = (t['mips24um'] - t['bg_24um']) 
sfr24 = (1e1**(-42.69 - 17)
         * mips24 * nu24
         * 4 * np.pi * t['bgps_distance_pc']**2 * 3.086e18**2 # 4 pi d^2
         * srpix.value)


# SFR from TIR (originally in erg/s/cm2)
hdrtir = fits.getheader('/mnt/bigdata/erosolow/'
                        + 'higal/70to500/HIGAL_L060_070to500_integral.fits')
srpix_tir = ((hdrtir['CDELT2']*u.deg)**2).to(u.sr)
scalefac =  srpix / srpix_tir

tir = (t['flux_tir'] - t['bg_tir']) * scalefac
sfrtir = (1e1**(-43.41)
          * tir
          * 4 * np.pi * t['bgps_distance_pc']**2 * 3.086e18**2
          * srpix) # 4 pi d^2)

tircol = Column(sfrtir, 'sfr_tir')
mipscol = Column(sfr24, 'sfr_24um')
pacscol = Column(sfr70, 'sfr_70um')

t.add_columns([tircol, mipscol, pacscol])
t.write('cohrs_withir_withsfr.fits', overwrite=True)


