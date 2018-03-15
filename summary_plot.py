from spectral_cube import SpectralCube
import numpy as np
import matplotlib.pyplot as plt
import aplpy

def summary_plot(filelist):
    for thisfile in filelist:
        s = SpectralCube.read(thisfile)
        outfile = thisfile.replace('.fits','_summary.png')
        mom0 = s.moment0()
        f = aplpy.FITSFigure(mom0.hdu)
        f.show_colorscale()
        f.show_colorbar()
        f.save(outfile)

        
