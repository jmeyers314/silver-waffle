import warnings
import numpy as np
import galsim

import matplotlib.pyplot as plt
# from matplotlib.figure import Figure
# from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib.animation as anim
from astropy.utils.console import ProgressBar


def makePlot(galMom, psfMom, arrays):
    # Make an ellipse with the right second moments...
    gT = galMom['xx'] + galMom['yy']
    ge1 = (galMom['xx'] - galMom['yy'])/gT
    ge2 = 2*galMom['xy']/gT
    pT = psfMom['xx'] + psfMom['yy']
    pe1 = (psfMom['xx'] - psfMom['yy'])/pT
    pe2 = 2*psfMom['xy']/pT
    cT = gT+pT
    ce1 = ((psfMom['xx'] + galMom['xx']) - (psfMom['yy'] + galMom['yy'])) / cT
    ce2 = 2*(psfMom['xy'] + galMom['xy']) / cT

    psfImg = galsim.Gaussian(fwhm=np.sqrt(pT)).shear(e1=pe1, e2=pe2).drawImage(nx=96, ny=96, scale=0.1)
    galImg = galsim.Gaussian(fwhm=np.sqrt(gT)).shear(e1=ge1, e2=ge2).drawImage(nx=96, ny=96, scale=0.1)
    convImg = galsim.Gaussian(fwhm=np.sqrt(cT)).shear(e1=ce1, e2=ce2).drawImage(nx=96, ny=96, scale=0.1)

    arrays[0].set_array(convImg.array/np.max(convImg.array))
    arrays[1].set_array(psfImg.array/np.max(psfImg.array))
    arrays[2].set_array(galImg.array/np.max(galImg.array))


def make_movie(args):
    # Code to setup the Matplotlib animation.
    metadata = dict(title='PSF Error Movie', artist='Matplotlib')
    writer = anim.FFMpegWriter(fps=15, bitrate=50000, metadata=metadata)

    # For the animation code, we essentially draw a single figure first, and then use various
    # `set_XYZ` methods to update each successive frame.
    fig, axes = plt.subplots(nrows=1, ncols=3, facecolor='k', figsize=(12, 4))
    # FigureCanvasAgg(fig)

    arrays = []
    for ax in axes:
        arrays.append(ax.imshow(
            np.ones((128, 128), dtype=np.float64),
            animated=True,
            vmin=0.0,
            vmax=1.0,
        ))

    for ax in axes:
        for _, spine in ax.spines.items():
            spine.set_color('k')
        ax.title.set_color('k')
        ax.xaxis.label.set_color('k')
        ax.yaxis.label.set_color('k')
        ax.tick_params(axis='both', colors='k')

    convMom = dict(xx=3, yy=4, xy=2)

    with ProgressBar(2*args.nstep) as bar:
        with writer.saving(fig, "PSF_size_error.mp4", 100):
            for t in range(args.nstep):

                T_PSF = (np.sin(2*np.pi*t/args.nstep)*0.5+0.5)*(args.Tmax-args.Tmin)+args.Tmin
                psfMom = dict(xx=T_PSF/2, yy=T_PSF/2, xy=0)
                galMom = dict(xx=convMom['xx']-psfMom['xx'], yy=convMom['yy']-psfMom['yy'], xy=convMom['xy']-psfMom['xy'])

                makePlot(galMom, psfMom, arrays)

                bar.update()
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    writer.grab_frame(facecolor=fig.get_facecolor())

        with writer.saving(fig, "PSF_ellip_error.mp4", 100):
            for t in range(args.nstep):
                e_PSF = np.sin(2*np.pi*t/args.nstep)*args.emax
                T_PSF = 2.0
                psfMom = dict(xx=T_PSF*(1+e_PSF)/2, yy=T_PSF*(1-e_PSF)/2, xy=0)
                galMom = dict(xx=convMom['xx']-psfMom['xx'], yy=convMom['yy']-psfMom['yy'], xy=convMom['xy']-psfMom['xy'])

                makePlot(galMom, psfMom, arrays)

                bar.update()
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    writer.grab_frame(facecolor=fig.get_facecolor())


if __name__ == '__main__':
    from argparse import ArgumentParser, RawDescriptionHelpFormatter
    parser = ArgumentParser()

    parser.add_argument("--emax", type=float, default=0.8)
    parser.add_argument("--Tmax", type=float, default=2.5)
    parser.add_argument("--Tmin", type=float, default=0.5)
    parser.add_argument("--nstep", type=int, default=100)

    args = parser.parse_args()
    make_movie(args)


# Some extra tricks to convert to gifs:
# ffmpeg -i PSF_size_error.mp4 -filter_complex "[0:v] palettegen" pallete.png
# ffmpeg -i PSF_size_error.mp4 -i pallete.png -filter_complex "[0:v][1:v] paletteuse" PSF_size_error.gif
# ffmpeg -i PSF_ellip_error.mp4 -i pallete.png -filter_complex "[0:v][1:v] paletteuse" PSF_ellip_error.gif
