import numpy.ma as ma
import numpy as np
import matplotlib.pyplot as plt

def show_field(data, organ="liver"):
    fig, axes = plt.subplots(nrows=2, ncols=2)
    counter = 0
    for ax in axes.flat:
        im = ax.imshow(data[counter], cmap="bone")
        ax.autoscale(True)
        ax.axis("off")
        plt.colorbar(im, ax=ax)
        counter += 1
    plt.savefig(organ + ".svg")

def nice_imshow(ax, data, vmin=None, vmax=None, cmap=None):
    if cmap is None:
        cmap = cm.jet
    if vmin is None:
        vmin = data.min()
    if vmax is None:
        vmax = data.max()

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    im = ax.imshow(data, vmin=vmin, vmax=vmax, interpolation='nearest', cmap=cmap)
    pl.colorbar(im, cax=cax)

def make_mosaic(imgs, nrows, ncols, border=1):
    nimgs = imgs.shape[0]
    imshape = imgs.shape[1:]

    mosaic = ma.masked_all((nrows * imshape[0] + (nrows - 1) * border,
                            ncols * imshape[1] + (ncols - 1) * border),
                           dtype=np.float32)

    paddedh = imshape[0] + border
    paddedw = imshape[1] + border
    for i in xrange(nimgs):
        row = int(np.floor(i / ncols))
        col = i % ncols

        mosaic[row * paddedh:row * paddedh + imshape[0],
        col * paddedw:col * paddedw + imshape[1]] = imgs[i]
    return mosaic