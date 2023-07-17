import numpy as np
import matplotlib.image as image
import matplotlib.pyplot as plt
import os
from bubbles import greyscale

def import_photos(directory):
    files = sorted(os.listdir(directory))
    pics = []
    names = []
    for n in files:
        if n.endswith('.jpeg') or n.endswith('.jpg'):
            im = image.imread(directory + '/' + n)
            pics.append(im)
            names.append(n)
    return greyscale(np.array(pics)/255), names

def plot_images(arrays, cmap = 'Greys_r'):
    if type(arrays) == list:
        n = len(arrays)
        fig, ax = plt.subplots(1,n, figsize = (10, 10*n))
        if n > 1:
            for k, array in enumerate(arrays):
                plt.set_cmap(cmap)
                ax[k].axis('off')
                if array.max() > 1:
                    ax[k].imshow(array/array.max())
                else:
                    ax[k].imshow(array)
        else:
            plt.set_cmap(cmap)
            ax.axis('off')
            ax.imshow(arrays[0])
        plt.show()
    else:
        fig, ax = plt.subplots(figsize = (10, 10))
        plt.set_cmap(cmap)
        ax.axis('off')
        if arrays.max() > 1:
            ax.imshow(arrays/arrays.max())
        else:
            ax.imshow(arrays)
        plt.show()

def plot_bubbles_on_image(im, centres, scale_factor=1):
    fig, ax = plt.subplots(figsize = (10,10))
    plt.set_cmap('Greys_r')
    ax.imshow(im)
    for centre in centres:
        y, x, r = centre
        circle = plt.Circle((x, y), r*scale_factor, color='r', alpha = 0.5, fill=False)
        ax.add_patch(circle)

def plot_constellation_threshold(parameters):
    t0, t100 = parameters['constellation_threshold']
    lower_bound, upper_bound, radius_rate = parameters['sampling_params']
    rads = np.arange(lower_bound, upper_bound, radius_rate)
    thres = t0 + (t100 - t0)*rads/100
    plt.plot(rads, thres)
    plt.show()