import numpy as np
import matplotlib.image as image
import matplotlib.pyplot as plt
import os
import cv2 as cv

def import_photos_and_background(directory):
    files = sorted(os.listdir(directory))
    pics = []
    for n in files:
        if n.endswith('.jpeg') or n.endswith('.jpg'):
            im = image.imread(directory + '/' + n)
            pics.append(im)
    return np.array(pics)/255

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