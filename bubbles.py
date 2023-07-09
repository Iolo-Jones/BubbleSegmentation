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

def greyscale(ims):
    greys = []
    for im in ims:
        avg = np.median(im, axis = (0,1))
        im_norm = im / avg
        greys.append(im_norm.sum(axis = -1))
    return np.array(greys)

def threshold_image(im, threshold):
    _, thres_image = cv.threshold(im, threshold, 1, cv.THRESH_BINARY)
    return thres_image

def threshold_image_variable(x, a, b):
    x0 = np.min([(x-a)/(b-a), np.ones(x.shape)], axis = 0)
    return np.max([x0, np.zeros(x.shape)], axis = 0)
    
def circle_density(x, params):
    a,b,c,d = params
    spike = np.min([(x-a)/(b-a), np.ones(x.shape), (d-x)/(d-c)], axis = 0)
    return np.max([spike, np.zeros(x.shape)], axis = 0)

def bubble_kernel(params, N):
    a,b,c,d = params
    a *= N/2
    b *= N/2
    c *= N/2
    d *= N/2
    xx, yy = np.mgrid[:N, :N]
    dist = (xx - (N-1)/2)**2 + (yy - (N-1)/2)**2
    dist = np.sqrt(dist)
    bub = -circle_density(dist, [a,b,c,d])
    bub -= bub.mean()
    return bub / np.linalg.norm(bub)

def normalise_brightness(im, Nid = 100):
    idker = np.ones((Nid,Nid))/Nid**2
    norm_map = cv.filter2D(src=im, ddepth=-1, kernel=idker)
    return im - norm_map

def sharpen_image(im, Nid = 3, t = 0.9):
    sharp_ker = - t*np.ones((Nid,Nid))/Nid**2
    sharp_ker[int((Nid-1)/2), int((Nid-1)/2)] += 1
    sharp_ker /= np.linalg.norm(sharp_ker)
    return cv.filter2D(src=norm, ddepth=-1, kernel=sharp_ker)

def compute_constellation(im, lower_bound = 10, upper_bound = 200, radius_rate = 5):
    constellation = []
    for N in range(lower_bound,upper_bound,radius_rate):
        bubbleker = bubble_kernel([a,b,c,d], N)
        conv = cv.filter2D(src=thres_image, ddepth=-1, kernel=bubbleker)
        conv /= N
        constellation.append(conv)
    return np.array(constellation)

def threshold_constellation(cons, threshold):
    constellation_thres = []
    for layer in cons:
        _, thres = cv.threshold(layer, threshold, 1, cv.THRESH_BINARY)
        constellation_thres.append(thres)
    return np.array(constellation_thres, dtype = np.uint8)

def stack_constellation(cons):
    constellation_stack = constellation_thres.sum(axis = 0)
    return np.array(constellation_stack>0, dtype = np.uint8)

def erode(im, erosion_size = 2):
    eroding_kernel = cv.getStructuringElement(cv.MORPH_RECT, (erosion_size,erosion_size))
    return cv.erode(constellation_stack, eroding_kernel)

def distribution_from_constellation(cons):
    dist = []
    for layer in cons:
        n, _ = cv.connectedComponents(layer, 8)
        dist.append(n)
    dist = np.array(dist)
    return dist/dist.sum()

def compute_bubble_centres(constellation_thres, constellation_stack, radius_rate, lower_bound, verbose = True):
    centres = []
    for k in range(1,number):
        if k % 200 == 0 and verbose:
            print('computing bubble %d of %d' %(k, number))
        mask = (labels == k)
        cluster = np.transpose((constellation_stack * mask).nonzero())
        x, y = np.array(np.round(np.median(cluster, axis = 0)), dtype = int)
        # rad = cluster[:,0]
        # r, y, x = np.median(cluster[rad == rad[-1]], axis = 0)
        try:
            r = constellation_thres[:, x, y].nonzero()[0].mean()
            centres.append([x, y, (radius_rate*r + lower_bound)*c/2])
        except:
            pass
    if verbose:
        print('Done')
    return np.array(centres)

def plot_bubbles_on_image(im, centres):
    fig, ax = plt.subplots(figsize = (10,10))
    ax.imshow(im)
    for centre in centres:
        y, x, r = centre
        circle = plt.Circle((x, y), r, color='r', alpha = 0.5, fill=False)
        ax.add_patch(circle)