import numpy as np
import cv2 as cv
from scipy.spatial.distance import pdist, squareform

parameters = {}
parameters['sampling_params'] = [10, 400, 5]
parameters['constellation_threshold'] = 0
parameters['image_threshold'] = -0.04
parameters['erosion_size'] = 2
parameters['bubble_scale_factor'] = 1

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

# def empirical_kernel(im, centres):
    bubbles = []
    for centre in centres:
        x, y, r = centre
        r = np.ceil(r)
        k = 1
        x1 = int(x-r-k)
        x2 = int(x+r+k)
        y1 = int(y-r-k)
        y2 = int(y+r+k)
        if (x1 >= 0) and (x2 <= 1600) and (y1 >= 0) and (y2 <= 1200):
            bubble = im[x1:x2,y1:y2]
            bubbles.append(cv.resize(bubble, [100,100]))

    bubbles = np.array(bubbles)
    mean_bubble = bubbles.mean(axis = 0)

    new_bubble_kernel = np.mean([mean_bubble, 
                                    np.rot90(mean_bubble), 
                                    np.rot90(mean_bubble, 2), 
                                    np.rot90(mean_bubble, 3)], axis = 0)
    new_bubble_kernel = np.mean([new_bubble_kernel, new_bubble_kernel.transpose()], axis = 0)
    new_bubble_kernel -= new_bubble_kernel.mean()
    return new_bubble_kernel / np.linalg.norm(new_bubble_kernel)

def normalise_brightness(im, Nid = 100):
    idker = np.ones((Nid,Nid))/Nid**2
    norm_map = cv.filter2D(src=im, ddepth=-1, kernel=idker)
    return im - norm_map

def sharpen_image(im, Nid = 3, t = 0.9):
    sharp_ker = - t*np.ones((Nid,Nid))/Nid**2
    sharp_ker[int((Nid-1)/2), int((Nid-1)/2)] += 1
    sharp_ker /= np.linalg.norm(sharp_ker)
    return cv.filter2D(src=im, ddepth=-1, kernel=sharp_ker)

def compute_constellation(im, parameters, kernel = None, default_bubble_shape = [0.7,0.85,0.95,1]):
    constellation = []
    lower_bound, upper_bound, radius_rate = parameters['sampling_params']
    for N in range(lower_bound,upper_bound,radius_rate):
        if kernel is None:
            bubbleker = bubble_kernel(default_bubble_shape, N)
        else:
            bubbleker = kernel
        conv = cv.filter2D(src=im, ddepth=-1, kernel=bubbleker)
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
    constellation_stack = cons.sum(axis = 0)
    return np.array(constellation_stack>0, dtype = np.uint8)

def erode(im, erosion_size = 2):
    eroding_kernel = cv.getStructuringElement(cv.MORPH_RECT, (erosion_size,erosion_size))
    return cv.erode(im, eroding_kernel)

# def distribution_from_constellation(cons):
    dist = []
    for layer in cons:
        n, _ = cv.connectedComponents(layer, 8)
        dist.append(n)
    dist = np.array(dist)
    return dist/dist.sum()

def compute_bubble_centres(constellation_thres, constellation_stack, number, labels, parameters, verbose = True):
    centres = []
    lower_bound, _, radius_rate = parameters['sampling_params']
    for k in range(1,number):
        if k % 200 == 0 and verbose:
            print('Computing bubble %d of %d' %(k, number))
        mask = (labels == k)
        cluster = np.transpose((constellation_stack * mask).nonzero())
        x, y = np.array(np.round(np.median(cluster, axis = 0)), dtype = int)
        # rad = cluster[:,0]
        # r, y, x = np.median(cluster[rad == rad[-1]], axis = 0)
        r = constellation_thres[:, x, y].nonzero()[0]
        if r.shape[0] > 0:
            r = r.mean()
            centres.append([x, y, (radius_rate*r + lower_bound)*0.95/2])
    if verbose:
        print('Done')
    centres = np.array(centres)
    return remove_contained_bubbles(centres)

def threshold_constellation(cons, bvf, parameters):
    lower_bound, upper_bound, radius_rate = parameters['sampling_params']
    rads = np.arange(lower_bound, upper_bound, radius_rate)
    threshold = threshold_model(rads,bvf) + parameters['constellation_threshold']
    constellation_thres = []
    for k, layer in enumerate(cons):
        _, thres = cv.threshold(layer, threshold[k], 1, cv.THRESH_BINARY)
        constellation_thres.append(thres)
    return np.array(constellation_thres, dtype = np.uint8)

def remove_contained_bubbles(centres):
    x, y, r = centres.transpose()
    xx = squareform(pdist(x.reshape(-1,1)))**2
    yy = squareform(pdist(y.reshape(-1,1)))**2
    rr = squareform(pdist(r.reshape(-1,1)))**2
    contained = xx + yy < rr
    contained_indices = contained.nonzero()

    smaller_indices = []
    for i in range(contained_indices[0].shape[0]):
        bubble1 = centres[contained_indices[0][i]]
        bubble2 = centres[contained_indices[1][i]]
        smaller = np.argmin([bubble1[2], bubble2[2]])
        index_pairs = [contained_indices[0][i], contained_indices[1][i]]
        smaller_indices.append(index_pairs[smaller])
    smaller_indices = np.unique(smaller_indices)
    return np.delete(centres, smaller_indices, axis = 0)

def norm_sharp_thres(im, parameters):
    norm = normalise_brightness(im)
    sharpened = sharpen_image(norm)
    return threshold_image(sharpened, parameters['image_threshold'])

def poly_through_points(points):
    x, y = points.transpose()
    n = x.shape[0]
    A = x.reshape(-1,1)**np.arange(n)
    coeffs = np.linalg.solve(A, y)

    def poly(x):
        n = coeffs.shape[0]
        return x.reshape(-1,1)**np.arange(n) @ coeffs
    
    return poly

def partition_function(x, a, b):
    x1 = np.max([np.zeros(x.shape), (x-a)/(b-a)], axis = 0)
    return np.min([x1,np.ones(x.shape)], axis = 0)

def threshold_model(X, bvf):
    x = X + 45*bvf
    poly_points = np.array([[0,0.13], 
                            [30,0.16], 
                            [60,0.2],
                            [100,0.2],
                            [140,0.165],
                            [170,0.155], 
                            [210,0.15]
                            ])
    poly = poly_through_points(poly_points)(x)

    part1 = partition_function(x, 15, 25)
    part2 = partition_function(x, 145, 170)

    return np.min([part1 * (1-part2) * poly + (1-part1) * 0.14 + part2 * 0.155, #- 0.01,
                   0.2*np.ones(x.shape)], axis = 0)

def bubbles_from_image(im, parameters, bvf, verbose=True):
    thres_image = norm_sharp_thres(im, parameters)
    constellation = compute_constellation(thres_image, parameters)
    constellation_thres = threshold_constellation(constellation, bvf, parameters)
    constellation_stack = stack_constellation(constellation_thres)
    constellation_stack = erode(constellation_stack, parameters['erosion_size'])
    if verbose:
        print('Finding bubble centres')
    number, labels = cv.connectedComponents(constellation_stack, 8)
    centres = compute_bubble_centres(constellation_thres, 
                                 constellation_stack, 
                                 number, 
                                 labels, 
                                 parameters,
                                 verbose=verbose)
    return centres * np.array([1,1,parameters['bubble_scale_factor']])