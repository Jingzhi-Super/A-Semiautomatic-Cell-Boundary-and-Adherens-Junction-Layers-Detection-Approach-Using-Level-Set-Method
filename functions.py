# -*- coding: utf-8 -*-
"""
Help functions
@author: jingzhiw
"""

import numpy as np
import scipy.signal as signal
import scipy.ndimage
import matplotlib.pyplot as plt


# gaussian kernel
def Gauss_kernel(sigma=1):
    h1 = 5
    h2 = 5
    x, y = np.mgrid[0:h1, 0:h2]
    x = x-h1/2
    y = y-h2/2
    g = np.exp(-(x**2 + y**2) / (2*sigma**2))
    return g / g.sum()

    
# divergence    
def div(sx, sy):
    nx, junk = np.gradient(sx)
    junk, ny = np.gradient(sy)
    return nx+ny


# zeros crossing
def zerocrossing(contour):
    out = list()
    for i in range(np.shape(contour)[0]-1):
        for j in range(np.shape(contour)[1]-1):
            if (contour[i, j] > 0) and ((contour[i-1, j]*contour[i+1, j] < 0) or (contour[i, j-1]*contour[i, j+1] < 0)):
                out.append([i, j])
    out = np.asarray(out)
    return out


# delete duplicates from list
def unique(iner):
    output = list()
    for i in iner:
        if i not in output:
            output.append(i)   
    return output


# define pixel intensity threshold based on cut-off threshold or ratio threshold
def thresholding(pic, expert, value):
    if expert == 0:
        threshold = value
    elif expert == 1:
        img = pic.flatten()
        img = np.sort(img)
        threshold = img[int(len(img)*value)]
    return threshold


# decide whether use y=kx+b or x=ky+b by performing least square
def lineardecision(x, y):
    p1 = np.polyfit(x, y, 1)
    _y = np.polyval(p1, x)
    error1 = np.sum((y-_y)**2)
    p2 = np.polyfit(y, x, 1)
    _x = np.polyval(p2, y)
    error2 = np.sum((x-_x)**2)
    if error1 <= error2:
        flag = 1
        return [flag, p1]
    else:
        flag = 2
        return [flag, p2]


# level-set method function
def levelset(dt, I, box, max_iter, flag):
    c0 = 5*flag
    row, col = I.shape
    phi = c0 * np.ones((row, col))
    phi[box > 0] = -c0
    area_record = list()
    phi_record = list()

    # calculate edge indicator g
    G = Gauss_kernel()
    conv = signal.convolve(I, G, mode='same')
    Ix, Iy = np.gradient(conv)
    g = 1 / (1 + Ix ** 2 + Iy ** 2)
    gx, gy = np.gradient(g)

    # level-set evolution
    for i in range(max_iter):
        # gradient of phi
        gradphix, gradphiy = np.gradient(phi)
        # magnitude of gradient of phi
        absgradphi = np.sqrt(gradphix ** 2 + gradphiy ** 2)

        part1 = g * div(gradphix, gradphiy)
        part2 = gx * gradphix + gy * gradphiy
        part3 = g * absgradphi
        L = part1 + part2 + part3
        phi = phi + dt * L
        if (i % 10 == 0):
            area_record.append(np.sum(phi > 0))
            phi_record.append(phi)
            # plt.imshow(I, cmap='gray')
            # plt.hold(True)
            # cros = plt.contour(phi, 0, colors='r', linewidths=2)
            # plt.hold(False)
            # plt.pause(0.1)

    area_record = np.asarray(area_record)
    da = (area_record[1:] - area_record[:-1]) / area_record[:-1]
    ind = np.max(np.argpartition(da, 5)[:5])
    phi = phi_record[ind]
    phi[phi >= 0] = 0
    phi[phi < 0] = -1
    label, num_features = scipy.ndimage.label(phi)
    result = list()
    for i in range(1, num_features + 1):
        temp = np.sum(label == i)
        result.append(temp)

    n = np.argmax(result) + 1
    label[label == n] = -1
    label[label >= 0] = 1

    return label


    

    

    
    


    
    
    
    
    
    


