#!/usr/bin/env python

import numpy as np
import math
import cv2

nBits = 4
nWords = 10
threshold = 0

def threshold_mat(mat):

  height, width, channels = mat.shape
  lbp = {}

  # assume 0 value
  if threshold == 0:
      
    #need to make sure the matrix is padded at the ends
    if height % 3 != 0:
      height = ((height / 3) + 1) * height
    
    if width % 3 != 0:
      width = ((width / 3) + 1) * width

    padded_img = np.zeros((height, width, channels), dtype=np.uint8)

    padded_img[0:height, 0:width] = mat

    
    y = 0
    while y < height:
      x = 0

      while x < width:
        x = x + 1

        #window is 3x3 starting at x,y offset, which go up/down by 3...
        #assume everyting filled with 0s on the edges

        # compute 3x3 threshold matrix

        # multiple times 256 color matrix

        # compute lbp for this block...

      y = y + 1
  else:
    # threshold is the center of the 3x3 (8) 'hood

    return

def build_window(img, magnitudes, img_h, img_w):
    height, width = img.shape

    channels = 3
    #print(height, width, channels)
    #print(len(magnitudes))

    window = np.zeros((img_h * height, img_w * width), dtype=np.uint8)

    x = 0
    while x < img_h:
        y = 0

        while y < img_w:
            offset_magnitudes = x * img_h + y
            offset_win_x = x * height
            offset_win_y = y * width

            print(offset_magnitudes, offset_win_x, offset_win_y)
            
            window[offset_win_x:offset_win_x + height,
                   offset_win_y:offset_win_y + width] = magnitudes[offset_magnitudes]
            y = y + 1

        x = x + 1

    return window

def get_magnitude_images(img, gabor_images):
    magnitude_images = []
    for gabor_img in gabor_images:
        gmi = np.zeros_like(img)
        mgi = np.sqrt(np.add(np.exp2(gabor_img.real), np.exp2(gabor_img.imag)))
        magnitude_images.append(mgi)
    
    return magnitude_images

def build_kernel_window(img_h, img_w, ksize, kernels):

    if kernels and len(kernels) > 0 and img_h > 0 and img_w > 0 and (img_h*img_w<=len(kernels)):

        height, width = ksize+1, ksize+1
        window = np.zeros((height*img_h, width*img_w), dtype=np.uint8)

        x = 0
        while x < img_h:
            y = 0

            while y < img_w:
                offset_kernels = x * img_w + y
                offset_win_x = x * height
                offset_win_y = y * width

                window[offset_win_x:offset_win_x + height,
                    offset_win_y:offset_win_y + width] = kernels[offset_kernels]

                y = y + 1

            x = x + 1

        return window

    else:
        print('No kernels were detected')
        return None

def gabor_fn(sigma, theta, Lambda, psi, gamma):
    sigma_x = sigma
    sigma_y = float(sigma) / gamma

    # Bounding box
    nstds = 3 # Number of standard deviation sigma
    xmax = max(abs(nstds * sigma_x * np.cos(theta)), abs(nstds * sigma_y * np.sin(theta)))
    xmax = np.ceil(max(1, xmax))
    ymax = max(abs(nstds * sigma_x * np.sin(theta)), abs(nstds * sigma_y * np.cos(theta)))
    ymax = np.ceil(max(1, ymax))
    xmin = -xmax
    ymin = -ymax
    (y, x) = np.meshgrid(np.arange(ymin, ymax + 1), np.arange(xmin, xmax + 1))

    # Rotation 
    x_theta = x * np.cos(theta) + y * np.sin(theta)
    y_theta = -x * np.sin(theta) + y * np.cos(theta)

    gb = np.exp(-.5 * (x_theta ** 2 / sigma_x ** 2 + y_theta ** 2 / sigma_y ** 2)) * np.cos(2 * np.pi / Lambda * x_theta + psi)
    return gb

def gaborFilterBank(u,v,m,n):

    #gaborArray = np.zeros((u, v), dtype=complex)
    fmax = 0.25
    gama = math.sqrt(2)
    eta = math.sqrt(2)
    pi = np.pi
    kernels = []

    for i in range(0, u):
        fu = fmax/((math.sqrt(2))**(i));
        alpha = fu/gama;
        beta = fu/eta;
        
        for j in range(0, v):

            tetav = ((j)/v)*pi;
            gFilter = np.zeros((m, n), dtype=complex)
            
            for x in range(0, m):
                for y in range(0, n):

                    xprime = (x-((m+1)/2))*np.cos(tetav)+(y-((n+1)/2))*np.sin(tetav);
                    yprime = -(x-((m+1)/2))*np.sin(tetav)+(y-((n+1)/2))*np.cos(tetav);
                    #print ( (fu**2/(pi*gama*eta))*np.exp(-((alpha**2)*(xprime**2)+(beta**2)*(yprime**2)))*np.exp(1j*2*pi*fu*xprime))
                    #print (gFilter[x, y])
                    gFilter[x, y] = (fu**2/(pi*gama*eta))*np.exp(-((alpha**2)*(xprime**2)+(beta**2)*(yprime**2)))*np.exp(1j*2*pi*fu*xprime)

            #gaborArray[i, j] = gFilter
            kernels.append(gFilter)
            
    return kernels #gaborArray

def build_filters(ksize):
    filter_list = []
    #thetas = v * np.pi  / 8

    # freq max is usually set to 0.25 per Struc and Pavesic
    f_max = 0.25
    #f_u = f_max  /  2 ** (u/2)

    #thetas = [22.5, 45.0, 67.5, 90.0, 112.5, 135.0, 157.5, 180.0]
    lambdas = [5.0, 10.0, 15.0, 20.0, 25.0]
    gamma = 1 #0.02 #math.sqrt(2) #1 #math.sqrt(2)
    sigma = math.sqrt(2) #4 #5
    psi = 0
    
    # eta = n = sqrt(2)

    # λ \lambda represents the wavelength of the sinusoidal factor 
    # θ \theta represents the orientation of the normal to the parallel stripes of a Gabor function
    # ψ \psi is the phase offset
    # σ = n \sigma is the sigma/standard deviation of the Gaussian envelope
    # γ \gamma is the spatial aspect ratio, and specifies the ellipticity of the support of the Gabor function.

    #sigma = gamma = math.sqrt(2)
    # λ = pi / f_u

    thetas = []
    lambdas = []

    for v in [1, 2, 3, 4, 5, 6, 7, 8]:
        thetas.append(float(v * math.pi / 8))

    for u in [0, 1, 2, 3, 4]:
        f_u = f_max  /  2 ** float(u/2)
        lambd = math.pi / float(180/f_u)
        print(lambd)
        lambdas.append(lambd)

    # generate 40 kernels
    for theta in thetas:
        for lambd in lambdas:
            kern = cv2.getGaborKernel(
                (ksize, ksize), sigma, theta, lambd, gamma, psi, ktype=cv2.CV_32F)
            filter_list.append(kern)

            """
            cv2.imshow('kern_%s_%s' % (str(theta), str(lambd)), cv2.resize(kern, (ksize*10, ksize*10)) )
            cv2.waitKey(0)
            """

    return filter_list

def process(img, filters):
    magnitudes = []
    accum = np.zeros_like(img)
    for kern in filters:
        fimg = cv2.filter2D(img, cv2.CV_8UC1, kern)
        magnitudes.append(fimg)
        np.maximum(accum, fimg, accum)
    return accum, magnitudes


if __name__ == '__main__':
    import sys

    try:
        img_fn = sys.argv[1]
    except:
        img_fn = 'examples/lady.png'

    img = cv2.imread(img_fn)
    if img is None:
        print('Failed to load image file:', img_fn)
        sys.exit(1)

    filters = build_filters(8)
    #filters = gaborFilterBank(5,8,39,39)

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    res1, filtered_images = process(gray_img, filters)

    res1 = build_window(gray_img, filtered_images, 4, 10)
    cv2.imshow('filtered_images', res1)

    magnitudes = get_magnitude_images(gray_img, filtered_images);
    res1 = build_window(gray_img, magnitudes, 4, 10)

    #gaborArray = gaborFilterBank(5,8,39,39)

   # print(gaborArray)

    cv2.imshow('result', res1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    """
    res1 = build_kernel_window(8, 5, 9, filters)
    cv2.imshow('kernels', res1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    """