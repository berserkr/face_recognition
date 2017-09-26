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

def build_window(img, magnitudes):
    height, width, channels = img.shape
    print(height, width, channels)
    print(len(magnitudes))

    window = np.zeros((4 * height, 10 * width, channels), dtype=np.uint8)

    x = 0
    while x < 4:
        y = 0
        while y < 10:
            offset_magnitudes = x * 4 + y
            offset_win_x = x * height
            offset_win_y = y * width

            print(offset_magnitudes, offset_win_x, offset_win_y)
            
            window[offset_win_x:offset_win_x + height,
                   offset_win_y:offset_win_y + width] = magnitudes[offset_magnitudes]
            y = y + 1

        x = x + 1

    return window

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

                    xprime = (x-((m)/2))*np.cos(tetav)+(y-((n)/2))*np.sin(tetav);
                    yprime = -(x-((m)/2))*np.sin(tetav)+(y-((n)/2))*np.cos(tetav);
                    #print ( (fu**2/(pi*gama*eta))*np.exp(-((alpha**2)*(xprime**2)+(beta**2)*(yprime**2)))*np.exp(1j*2*pi*fu*xprime))
                    #print (gFilter[x, y])
                    gFilter[x, y] = (fu**2/(pi*gama*eta))*np.exp(-((alpha**2)*(xprime**2)+(beta**2)*(yprime**2)))*np.exp(1j*2*pi*fu*xprime)

            #gaborArray[i, j] = gFilter
            kernels.append(gFilter)
            
    return kernels #gaborArray

def build_filters():
    filter_list = []
    ksize = 9

    """
     Size ksize = new Size(9, 9);
    double[] theta = new double[]{22.5, 45.0, 67.5, 90.0, 112.5, 135.0, 157.5, 180.0};
    double[] lambd = new double[]{5.0, 10.0, 15.0, 20.0, 25.0};
    double gamma = 1;
    double sigma = 5;

    for(int i = 0; i < lambd.length; i++){
        for(int j = 0; j < theta.length; j++){
            Mat kernel = Imgproc.getGaborKernel(ksize, sigma, theta[j], lambd[i], gamma);
            kernels.add(kernel);
        }
    }  

    """
    #thetas = v * np.pi  / 8
    f_max = 0.25
    #f_u = f_max  /  2 ** (u/2)

    #thetas = [22.5, 45.0, 67.5, 90.0, 112.5, 135.0, 157.5, 180.0]
    #lambdas = [5.0, 10.0, 15.0, 20.0, 25.0]
    gamma = math.sqrt(2)
    sigma = 5
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

    for v in [0, 1, 2, 3, 4, 5, 6, 7]:
        thetas.append(v * math.pi / 8)
    
    for u in [4, 5, 6, 7, 8]:
        f_u = f_max  /  2 ** (u/2)
        lambd = 2 * math.pi / f_u
        print(lambd)
        lambdas.append(u)

    # generate 40 kernels
    for theta in thetas:
        for lambd in lambdas:
          kern = cv2.getGaborKernel(
              (ksize, ksize), sigma, theta, lambd, gamma, psi, ktype=cv2.CV_32F)

          print (kern)

          filter_list.append(kern)

    return filter_list


def process(img, filters):
    magnitudes = []
    accum = np.zeros_like(img)
    for kern in filters:
        fimg = cv2.filter2D(img, cv2.CV_8UC3, kern.real)
        magnitudes.append(fimg)
        np.maximum(accum, fimg, accum)
    return accum, magnitudes


if __name__ == '__main__':
    import sys

    try:
        img_fn = sys.argv[1]
    except:
        img_fn = 'examples/lena-128x128.jpg'

    img = cv2.imread(img_fn)
    if img is None:
        print('Failed to load image file:', img_fn)
        sys.exit(1)

    filters = build_filters()
    #filters = gaborFilterBank(5,8,39,39)

    res1, magnitudes = process(img, filters)
    res1 = build_window(img, magnitudes)

    #gaborArray = gaborFilterBank(5,8,39,39)

   # print(gaborArray)

    cv2.imshow('result', res1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
