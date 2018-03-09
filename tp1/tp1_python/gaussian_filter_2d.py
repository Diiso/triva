import math
import numpy as np

def gaussianFilter2d(sigma):
    int_sigma = math.ceil(sigma)

    x = np.arange(-2*int_sigma,2*int_sigma+1,1.)
    ##print(x)
    gaussian = lambda t: math.exp((-math.pow(t,2))/(2*pow(sigma,2)))
    fx = np.vectorize(gaussian)

    y = np.arange(-2*int_sigma,2*int_sigma+1,1)
    fy = np.vectorize(gaussian)

    return np.dot(fx(x)[:,None],fy(y)[None,:])
    
