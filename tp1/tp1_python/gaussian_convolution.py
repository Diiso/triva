from scipy import signal
import gaussian_filter_2d as gf2d

def gaussianConvolution(I,sigma):
    G = gf2d.gaussianFilter2d(sigma)
    return signal.convolve2d(I,G,mode="same",boundary="symm")
    ## mode is the format of the returned array
    ## boundary is how you treat boundaries
