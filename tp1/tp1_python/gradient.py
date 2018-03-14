import numpy as np
import math
from scipy import signal

class imageGradient():
    def __init__(self, dIx, dIy, dI_norm, dI_orientation):
        self.dIx = dIx
        self.dIy = dIy
        self.dI_norm = dI_norm
        self.dI_orientation = dI_orientation
        

def computeGradientNoConvolution(I):
    n,m = I.shape
    ## These outputs have the size of I.shape-1 because of borders
    dIx = np.zeros((n-1,m-1))
    dIy = np.zeros((n-1,m-1))
    dI_norm = np.zeros((n-1,m-1))
    dI_orientation = np.zeros((n-1,m-1))
    
    for i in range(1,n-1):
        for j in range(1,m-1):
            dIx[i,j] = I[i+1,j]-I[i-1,j]
            dIy[i,j] = I[i,j+1]-I[i,j-1]
            dI_norm[i,j] = math.sqrt(math.pow(dIx[i,j],2)+math.pow(dIy[i,j],2))
            dI_orientation[i,j] = math.atan(dIy[i,j]/(0.0000001+dIx[i,j]))
            
    return imageGradient(dIx, dIy, dI_norm, dI_orientation)

def computeGradient(I):
    n,m = I.shape
 
    # G is the kernel with which we calculate the convolution
    # Here I used the Prewitt derivative filters
    Gy = np.matrix([[-1,0,1],[-2,0,2],[-1,0,1]])
    Gx = np.matrix([[1,2,1],[0,0,0],[-1,-2,-1]])
    
    dIx = signal.convolve2d(I,Gx,mode="same",boundary="symm")
    dIy = signal.convolve2d(I,Gy,mode="same",boundary="symm")
    
    dI_norm = np.zeros((n-1,m-1))
    dI_orientation = np.zeros((n-1,m-1))
    
    for i in range(1,n-1):
        for j in range(1,m-1):
            dI_norm[i,j] = math.sqrt(math.pow(dIx[i,j],2)+math.pow(dIy[i,j],2))
            dI_orientation[i,j] = math.atan(dIy[i,j]/(0.0000001+dIx[i,j]))
            
    return imageGradient(dIx, dIy, dI_norm, dI_orientation)