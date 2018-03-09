import numpy as np
import math

class imageGradient():
    def __init__(self, dIx, dIy, dI_norm, dI_orientation):
        self.dIx = dIx
        self.dIy = dIy
        self.dI_norm = dI_norm
        self.dI_orientation = dI_orientation
        

def compute_gradient(I):
    n,m = I.shape
    ## These outputs have the size of I.shape-1 because of borders
    dIx = np.zeros((n-1,m-1))
    dIy = np.zeros((n-1,m-1))
    dI_norm = np.zeros((n-1,m-1))
    dI_orientation = np.zeros((n-1,m-1))
    
    for i in range(1,n-1):
        for j in range(1,m-1):
            dIx[i][j] = I[i][j+1]-I[i][j-1]
            dIy[i][j] = I[i+1][j]-I[i-1][j]
            dI_norm[i][j] = math.sqrt(math.pow(dIx[i][j],2)+math.pow(dIy[i][j],2))
            dI_orientation = math.atan(dIx[i][j]/dIy[i][j])
            
    return imageGradient(dIx, dIy, dI_norm, dI_orientation)