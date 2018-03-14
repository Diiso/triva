import numpy as np
import math

def quantifyGradient(orientation):
    n,m = orientation.shape
    quant_orientation = np.zeros((n,m))
    PI = math.pi
    
    for i in range(n):
        for j in range(m):
            if orientation[i,j]>3*PI/8 or orientation[i,j]<=-3*PI/8:
                quant_orientation[i,j] = 1
            if orientation[i,j]<=3*PI/8 and orientation[i,j]>PI/8:
                quant_orientation[i,j] = 2
            if orientation[i,j]>-PI/8 and orientation[i,j]<=PI/8:
                quant_orientation[i,j] = 3
            if orientation[i,j]<=-PI/8 and orientation[i,j]>-3*PI/8:
                quant_orientation[i,j] = 4
    
    return quant_orientation