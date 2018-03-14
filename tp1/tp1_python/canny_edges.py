import numpy as np
from gaussian_convolution import gaussianConvolution
from gradient import computeGradient
from quantify_gradient import quantifyGradient
from non_max_supression import nms

def cannyEdges(I,sigma,t1,t2):
    #t1>t2
    
    I_blurred = gaussianConvolution(I,sigma)
    I_gradient = computeGradient(I_blurred)
    dIx,dIy,dI_norm,dI_orientation = I_gradient.dIx,I_gradient.dIy,I_gradient.dI_norm,I_gradient.dI_orientation
    quantified_orientation = quantifyGradient(dI_orientation)
    nms_edges_1 = nms(dI_norm, quantified_orientation, t1)
    nms_edges_2 = nms(dI_norm, quantified_orientation, t2)
    
    #Hysteresis
    edges_to_visit = []
    n,m = nms_edges_1.shape
    for i in range(n):
        for j in range(m):
            if nms_edges_1[i,j] == 1:
                edges_to_visit.append((i,j))
                

    while len(edges_to_visit)>0:
        new_edges_to_visit = []
        for k in range(len(edges_to_visit)):
            edge_i,edge_j = edges_to_visit[k]
            
            for i in range(-1,2):
                for j in range(-1,2):
                    if (i!=0) and (j!=0):
                        if (edge_i+i>=0) and (edge_i+i<n) and (edge_j+j>=0) and (edge_j+j<m):
                            pei = edge_i + i
                            pej = edge_j + j
                            if (nms_edges_2[pei,pej] == 1) and (nms_edges_1[pei,pej] == 0):
                                nms_edges_1[pei,pej]=1
                                new_edges_to_visit.append((pei,pej))
        edges_to_visit = new_edges_to_visit

    return nms_edges_1
                        