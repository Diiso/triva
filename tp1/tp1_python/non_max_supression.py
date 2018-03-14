import numpy as np

def nms(dI_norm, quantified_orientation, threshold):
    n,m = dI_norm.shape
    
    candidate_edges = []
    
    #find all candidate edges
    for i in range(n):
        for j in range(m):
            if dI_norm[i,j]>threshold:
                candidate_edges.append((i,j))
    
    # nms_edges is a boolean matrix
    nms_edges = np.zeros((n-1,m-1))
    
    #Consider all candidates edges
    for k in range(len(candidate_edges)):
        i,j = candidate_edges[k]
        
        if i>0 and j>0 and i<n-1 and j<m-1:
            if quantified_orientation[i,j] == 3:
                if (dI_norm[i,j]>dI_norm[i-1,j]) and (dI_norm[i,j]>dI_norm[i+1,j]):
                        nms_edges[i,j] = 1
            elif quantified_orientation[i,j] == 4:
                if (dI_norm[i,j]>dI_norm[i-1,j-1]) and (dI_norm[i,j]>dI_norm[i+1,j+1]):
                        nms_edges[i,j] = 1
            elif quantified_orientation[i,j] == 1:
                if (dI_norm[i,j]>dI_norm[i,j-1]) and (dI_norm[i,j]>dI_norm[i,j+1]):
                        nms_edges[i,j] = 1
            elif quantified_orientation[i,j] == 2:
                if (dI_norm[i,j]>dI_norm[i+1,j-1]) and (dI_norm[i,j]>dI_norm[i-1,j+1]):
                        nms_edges[i,j] = 1
            else:
                print("non_max_suppression : the input orientation matrix is not valid.")
                
    return nms_edges