import numpy as np
from numpy import linalg as LA
import cv2
import math
import sys
import random
import matplotlib
from matplotlib import pyplot as plt
from operator import itemgetter

def Normalization(x):

    x = np.asarray(x)
    x = x  / x[2,:]
    
    m, s = np.mean(x, 1), np.std(x)
    s = np.sqrt(2)/s;
 
    Tr = np.array([[s, 0, -s*m[0]], [0, s, -s*m[1]], [0, 0, 1]])


    xt = Tr @ x
        
    return Tr, xt

def fundamental_matrix(points1, points2):
    
    # Normalize points in both images
    T1, points1n = Normalization(points1)
    T2, points2n = Normalization(points2)

    A = []
    n = points1.shape[1]

    for i in range(n):
        x, y = points1n[0, i], points1n[1, i]
        u, v = points2n[0, i], points2n[1, i]

        A.append( [u*x, u*y, u, v*x, v*y, v, x, y, 1] )
    

    # Convert A to array
    A = np.asarray(A) 

    U, d, Vt = np.linalg.svd(A)
    
    # Extract fundamental matrix (last line of Vt)
    L = Vt[-1, :] / Vt[-1, -1]
    F = L.reshape(3, 3)
    
    # Enforce constraint that fundamental matrix has rank 2 by performing
    # an SVD and then reconstructing with the two largest singular values.
    U, d, Vt = np.linalg.svd(F)
    D = np.zeros((3,3))
    D[0,0] = d[0]
    D[1,1] = d[1]
    F = U @ D @ Vt
    
    # Denormalise
    F = T2.T @ F @ T1
    
    return F

def Inliers(F, x1, x2, th):
    
    Fx1 = F @ x1
    Ftx2 = F.T @ x2
    
    n = x1.shape[1]
    x2tFx1 = np.zeros((1, n))
 
    for i in range(n):
        x2t = x2[:,i]
        x2t = x2t.T
        x2tFx1[0,i] = x2t @ F @ x1[:,i]
    
    # evaluate distances
    den = Fx1[0,:]**2 + Fx1[1,:]**2 + Ftx2[0,:]**2 + Ftx2[1,:]**2
    den = den.reshape((1, n))
   
    d = x2tFx1**2 / den
    
    inliers_indices = np.where(d[0,:] < th)
    
    return inliers_indices[0]



def Ransac_fundamental_matrix(points1, points2, th, max_it_0):
    
    Ncoords, Npts = points1.shape
    
    it = 0
    best_inliers = np.empty(1)
    max_it = max_it_0

    while it < max_it:
        indices = random.sample(range(1, Npts), 8)
        F = fundamental_matrix(points1[:,indices], points2[:,indices])
        inliers = Inliers(F, points1, points2, th)
        
        # test if it is the best model so far
        if inliers.shape[0] > best_inliers.shape[0]:
            best_inliers = inliers
            
        # update estimate of iterations (the number of trials) to ensure we pick, with probability p, 
        # an initial data set with no outliers
        fracinliers = inliers.shape[0]/Npts
        pNoOutliers = 1 -  fracinliers**8
        eps = sys.float_info.epsilon
        pNoOutliers = max(eps, pNoOutliers)   # avoid division by -Inf
        pNoOutliers = min(1-eps, pNoOutliers) # avoid division by 0
        p = 0.99
        max_it = math.log(1-p)/math.log(pNoOutliers)
        if max_it > max_it_0:
            max_it = max_it_0

        it += 1
        
    
    # compute H from all the inliers
    F = fundamental_matrix(points1[:,best_inliers], points2[:,best_inliers])
    
    print(it,inliers.shape[0])
    
    inliers = best_inliers
    
    return F, inliers