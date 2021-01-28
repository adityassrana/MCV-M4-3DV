import sys
import cv2
import numpy as np

import utils as h
import math
import maths as mth
import random
from operator import itemgetter

def normalise_coord(p1, p2):
    # normalise both sets

    #alternative way of computing S1
    #dist_mean = np.mean(np.sqrt(((p1[:,:2] - mean_1)**2).sum(axis=1)), dtype=np.float32)
    #s1 = np.sqrt(2.)/dist_mean

    mean_1 = np.mean(p1[:,:2], axis=0, dtype=np.float32)
    S1 = np.sqrt(2.) / np.std(p1[:, :2], dtype=np.float32)
    T1 = np.float32(np.array([[S1, 0, -S1*mean_1[0]], [0, S1, -S1*mean_1[1]], [0, 0, 1]]))
    p1 = T1@p1.T

    mean_2 = np.mean(p2[:,:2],axis=0, dtype=np.float32)
    S2 = np.sqrt(2.) / np.std(p2[:, :2], dtype=np.float32)
    T2 = np.float32(np.array([[S2, 0, -S2*mean_2[0]], [0, S2, -S2*mean_2[1]], [0, 0, 1]]))
    p2 = T2@p2.T

    if h.debug >= 0:
        print("    Coordinates normalised")

    return p1.T, p2.T, T1, T2

def compute_fundamental_robust(matches, points1, points2):
    #Robust estimation of the fundamental matrix #
    
    points1 = np.asarray(points1)
    points1 = points1.T
    points2 = np.asarray(points2)
    points2 = points2.T

    F, indices_inlier_matches = Ransac_fundamental_matrix(points1, points2, 1, 5000)
    inlier_matches = itemgetter(*indices_inlier_matches)(matches)
    
    return F, inlier_matches


def compute_fundamental(p1, p2, eight_alg):
    # compute fundamental matrix with normalised coordinates

    tol_rsc = np.array([1.5, 1.5, 1])

    if h.normalise:
        # make coordinates homogeneous:
        p1h = make_homogeneous(p1)
        p2h = make_homogeneous(p2)
        # normalise coordinates 
        p1_norm, p2_norm, T1, T2 = normalise_coord(p1h, p2h)
        #tolerance normalised for RANSAC method (not used by LMEDS, left for
        #compatibility with normalise=False path)
        tol_rsc_nrm = T1@tol_rsc
        # Only LMEDS seems to work well with normalised coordinates
        fund_method = cv2.FM_LMEDS
    else:
        p1_norm, p2_norm = p1, p2
        tol_rsc_nrm = tol_rsc
        fund_method = cv2.FM_RANSAC

    if eight_alg:
        fund_method = cv2.FM_8POINT

    F, mask = cv2.findFundamentalMat(p1_norm[:, :2], p2_norm[:, :2], fund_method, tol_rsc_nrm[0], 0.99)

    if h.normalise:
        # denormalise F
        F = T2.T@F@T1 
        F = F / F[2][2]
  
    if h.debug >= 0:
        print('    Fundamental Matrix estimated')
    if h.debug > 1:
        print("      Fundamental Matrix: \n", F)

    return F, mask


def apply_mask(x1, x2, mask, F):
    # use F mask for filtering out outliers
    if h.debug > 2: 
        print("before F mask:\n")
        xh1 = make_homogeneous(x1)
        xh2 = make_homogeneous(x2)
        mth.print_epipolar_eq(xh1, xh2, F)

    # apply mask of inliers to the set of matches
    x1 = x1[mask.ravel() == 1]
    x2 = x2[mask.ravel() == 1]

    if h.debug > 2: 
        print("after F mask:\n")
        xh1 = make_homogeneous(x1)
        xh2 = make_homogeneous(x2)
        mth.print_epipolar_eq(xh1, xh2, F)

    if h.debug >= 0:
        print("    Mask given by F applied")
    if h.debug > 0:
        print("      F mask has selected", x1.shape[0], "inliers")

    return x1, x2


def refine_matches(x1, x2, F):
    # use the optimal triangulation method (Algorithm 12.1 from MVG)
    nx1, nx2 = cv2.correctMatches(F, np.reshape(x1, (1, -1, 2)), np.reshape(x2, (1, -1, 2)))

    # get the points back in matrix configuration
    xr1 = np.float32(np.reshape(nx1,(-1, 2)))
    xr2 = np.float32(np.reshape(nx2,(-1, 2)))

    if h.debug >= 0:
        print("  Matches corrected with Optimal Triangulation Method")

    if h.debug > 2: 
        print("xr1: \n", xr1)
        print("xr2: \n", xr2)
        print("after correctMatches: ")
        xrh1 = make_homogeneous(xr1)
        xrh2 = make_homogeneous(xr2)
        mth.print_epipolar_eq(xrh1, xrh2, F)

    return xr1.T, xr2.T 


def search_more_matches(out1, out2, F):
    # your code here

    e = 0.00155

    outh1 = make_homogeneous(out1)
    outh2 = make_homogeneous(out2)

    xn1 = np.empty([0, 2], dtype=np.float32)
    xn2 = np.empty([0, 2], dtype=np.float32)
    on1 = np.empty([0, 2], dtype=np.float32)
    on2 = np.empty([0, 2], dtype=np.float32)

    for oh1, oh2 in zip(outh1, outh2):
        # compute epipolar lines
        l1 = F.T@oh2
        l2 = F@oh1

        # distance from a point to a line
        d1 = abs(np.dot(l1, oh1)) / np.sqrt(np.sum(l1[0:2]**2))
        d2 = abs(np.dot(l2, oh2)) / np.sqrt(np.sum(l2[0:2]**2))
        d = d1 + d2

        if d < e:
            xn1 = np.r_[xn1, [np.float32(oh1[0:2]/oh1[2])]]
            xn2 = np.r_[xn2, [np.float32(oh2[0:2]/oh2[2])]]
        else:
            on1 = np.r_[on1, [np.float32(oh1[0:2]/oh1[2])]]
            on2 = np.r_[on2, [np.float32(oh2[0:2]/oh2[2])]]

    return xn1, xn2, on1, on2


def make_homogeneous(p):
    if p.shape[1] != 2:
        print("WARNING - Coordinates have ", p.shape, " dimensions: not made homogeneous")
        return p
    
    # Add homogeneous coordinates 
    hom_c = np.ones_like(p[:, 1], dtype=np.float32)
    p = np.c_[p, hom_c]

    # Perhaps is quicker to use opencv implementation, but it's 
    # cumbersome... seems to use lists
    #p = np.reshape(cv2.convertPointsToHomogeneous(p), (-1, 3))

    if h.debug > 0:
        print("    Coordinates made homogeneous")

    return p

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
