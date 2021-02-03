import sys
import random
import math

import numpy as np
import matplotlib.pyplot as plt

from math import ceil
from scipy.ndimage import map_coordinates

from scipy.optimize import least_squares
from math import ceil


def Normalise_last_coord(x):
    xn = x  / x[-1]
    return xn

#-----Helper Functions for DLT_homography
def normalize_points(points):
    """
    Normalize points using a scaling and translation(similarity matrix)
    such that centroid of the new points is at (0,0) and their average 
    distance from origin is square root of 2.

    Args:
        points: points in homogeneous form and in (3, num_points) format
    Returns:
        transformed points
    """
    means = np.mean(points, axis=1)
    mean_x = means[0]
    mean_y = means[1]
    
    dist = np.sqrt((np.square(points[0] - mean_x)) + np.square(points[1] - mean_y))
    mean_dist = np.mean(dist)
    
    s = np.sqrt(2)/mean_dist
    tx = -s*mean_x
    ty = -s*mean_y
    
    T = np.array([[s,0, tx],
                 [0, s, ty],
                 [0,0,1]])
    
    norm_points = T@points
    return norm_points, T


def normalize_world_points(pts):
    """
    Like normalize_points but for world (3D) homogeneous points.
    """
    means = np.mean(pts, axis=1)
    #print(f"Means: {means.shape}")
    mean_x = means[0]
    mean_y = means[1]
    mean_z = means[2]
    
    dist = np.sqrt((np.square(pts[0] - mean_x)) + np.square(pts[1] - mean_y))
    mean_dist = np.mean(dist)
    s = np.sqrt(2)/mean_dist
    tx = -s*mean_x
    ty = -s*mean_y
    tz = -s*mean_z
    
    T = np.array([[s, 0, 0, tx],
                 [ 0, s, 0, ty],
                 [ 0, 0, s, tz],
                 [ 0, 0, 0, 1]])
    
    norm_points = T@pts
    #print(f"Normalized 3D points: {norm_points}")
    return norm_points, T

def get_equations_from_points(X, x_):
    """
    Form constraints required for solving the projective camera matrix from a set of
    3D (X) and 2D (x_) points.

    Args:
        set of points in homogeneous format
    Returns:
        set of equations derived from the points
    """
    X = Normalise_last_coord(X)
    x_, y_, w_ = Normalise_last_coord(x_)
    
    eq1 = np.concatenate((np.zeros_like(X), -w_ * X, y_ * X))
    eq2 = np.concatenate((w_ * X, np.zeros_like(X), -x_ * X))
    #print(f"Equation1: {eq1.shape}")
    return eq1,eq2


def euclid(x):
    return x[:, :-1] / x[:, [-1]]


# ----------------------------------- DLT algorithm (for camera projection matrix) -----------------------------------

def get_camera_projection_matrix(X, x_):
    """
    DLT based: X are 3D points and x_ their 2D correspondences in the image
    """
    # normalize points
    X, T1 = normalize_world_points(X)
    x_, T2 = normalize_points(x_)
    
    num_points = X.shape[1]
    Eq_list = []
    for point1, point2 in zip(X.T,x_.T):
        eq1,eq2 = get_equations_from_points(point1, point2)
        Eq_list.append(eq1)
        Eq_list.append(eq2)
        
    Eq = np.array(Eq_list)
    #print(Eq)
    
    U,D,Vt = np.linalg.svd(Eq)
    
    # take the last row of V_transpose
    # this is equivalent to taking the last column of V
    P_tilde = np.reshape(Vt[-1],(3,4))
    
    T2_inv = np.linalg.inv(T2)
    
    P = T2_inv@P_tilde@T1
    
    #normalize
    P = P/P[-1,-1]
    
    return P


# ----------------------------------- GOLD STANDARD -----------------------------------

def geometric_error_terms(variables, points):
    P = variables[:12].reshape(3,4)

    npoints = int(len(points)/7)
    dpoints1 = points[:npoints*4] #3D => size 4
    dpoints2 = points[npoints*4:] #2D => size 3
    
    #reshape back to original sizes
    X = np.reshape(dpoints1,(4,-1))
    x_ = np.reshape(dpoints2,(3,-1))
    
    # get optimal x_hat and x_hatp
    Xhat = variables[12:]
    Xhat = np.reshape(Xhat,(4,-1))
    
    Ppoints = P@Xhat
    Ppoints /= Ppoints[-1, :]
    Ppoints = np.concatenate((Ppoints[0,:], Ppoints[1,:], Ppoints[2,:]), axis=0)
    
    e1 = dpoints1 - variables[12:]
    e2 = dpoints2 - Ppoints
    
    return np.concatenate((e1, e2), axis=0)


def refine_projective_matrix(P, points3D, points2D):
    data_points = np.concatenate([points3D.flatten(),points2D.flatten()])
    variables0 = np.concatenate([P.flatten(),points3D.flatten()])

    result = least_squares(geometric_error_terms, variables0, method='trf', args=([data_points]), verbose=2)
    Pr = np.reshape(result.x[:12],(3,4))

    return Pr


                  
# ----------------------------------- RANSAC -----------------------------------



def Inliers(P, X, x_, th):
    '''
    Computes the inliers on a set of putative correspondeces for a given transformation.
    Input:
        - H: 3 x 4 homography
        - X: Points on image 1 in the format (4, num points)
        - xp: Points on image 2 in the format (3, num points)
        - th: distance threshold. Pairs with a distance smaller than th will be considered inliers
    Returns:
        - numpy array containing the indeces of the inliers
    '''
    trans_x_ = P@X
    # Normalize
    x_ /= x_[2,:]
    trans_x_ /= trans_x_[2,:]

    #print(x_.shape)
    #print(trans_x_.shape)
    dist = np.sum(np.square(euclid(x_) - euclid(trans_x_)), axis=0)
    #print(f"Dist shape: {dist.shape}")
    return np.where(dist < th*th)[0]



def get_camera_projection_matrix_RANSAC(points3D, points2D, th, max_it):
    Ncoords, Npts = points3D.shape
    
    it = 0
    best_inliers = np.empty(1)
    
    while it < max_it:
        indices = random.sample(range(1, Npts), 6)
        P = get_camera_projection_matrix(points3D[:,indices], points2D[:,indices])
        inliers = Inliers(P, points3D, points2D, th)
        
        # test if it is the best model so far
        if inliers.shape[0] > best_inliers.shape[0]:
            best_inliers = inliers
        
        # update estimate of iterations (the number of trials) to ensure we pick, with probability p,
        # an initial data set with no outliers
        fracinliers = inliers.shape[0]/Npts
        pNoOutliers = 1 -  fracinliers**6
        eps = sys.float_info.epsilon
        pNoOutliers = max(eps, pNoOutliers)   # avoid division by -Inf
        pNoOutliers = min(1-eps, pNoOutliers) # avoid division by 0
        p = 0.99
        max_it = math.log(1-p)/math.log(pNoOutliers)
        
        it += 1
    
    # compute H from all the inliers
    P = get_camera_projection_matrix(points3D[:,best_inliers], points2D[:,best_inliers])
    
    # we return the refined homography
    return refine_projective_matrix(P, points3D[:, best_inliers], points2D[:, best_inliers])


# ----------------------------------------------------------------------------------------