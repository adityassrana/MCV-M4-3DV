import numpy as np

import utils as h

def print_epipolar_eq(x1, x2, F):
    for pt1, pt2 in zip(x1, x2):
        print ("epipolar equation: ", pt2.T@F@pt1)

def nullspace(A, atol=1e-13, rtol=0):
    # from https://scipy-cookbook.readthedocs.io/items/RankNullspace.html 
    A = np.atleast_2d(A)
    u, s, vh = np.linalg.svd(A)
    tol = max(atol, rtol * s[0])
    nnz = (s >= tol).sum()
    ns = vh[nnz:].conj().T
    
    return ns

def hat_operator(v):
    # returns the skew-symmetric matrix generated from array v
    # In https://en.wikipedia.org/wiki/Skew-symmetric_matrix#Cross_product the hat operator of a vector a is represented by [a]x
    # taken from https://pythonpath.wordpress.com/2012/09/04/skew-with-numpy-operations/
    if len(v) == 4: v = v[:3]/v[3] # for homogeneous vectors
    skv = np.roll(np.roll(np.diag(v.flatten()), 1, 1), -1, 0)
    return skv - skv.T

def skew(a):
    """ Skew matrix A such that a x v = Av for any v. """

    return np.array([[0,-a[2],a[1]],[a[2],0,-a[0]],[-a[1],a[0],0]])
