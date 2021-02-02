import cv2
import numpy as np

import utils as h

def find_features_sift(img, i):
    # find the keypoints and descriptors in img
    #SIFT used
    sift = cv2.xfeatures2d.SIFT_create()

    kp, des = sift.detectAndCompute(img, None)

    if h.debug >= 0:
        print ("  Features detected in image ",i)
    if (h.debug > 0):
        print("    Found", len(kp), "features ")

    return kp, des 

def find_features_orb(img, i):
    # find the keypoints and descriptors in img
    #ORB used
    orb = cv2.ORB_create(3000)
    kp, des = orb.detectAndCompute(img,None)

    if h.debug >= 0:
        print ("  Features detected in image ",i)
    if (h.debug > 0):
        print("    Found", len(kp), "features ")

    return kp, des

def match_features_kdtree(des1, des2, i, j):
    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    if h.debug >= 0:
        print ("  Correspondences matched between images", i, "and", j)
    if (h.debug > 0):
        print ("    Found", len(matches), "matching correspondences")

    return matches

def match_features_hamming(des1, des2, i, j):
    # Keypoint matching
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1,des2)
    
    if h.debug >= 0:
        print ("  Correspondences matched between images", i, "and", j)
    if (h.debug > 0):
        print ("    Found", len(matches), "matching correspondences")

    return matches

def filter_matches(kp1, kp2, matches, imgi, imgj):
    x1 = np.empty([0, 2], dtype=np.float32)
    x2 = np.empty([0, 2], dtype=np.float32)
    o1 = np.empty([0, 2], dtype=np.float32)
    o2 = np.empty([0, 2], dtype=np.float32)

    # ratio test as per Lowe's paper, when 2 candidates are provided 
    for m, n in matches:
        if m.distance < 0.8*n.distance:
            x1 = np.r_[x1, [np.float32(np.array(kp1[m.queryIdx].pt))]]
            x2 = np.r_[x2, [np.float32(np.array(kp2[m.trainIdx].pt))]]
        else:
            o1 = np.r_[o1, [np.float32(np.array(kp1[m.queryIdx].pt))]]
            o2 = np.r_[o2, [np.float32(np.array(kp2[m.trainIdx].pt))]]

    x1_u, x2_u = remove_duplicates(x1, x2)
    o1_u, o2_u = remove_duplicates(o1, o2)

    if h.debug >= 0:
        print ("  Matches between", imgi, "and", imgj, "filtered with Lowe's ratio")
    if (h.debug > 0):
        print ("    Selected", x1_u.shape[0], "matches")
        #print ("    Selected", x1.shape[0], "matches")

    return [x1_u, x2_u, o1_u, o2_u]

def remove_duplicates(x1, x2):
    # Remove duplicates from matches x1, x2
    arr1, uniq_cnt1 = np.unique(x1, axis=0, return_counts=True)
    arr2, uniq_cnt2 = np.unique(x2, axis=0, return_counts=True)

    x1_u = np.empty([0, 2], dtype=np.float32)
    x2_u = np.empty([0, 2], dtype=np.float32)
    if arr1.shape[0] <= arr2.shape[0]:
        small = arr1
        sm_ct = uniq_cnt1
        big = arr2
        bg_ct = uniq_cnt2
        x_small = x1_u
        x_big = x2_u
    else: 
        small = arr2
        sm_ct = uniq_cnt2
        big = arr1
        bg_ct = uniq_cnt1
        x_small = x2_u
        x_big = x1_u
    
    if (h.debug > 1):
        print("      Size of small in duplicates:",small.shape)
        print("      Size of big in duplicates:",big.shape)

    for it in range(small.shape[0]):
        if sm_ct[it] == 1 and bg_ct[it] == 1:
            x_small = np.r_[x_small, [small[it]]]
            x_big = np.r_[x_big, [big[it]]]

    if (h.debug > 1):
        print("      Size of duplicates:", x2_u.shape)

    if (small == arr1):
        return x_small, x_big
    else:
        return x_big, x_small
