#!/usr/bin/env python
# coding: utf-8


import numpy as np
import cv2
import matplotlib
from matplotlib import pyplot as plt

def crop(I):
    ys, xs = np.where((I[:,:,0] != 0) & (I[:,:,1] != 0) & (I[:,:,2] != 0))
    x0, y0, x, y = np.min(xs), np.min(ys), np.max(xs), np.max(ys)
    return I[y0:y, x0:x]

class Stitcher:
    def __init__(self):
        pass
        
    def stitch(self, images, ratio=0.85, reprojThresh=4.0, showMatches=False):
        
        imageB, imageA = images
        kpsA, featuresA = self.detectAndDescribe(imageA)
        kpsB, featuresB = self.detectAndDescribe(imageB)

        # match features between the two images
        M = self.matchKeypoints(kpsA, kpsB,
            featuresA, featuresB, ratio, reprojThresh)
        # if the match is None, then there aren't enough matched
        # keypoints to create a panorama

        if M is None:
            return None
        
        # otherwise, apply a perspective warp to stitch the images
        # together

        (matches, H, status) = M
        result = cv2.warpPerspective(imageA, H, 
                (imageA.shape[1] + imageB.shape[1], imageA.shape[0]))

        result[0:imageB.shape[0], 0:imageB.shape[1]] = imageB

        # check to see if the keypoint matches should be visualized
        if showMatches:
            vis = self.drawMatches(imageA, imageB, kpsA, kpsB, matches,
                status)
            # return a tuple of the stitched image and the
            # visualization
            return (result, vis)
        # return the stitched image
        return result

    def detectAndDescribe(self, image):
        # convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # check to see if we are using OpenCV 3.X

        # detect and extract features from the image
        descriptor = cv2.ORB_create()
        (kps, features) = descriptor.detectAndCompute(image, None)
        # otherwise, we are using OpenCV 2.4.X

        kps = np.float32([kp.pt for kp in kps])
        # return a tuple of keypoints and features
        return (kps, features)
    
    def matchKeypoints(self, kpsA, kpsB, featuresA, featuresB,
        ratio, reprojThresh):
        # compute the raw matches and initialize the list of actual
        # matches
        matcher = cv2.BFMatcher(normType=cv2.NORM_HAMMING)

        rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
        matches = []
        # loop over the raw matches
        for m in rawMatches:
            # ensure the distance is within a certain ratio of each
            # other (i.e. Lowe's ratio test)
            if len(m) == 2 and m[0].distance < m[1].distance * ratio:
                matches.append((m[0].trainIdx, m[0].queryIdx))

        # computing a homography requires at least 4 matches
        if len(matches) > 4:
            # construct the two sets of points
            ptsA = np.float32([kpsA[i] for (_, i) in matches])
            ptsB = np.float32([kpsB[i] for (i, _) in matches])
            # compute the homography between the two sets of points
            (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC,
                reprojThresh)
            # return the matches along with the homograpy matrix
            # and status of each matched point
            return (matches, H, status)
        # otherwise, no homograpy could be computed
        return None
    
    def drawMatches(self, imageA, imageB, kpsA, kpsB, matches, status):
        # initialize the output visualization image
        (hA, wA) = imageA.shape[:2]
        (hB, wB) = imageB.shape[:2]
        vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
        vis[0:hA, 0:wA] = imageA
        vis[0:hB, wA:] = imageB
        # loop over the matches
        for ((trainIdx, queryIdx), s) in zip(matches, status):
            # only process the match if the keypoint was successfully
            # matched
            if s == 1:
                # draw the match
                ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
                ptB = (int(kpsB[trainIdx][0]) + wA, int(kpsB[trainIdx][1]))
                cv2.line(vis, ptA, ptB, (0, 255, 0), 1)
        # return the visualization
        return vis


if __name__ == '__main__':     
    # img1 = cv2.imread('Data/llanes/llanes_a.jpg')
    # img2 = cv2.imread('Data/llanes/llanes_b.jpg')
    # img3 = cv2.imread('Data/llanes/llanes_c.jpg')

    # img1 = cv2.imread('Data/castle_int/0014_s.png')
    # img2 = cv2.imread('Data/castle_int/0015_s.png')
    # img3 = cv2.imread('Data/castle_int/0016_s.png')

    img1 = cv2.imread("Data/aerial/site22/frame_00001.tif")
    img2 = cv2.imread("Data/aerial/site22/frame_00018.tif")
    img3 = cv2.imread("Data/aerial/site22/frame_00030.tif")

    stitcher = Stitcher()

    (result, vis) = stitcher.stitch([img2, img1], showMatches=True)
    #result = crop(result)

    (result2, vis2) = stitcher.stitch([img3, result], showMatches=True)

    #cv2.imshow("Image A", img1)
    #cv2.imshow("Image B", img2)
    #cv2.imshow("Image C", img3)

    #cv2.imshow("Keypoint Matches 1", vis)
    cv2.imshow("Result 1", crop(result))

    #cv2.imshow("Keypoint Matches 2", vis2)
    cv2.imshow("Result 2", crop(result2))

    cv2.waitKey(0)




