#!/usr/bin/env python
#-*-coding:utf-8-*-
import os
import sys
from os import path as osp
import cv2
import numpy as np

def alignImages(im1, im2, max_features=500, good_match_percent=0.01):
    im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    # Detect ORB features and compute descriptors.
    orb = cv2.ORB_create(max_features)
    keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)

    # BRUTEFORCE_HAMMING
    matcher = cv2.DescriptorMatcher_create(4)
    matches = matcher.match(descriptors1, descriptors2, None)

    # Sort matches by score
    matches.sort(key=lambda x: x.distance, reverse=False)

    # Remove not so good matches
    numGoodMatches = int(len(matches) * good_match_percent)
    matches = matches[:numGoodMatches]


    # Draw top matches
    #cv.drawMatchesKnn(img1,kp1,img2,kp2,good, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    imMatches = np.empty((max(im1.shape[0], im2.shape[0]), im1.shape[1] + im2.shape[1], 3), dtype=np.uint8)
    cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, outImg=imMatches, flags=2)
    cv2.imwrite('matches.jpg', imMatches)


    # Extract location of good matches
    points1 = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
    points2 = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)
    M, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

    # Use homography to warp image
    im1Reg = cv2.warpPerspective(im1, M, dsize=im2.shape[:2])
    return M, im1Reg

def main():
    ref_image = sys.argv[1]
    target_image = sys.argv[2]
    outFilename = 'aligned.jpg'
    print('Reading reference image: {}'.format(ref_image))
    imReference = cv2.imread(ref_image)

    # Read image to be aligned
    print('Reading image to align: {}'.format(target_image)) 
    im = cv2.imread(target_image)

    # Align images
    print('Aligning images ...') 
    M, imReg = alignImages(im, imReference)

    # Write aligned image to disk. 
    print('Saving aligned image: {}'.format(outFilename)) 
    cv2.imwrite(outFilename, imReg)

    # Print estimated homography
    print('Estimated homography:\n{}'.format(M))

if __name__ == '__main__':
    main()