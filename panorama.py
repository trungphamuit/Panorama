#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
from matplotlib import pyplot as plt
import imutils
import os
import math

class Pano():
    def __init__(self, path, ratio=0.75, mindist=40, half=True):
        '''
        path: path to dir contains images to be stitched
        ratio: ratio distance to throw away ambigious match
        mindist: min distance to throw away bad match
        half: only look for match in right half of left photo to avoid bad matches
        '''
        filepaths = [os.path.join(path, i) for i in os.listdir(path)]
        self.images = []
        for path in filepaths:
            self.images.append(cv2.imread(path))
        self.ratio = ratio
        self.mindist = mindist
        self.half = half

    def featureExtractor(self, im1, im2, method):
        '''
        im1, im2: images to detect features
        '''
        # convert images to grayscale
        query = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
        train = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
        # Detect and extract features from the image
        # Local invariant descriptors (SIFT, SURF, ORB)
        if method == 'sift':
            des = cv2.xfeatures2d.SIFT_create()
        elif method == 'surf':
            des = cv2.xfeatures2d.SURF_create()
        elif method == 'brisk':
            des = cv2.BRISK_create()
        elif method == 'orb':
            des = cv2.ORB_create()
        # t keypoints and descriptors
        kp1, des1 = des.detectAndCompute(query, None)
        kp2, des2 = des.detectAndCompute(train, None)
        return kp1, des1, kp2, des2

    def matchFeatures(self, des1, des2, method):
        '''
        des1, des2: descriptors to match
        '''
        "Create and return a Matcher Object"

        if method == 'sift' or method == 'surf':
            bf = cv2.BFMatcher(cv2.NORM_L2)
        elif method == 'orb' or method == 'brisk':
            bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        matches = bf.knnMatch(des1, des2, k=2)
        # Apply ratio test
        good = []
        for m, n in matches:
            if m.distance < self.ratio * n.distance and m.distance < self.mindist:
                good.append(m)
        good = np.asarray(good)
        return good

    def getNumberOfMatches(self, im1, im2, method):
        '''
        im1, im2: images to get number of matches from
        '''
        _, des1, _, des2 = self.featureExtractor(im1, im2, method)
        matches = self.matchFeatures(des1, des2, method)
        return len(matches)
    def cylindricalWarpImage(self, img1, K):
        f = K[0,0]

        im_h,im_w = img1.shape[:2]

        # go inverse from cylindrical coord to the image
        # (this way there are no gaps)
        cyl = np.zeros_like(img1)
        cyl_h, cyl_w = cyl.shape[:2]
        x_c = float(cyl_w) / 2.0
        y_c = float(cyl_h) / 2.0
        for x_cyl in np.arange(0,cyl_w):
            for y_cyl in np.arange(0,cyl_h):
                theta = (x_cyl - x_c) / f
                h     = (y_cyl - y_c) / f

                X = np.array([math.sin(theta), h, math.cos(theta)])
                X = np.dot(K,X)
                x_im = X[0] / X[2]
                if x_im < 0 or x_im >= im_w:
                    continue

                y_im = X[1] / X[2]
                if y_im < 0 or y_im >= im_h:
                    continue

                cyl[int(y_cyl),int(x_cyl)] = img1[int(y_im),int(x_im)]

        return cyl

    def stitch(self, query_image, train_image, method, mode):
        #query_image, train_image: images to stich together
        if mode == 'affine':
            h,w = query_image.shape[:2]
            f = 1000
            K = np.array([[f, 0, w/2], [0, f, h/2], [0, 0, 1]])
            query_image = self.cylindricalWarpImage(query_image, K)

            h,w = train_image.shape[:2]
            f = 1000
            K = np.array([[f, 0, w/2], [0, f, h/2], [0, 0, 1]]) # mock calibration matrix
            train_image = self.cylindricalWarpImage(train_image, K)

        # add some room for train image to be warped without losing parts
        query_image = cv2.copyMakeBorder(query_image,
                                         int(query_image.shape[0] * 0.3),
                                         0,
                                         int(query_image.shape[1] * 0.3),
                                         0,
                                         cv2.BORDER_CONSTANT, (0, 0, 0))
        if self.half:
            k = min(query_image.shape[1] // 2, query_image.shape[0] * 2)
            q = query_image.copy()
            q[:, :k] = 0
            kp1, des1, kp2, des2 = self.featureExtractor(q, train_image, method)
        else:
            kp1, des1, kp2, des2 = self.featureExtractor(query_image, train_image, method)
        # Feature matching
        good = self.matchFeatures(des1, des2, method)
        if len(good) < 4:
            return query_image
        # Sort good features
        good = sorted(good, key=lambda x: x.distance)
        if len(good) > 55:
            good = good[:55]
        # Homography estimation using RANSAC
        dst = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        src = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        if mode == 'homo':
            H, masked = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
            print('Transformation matrix: {}'.format(H))
            # Perspective warping
            width = train_image.shape[1] + query_image.shape[1]
            height = train_image.shape[0] + query_image.shape[0]
            result = cv2.warpPerspective(train_image, H, (width, height))
        
        if mode == 'affine':
            H, masked = cv2.estimateAffine2D(src, dst, cv2.RANSAC, ransacReprojThreshold=5.0)
            print('Transformation matrix: {}'.format(H))
            width = train_image.shape[1] + query_image.shape[1]
            height = train_image.shape[0] + query_image.shape[0]
            result = cv2.warpAffine(train_image, H, (width, height))
        # blend query image onto warped, erosion helps with smooth edges
        a = np.zeros_like(result)
        a[0:query_image.shape[0], 0:query_image.shape[1]] = query_image
        kernel = np.ones((5, 5), np.uint8)
        result = np.where(cv2.erode(a, kernel, iterations=1) == 0, result, a)
        return result

    def right_stitch(self, query_image, train_image, method, mode):
        '''
        query_image, train_image: images to stich together
        '''
        # mirror image, then stitch, then mirror back
        train_image, query_image = cv2.flip(train_image, +1), \
                                   cv2.flip(query_image, +1)
        result = cv2.flip(self.stitch(query_image, train_image, method, mode), +1)
        return result

    def crop(self, pano):
        '''
        pano: image to crop
        '''
        # gray and then thresh
        gray = cv2.cvtColor(pano, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
        # dilate to blend edges, help find better contour
        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.dilate(thresh, kernel, iterations=5)
        contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_NONE)
        cnt = imutils.grab_contours(contours)
        # get largest contour
        c = max(cnt, key=cv2.contourArea)
        # get bounding rectangle
        x, y, w, h = cv2.boundingRect(c)
        # crop
        result = pano[y:y + h, x:x + w]
        return result

    def createPanorama(self, method, mode):
        # find leftmost image
        numberOfMatches = [0] * len(self.images)
        for pos, i in enumerate(self.images):
            # only look at left half
            left = i[:, :i.shape[1] // 2]
            for j in self.images:
                right = j[:, j.shape[1] // 2:]
                numberOfMatches[pos] += self.getNumberOfMatches(left, right, method)
        # now sort images from left to right
        leftmost = self.images.pop(np.argmin(numberOfMatches))
        imagesRtoL = [leftmost]
        ith_leftmost = imagesRtoL[0]
        while len(self.images) != 0:
            numberOfMatches = [0] * len(self.images)
            for pos, i in enumerate(self.images):
                numberOfMatches[pos] += self.getNumberOfMatches(ith_leftmost, i, method)
            imagesRtoL.append(self.images.pop(np.argmax(numberOfMatches)))
        # pick the middle image
        n = len(imagesRtoL)
        mid = n // 2
        result = imagesRtoL[mid]
        # stitch images to the right of middle
        for i in range(mid + 1, n):
            result = self.crop(self.stitch(result, imagesRtoL[i], method, mode))
        # stitch images to the left of middle
        for i in range(mid - 1, -1, -1):
            result = self.crop(self.right_stitch(result, imagesRtoL[i], method, mode))
        return result


def demo():
    p = Pano('test1', 0.75, 47, True)  # using test1 folder as demo
    method = 'orb'
    mode = 'affine'
    pano = p.createPanorama(method, mode)
    cv2.imwrite('demo.png', pano)


def main(folder):
    p = Pano(folder, 0.75, 47, True)
    method = 'orb'
    mode = 'homo'
    pano = p.createPanorama(method, mode)
    cv2.imwrite('pano.png', pano)


if __name__ == '__main__':
    demo()
