#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
from matplotlib import pyplot as plt
import imutils
import os

class Pano():
    def __init__(self, path, ratio=0.75, mindist=40, half=True):
        '''
        path: path to dir contains images to be stitched
        ratio: ratio distance to throw away ambigious match
        mindist: min distance to throw away bad match
        half: only look for match in right half of left photo to avoid bad matches
        '''
        filepaths = [os.path.join(path,i) for i in os.listdir(path)]
        self.images = []
        for path in filepaths:
            self.images.append(cv2.imread(path))
        self.ratio = ratio
        self.mindist = mindist
        self.half = half
    
    def featureExtractor(self, im1, im2):
        '''
        im1, im2: images to detect features
        '''
        # convert images to grayscale
        query = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
        train = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
        #Keypoint detection
        #Local invariant descriptors (SIFT, SURF, ORB)
        orb = cv2.ORB_create()
        kp1, des1 = orb.detectAndCompute(query,None)
        kp2, des2 = orb.detectAndCompute(train,None)
        return kp1, des1, kp2, des2

    def matchFeatures(self, des1, des2):
        '''
        des1, des2: descriptors to match
        '''
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        matches = bf.knnMatch(des1, des2, k=2)
        # Apply ratio test
        good = []
        for m,n in matches:
            if m.distance < self.ratio*n.distance and m.distance < self.mindist :
                good.append(m)
        good = np.asarray(good)
        return good
    
    def getNumberOfMatches(self, im1, im2):
        '''
        im1, im2: images to get number of matches from
        '''
        _, des1, _, des2 = self.featureExtractor(im1, im2)
        matches = self.matchFeatures(des1, des2)
        return len(matches)
    
    def stitch(self, query_image, train_image):
        '''
        query_image, train_image: images to stich together
        '''
        # add some room for train image to be warped without losing parts
        query_image = cv2.copyMakeBorder(query_image,
                                         int(query_image.shape[0]*0.3),
                                         0,
                                         int(query_image.shape[1]*0.3),
                                         0,
                                         cv2.BORDER_CONSTANT, (0, 0, 0))
        if self.half:
            k = min(query_image.shape[1]//2, query_image.shape[0]*2)
            q = query_image.copy()
            q[:,:k] = 0
            kp1, des1, kp2, des2 = self.featureExtractor(q, train_image)
        else:
            kp1, des1, kp2, des2 = self.featureExtractor(query_image, train_image)
        #Feature matching
        good = self.matchFeatures(des1, des2)
        if len(good) < 4:
            return query_image
        #Sort good features
        good = sorted(good, key=lambda x: x.distance)
        if len(good) > 55:
            good = good[:55]
        # Homography estimation using RANSAC
        dst = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        src = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
        H, masked = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        print('Transformation matrix: {}'.format( H))
        # Perspective warping
        width = train_image.shape[1] + query_image.shape[1]
        height = train_image.shape[0] + query_image.shape[0]
        result = cv2.warpPerspective(train_image, H, (width, height))
        # blend query image onto warped, erosion helps with smooth edges
        a= np.zeros_like(result)
        a[0:query_image.shape[0], 0:query_image.shape[1]] = query_image
        kernel = np.ones((5,5), np.uint8) 
        result = np.where(cv2.erode(a, kernel, iterations=1) == 0, result, a)
        return result
        
    def right_stitch(self, query_image, train_image):
        '''
        query_image, train_image: images to stich together
        '''
        # mirror image, then stitch, then mirror back
        train_image, query_image = cv2.flip(train_image, +1),\
        cv2.flip(query_image, +1)
        result = cv2.flip(self.stitch(query_image, train_image), +1)
        return result

    def crop(self, pano):
        '''
        pano: image to crop
        '''
        # gray and then thresh
        gray = cv2.cvtColor(pano,cv2.COLOR_BGR2GRAY)
        _,thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY)
        # dilate to blend edges, help find better contour
        kernel = np.ones((5,5), np.uint8) 
        thresh = cv2.dilate(thresh, kernel, iterations=5)
        contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
        			cv2.CHAIN_APPROX_NONE)
        cnt = imutils.grab_contours(contours)
        # get largest contour
        c = max(cnt, key=cv2.contourArea)
        # get bounding rectangle
        x,y,w,h = cv2.boundingRect(c)
        # crop
        result = pano[y:y + h, x:x + w]
        return result

    def createPanorama(self):
        # find leftmost image
        numberOfMatches = [0]*len(self.images)
        for pos, i in enumerate(self.images):
            # only look at left half
            left = i[:, :i.shape[1]//2]
            for j in self.images:
                right = j[:,j.shape[1]//2:]
                numberOfMatches[pos] += self.getNumberOfMatches(left, right)
        # now sort images from left to right
        leftmost = self.images.pop(np.argmin(numberOfMatches))
        imagesRtoL = [leftmost]
        ith_leftmost = imagesRtoL[0]
        while len(self.images) != 0:
            numberOfMatches = [0]*len(self.images)
            for pos, i in enumerate(self.images):
                numberOfMatches[pos] += self.getNumberOfMatches(ith_leftmost, i)    
            imagesRtoL.append(self.images.pop(np.argmax(numberOfMatches)))
        # pick the middle image
        n = len(imagesRtoL)
        mid = n//2
        result = imagesRtoL[mid]
        # stitch images to the right of middle
        for i in range(mid+1, n):
            result = self.crop(self.stitch(result, imagesRtoL[i]))
        # stitch images to the left of middle
        for i in range(mid-1, -1, -1):
            result = self.crop(self.right_stitch(result, imagesRtoL[i]))
        return result


def demo():
    p = Pano('test1', 0.75, 47, True) # using test1 folder as demo
    pano = p.createPanorama()
    cv2.imwrite('demo.png', pano)


def main(folder):
    p = Pano(folder, 0.75, 47, True)
    pano = p.createPanorama()
    cv2.imwrite('pano.png', pano)


if __name__ == '__main__':
    demo()
