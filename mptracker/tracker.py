#! /usr/bin/env python
# Mouse pupil tracker main objects.

import numpy as np
import cv2
from .utils import *
import os

def ellipseToContour(pupil_pos,a,b,phi,R=np.linspace(0,2.1*np.pi, 20)):
    xx = pupil_pos[1] + a*np.cos(R)*np.cos(phi) - b*np.sin(R)*np.sin(phi)
    yy = pupil_pos[0] + a*np.cos(R)*np.sin(phi) + b*np.sin(R)*np.cos(phi)
    shape1 = np.zeros((len(xx),1,2),dtype='int32')
    shape1[:,0,0] = xx.astype('int32')
    shape1[:,0,1] = yy.astype('int32')
    return shape1

class MPTracker(object):
    def __init__(self,parameters = None):
        if parameters is None:
            self.parameters = {
                'contrast_clipLimit':10,
                'contrast_gridSize':5,
                'gaussian_filterSize':7,
                'threshold':40,
                'eye_radius_mm':2.4, #this was set to 3*0.8 in the matlab version
                'number_frames':0,
            }
        else:
            self.parameters = parameters
        self.set_clhe()
        self.ROIpoints = []
        self.crApprox = None
        self.R = np.linspace(0,2.1*np.pi, 20)
        self.concatenateBinaryImage=False
    def setROI(self, points):
        self.ROIpoints = points
    def set_clhe(self):
        if (self.parameters['contrast_gridSize']>3):
            self.clahe = cv2.createCLAHE(
                clipLimit=self.parameters['contrast_clipLimit'],
                tileGridSize=(self.parameters['contrast_gridSize'],
                              self.parameters['contrast_gridSize']))
        else:
            class dummy(object):
                def __init__(self):
                    pass
                def apply(self,img):
                    return img
            self.clahe = dummy()

    def apply(self,img):
        #return self.applyStarburst(img)
        return self.applyShapeAnalysis(img)
    
    def applyShapeAnalysis(self,img):
        # Image improvements
        img = self.clahe.apply(img)
        img = cv2.GaussianBlur(img,
                               (self.parameters['gaussian_filterSize'],
                                self.parameters['gaussian_filterSize']),0)
        x1,y1 = (0,0)
        if len(self.ROIpoints) >= 4:
             pts = np.array(self.ROIpoints).reshape((-1,1,2))
             rect = cv2.boundingRect(pts)
             (x1, y1, w, h) = rect
             x2 = x1 + w
             y2 = y1 + h
             img = img[y1:y2,x1:x2]
             if not os.name == 'nt' and self.crApprox is None:
                 S,(mag,imgx,imgy) = radial_transform(img.astype(np.float32))
                 minV,maxV,minL,maxL = cv2.minMaxLoc(S)
                 self.crApprox = np.vstack([np.array([-20,20])+maxL[0] ,
                                            np.array([-20,20])+maxL[1]])
        d2,d1 = img.shape
        ret,thresh = cv2.threshold(img,self.parameters['threshold'],255,0)
        if not self.crApprox is None:
            minV,maxV,minL,maxL = cv2.minMaxLoc(img[
                self.crApprox[1,0]:self.crApprox[1,1],
                self.crApprox[0,0]:self.crApprox[0,1]])
            maxL = (maxL[0]+self.crApprox[0,0],maxL[1]+self.crApprox[1,0])
        else:
            maxL = (0,0)
        # Closing morphologies
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        im,contours,hierarchy = cv2.findContours(thresh.copy(),cv2.RETR_LIST,
                                                 cv2.CHAIN_APPROX_SIMPLE)
        img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
        thresh = cv2.cvtColor(thresh,cv2.COLOR_GRAY2RGB)
        if not self.crApprox is None:
            img = cv2.circle(img, maxL, 4, (0,0,255), -1)
        # Shape analysis
        area = np.array([cv2.contourArea(c) for c in contours],
                        dtype = np.float32)
        #rect = np.array([cv2.boundingRect(c)[2:] for c in contours],
        #                dtype = np.float32)
        minArea = 200
        maxArea = 10000
        #radius = np.max(rect,axis = 1)/2.
        circleIdx = np.where((area > minArea) & (area < maxArea))[0]
        #(np.abs(1 - rect[:,0]/rect[:,1]) <= 0.5) & 
        #(np.abs(1 - area/(np.pi * (radius**2))) <= 0.8)
        score = np.ones_like(circleIdx,dtype=float)
        # Try to fit the contours
        for e,i in enumerate(circleIdx):
            cX,cY = self.getCenterOfMass(contours[i])
            dist = np.sqrt((cX - d1/2)**2 + (cY - d2/2)**2)
            pts = contours[i][:,0,:]
            distM = np.sqrt((pts[:,0] - cX)**2 + (pts[:,1] - cY)**2)
            mm,ss = (np.median(distM),np.std(distM))
            ptsIdx = (distM<mm+ss*1.5) & (distM>mm-ss*1.5)
            pts = pts[ptsIdx,:]
            (pupil_pos,
             (long_axis,
              short_axis)), (b,a,phi) = fitEllipse(
                  np.fliplr(pts).astype(np.float32))
            if np.sum(np.isfinite([b,a]))==0:
                s1 = self.ellipseToContour(pupil_pos,a,b,phi,R=self.R)
                score[e] = cv2.matchShapes(contours[i],s1,1,0)
            score[e] *= (dist**2)
            img = cv2.drawContours(img,
                                   [contours[i]], -1, (70, 0, 150),1)
            #font = cv2.FONT_HERSHEY_SIMPLEX
            #img = cv2.putText(img,'{0}'.format(round(dist)),
            #          (cX,cY), font, 0.5,(0,0,255),2,cv2.LINE_AA)
        # Get the actual estimate for the contour with best score
        pupil_pos = [0,0]
        (long_axis,short_axis) = [np.nan,np.nan]
        (b,a,phi) = (np.nan,np.nan,0)
        if len(score):
            idx = circleIdx[np.argmin(score)]
            # discard outliers
            cX,cY = self.getCenterOfMass(contours[idx])
            pts = contours[idx][:,0,:]
            dist = np.sqrt((pts[:,0] - cX)**2 + (pts[:,1] - cY)**2)
            mm,ss = (np.median(dist),np.std(dist))
            ptsIdx = (dist<mm+ss*1.5) & (dist>mm-ss*1.5)
            pts = pts[ptsIdx,:]
            # Estimate pupil diam and position
            (pupil_pos,
             (long_axis,short_axis)), (b,a,phi) = fitEllipse(
                 np.fliplr(pts).astype(np.float32))
            img = cv2.drawContours(img,
                                   [contours[idx]], -1, (0, 255, 0),1)
            s1 = ellipseToContour(pupil_pos,a,b,phi,R=self.R)
            img = cv2.drawContours(img,
                                   [s1], -1, (0, 255, 255),2)
            thresh = cv2.drawContours(thresh,
                                      [s1], -1, (0, 255, 255),2)
            # Absolute positions
            pupil_pos[0] += y1 
            pupil_pos[1] += x1
        if self.concatenateBinaryImage:
            img = np.concatenate((img,thresh),axis=0)
        return img,(maxL[1],maxL[0]),(pupil_pos[1],pupil_pos[0]),(short_axis,long_axis),(b,a,phi)

    def getCenterOfMass(self,contour):
        # center of mass
        M = cv2.moments(contour)
        return (int(M["m10"] / M["m00"]),int(M["m01"] / M["m00"]))

    def applyStarburst(self,img):
        img = self.clahe.apply(img.astype('uint8'))
        if len(self.ROIpoints) >= 4:
             pts = np.array(self.ROIpoints).reshape((-1,1,2))
             rect = cv2.boundingRect(pts)
             (x1, y1, w, h) = rect
             x2 = x1 + w
             y2 = y1 + h
             img = img[y1:y2,x1:x2]
        img = cv2.GaussianBlur(img,
                               (self.parameters['gaussian_filterSize'],
                                self.parameters['gaussian_filterSize']),0)
        S,(mag,imgx,imgy) = radial_transform(
            img.astype('float32'))
        pupil_guess,cr_guess = find_minmax(S)#np.where(S == np.min(S))
        (cr_position,
         cr_radius,
         cr_err),(pupil_position,
                  pupil_radius,
                  pupil_err),pupil_ellipse_par = pupil_estimate(mag, cr_guess, pupil_guess)
        (b,a,phi) = pupil_ellipse_par
        s1 = ellipseToContour(pupil_pos,a,b,phi,R=self.R)
        if self.draw:
            img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
            img = cv2.drawContours(img,
                                   [s1], -1, (0, 255, 255),2)
        return img,cr_position,pupil_position,max(pupil_radius),pupil_ellipse_par
