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

def getCenterOfMass(contour):
    # center of mass
    M = cv2.moments(contour)
    return (int(M["m10"] / M["m00"]),int(M["m01"] / M["m00"]))

def extractPupilShapeAnalysis(img,params,
                              ROIpoints = [],
                              expectedDiam = None,
                              expectedPosition = None,
                              clahe = None,
                              drawProcessedFrame = False,
                              concatenateBinaryImage = False):
    # Contrast and filtering
    x1,y1 = (0,0)
    w,h = img.shape
    roiArea = w*h
    if len(ROIpoints) >= 4:
        pts = np.array(ROIpoints).reshape((-1,1,2))
        rect = cv2.boundingRect(pts)
        (x1, y1, w, h) = rect
        x2 = x1 + w
        y2 = y1 + h
        roiArea = w*h
        img = img[y1:y2,x1:x2]
        if params['crApprox'] is None:
            try:
                mag,imgx,imgy = sobel3x3(cv2.GaussianBlur(img,(21,21),100))
                minV,maxV,minL,maxL = cv2.minMaxLoc(cv2.GaussianBlur(mag,(21,21),100))
                params['crApprox'] = np.array([[maxL[0]-int(w*0.05),
                                                maxL[0]+int(w*0.05)],
                                               [maxL[1]-int(h*0.05),
                                                maxL[1]+int(h*0.05)]])
            except Exception as e:
                 print(e)
    d2,d1 = img.shape
    if not params['crApprox'] is None:
        crtmp = img[
            params['crApprox'][1][0]:params['crApprox'][1][1],
            params['crApprox'][0][0]:params['crApprox'][0][1]].copy()
        crtmp = cv2.GaussianBlur(crtmp, (21, 21), 2)
        minV,maxV,minL,maxL = cv2.minMaxLoc(crtmp)
        # Testing the averaging
        maxL = (maxL[0]+params['crApprox'][0][0],
                maxL[1]+params['crApprox'][1][0])
    else:
        maxL = (0,0)
    outimg = img.copy()
    outimg = cv2.cvtColor(outimg,cv2.COLOR_GRAY2RGB)

    # Contrast equalization
    # Gaussian blurring
    img = cv2.GaussianBlur(img,
                           (params['gaussian_filterSize'],
                            params['gaussian_filterSize']),0)
    img = clahe.apply(img)
    # Morphological operations (Open)
    if params['open_kernelSize'] > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                           (params['open_kernelSize'],
                                            params['open_kernelSize']))
        img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    # Morphological operations (Close)
    if params['close_kernelSize'] > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                           (params['close_kernelSize'],
                                            params['close_kernelSize']))
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    # Threshold image
    if params['invertThreshold']:
        if not params['crApprox'] is None:
            tmp = img[params['crApprox'][1][0]:params['crApprox'][1][1],
                      params['crApprox'][0][0]:params['crApprox'][0][1]]
            img[params['crApprox'][1][0]:params['crApprox'][1][1],
                params['crApprox'][0][0]:params['crApprox'][0][1]] = (tmp.astype(
                    np.float32) * (1. - crtmp/float(maxV))).astype(img.dtype)
        ret,thresh = cv2.threshold(img,params['threshold'],255,0)
        thresh = cv2.bitwise_not(thresh)
    else:
        ret,thresh = cv2.threshold(img,params['threshold'],255,0)
    # Find the contours
    im,contours,hierarchy = cv2.findContours(thresh.copy(),cv2.RETR_LIST,
                                             cv2.CHAIN_APPROX_SIMPLE)
    #img = cv2.drawContours(img,
    #                       contours, -1, (20, 0, 250),1)
    # For display only
    if drawProcessedFrame:
        outimg = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
    if not params['crApprox'] is None:
        outimg = cv2.circle(outimg, maxL, 2, (0,0,255), 1)
    # Shape analysis (get the area of each contour)
    area = np.array([cv2.contourArea(c) for c in contours],
                    dtype = np.float32)
    # Discard very large and very small areas.
    minArea = params['minPupilArea']*roiArea
    maxArea = roiArea*params['maxPupilArea']
    circleIdx = np.where((area > minArea) & (area < maxArea))[0]

    #font = cv2.FONT_HERSHEY_SIMPLEX
    #for i,c in enumerate(contours):
    #    try:
    #        cX,cY = self.getCenterOfMass(c)
    #        
    #        img = cv2.putText(img,'{0}'.format(round(area[i])),
    #                          (cX,cY), font, 0.5,(0,0,255),2,cv2.LINE_AA)
    #    except:
    #        print(c)
    score = np.ones_like(circleIdx,dtype=float)
    # Try to fit the contours
    mask = np.zeros(thresh.shape,np.uint8)
    tmpe = np.zeros_like(outimg[:,:,0])
    for e,i in enumerate(circleIdx):
        cX,cY = getCenterOfMass(contours[i])
        dist = np.sqrt((cX - d1/2)**2 + (cY - d2/2)**2)
        pts = contours[i][:,0,:]
        distM = np.sqrt((pts[:,0] - cX)**2 + (pts[:,1] - cY)**2)
        mm,ss = (np.median(distM),np.std(distM))
        ptsIdx = (distM<mm+ss*1.3) & (distM>mm-ss*1.3)
        pts = pts[ptsIdx,:]
        ellipse = cv2.fitEllipse(pts) 
        tmpe[:] = 0
        cv2.ellipse(tmpe,ellipse,255,-1)
        _,econt,_ = cv2.findContours(tmpe,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        if len(econt):
            score[e] = cv2.matchShapes(contours[i],econt[0],2,0.0)
        else:
            score[e] = 1000**2
        # Remove candidates that are close to the corneal reflection center
        #if np.sqrt((maxL[0] - cX)**2 + (maxL[1] - cY)**2) < h*0.03:
        #    dist = 1000
        score[e] *= (dist**2)
        cv2.drawContours(mask,[contours[i]],0,255,-1)
        mean_val = cv2.mean(thresh,mask = mask)
        mask = cv2.drawContours(img,
                                [contours[i]], -1, (70, 0, 150),1)
        # Make it easier to be the pupil if close to the expected location
        if drawProcessedFrame:
            outimg = cv2.drawContours(outimg,
                                      [contours[i]], -1, (70, 0, 150),1)
        # Text?
    # Get the actual estimate for the contour with best score
    pupil_pos = [0,0]
    (long_axis,short_axis) = [np.nan,np.nan]
    (b,a,phi) = (np.nan,np.nan,0)
    thresh = cv2.cvtColor(thresh,cv2.COLOR_GRAY2RGB)
    if len(score):
        # Select the one with the best score
        idx = circleIdx[np.argmin(score)]
        # discard outliers (points that are far from the median)
        cX,cY = getCenterOfMass(contours[idx])
        pts = contours[idx][:,0,:]
        dist = np.sqrt((pts[:,0] - cX)**2 + (pts[:,1] - cY)**2)
        mm,ss = (np.median(dist),np.std(dist))
        ptsIdx = (dist<mm+ss*1.) & (dist>mm-ss*1.)
        pts = pts[ptsIdx,:]
        # Estimate pupil diam and position
        ellipse = cv2.fitEllipse(pts) 
        # is it a circle-ish thing?
        if (ellipse[1][1]/ellipse[1][0]) < 1.4:
            outimg = cv2.drawContours(outimg,
                                        [contours[idx]], -1, (0, 255, 0),1)
            cv2.ellipse(outimg,ellipse,(0,255,255),2,cv2.LINE_AA)
            # Absolute positions
            pupil_pos = np.array([ellipse[0][1],ellipse[0][0]])
            short_axis = ellipse[1][0]
            long_axis = ellipse[1][1]
            phi = ellipse[2]
            pupil_pos[0] += x1
            pupil_pos[1] += y1 
    if concatenateBinaryImage and drawProcessedFrame:
        outimg = np.concatenate((outimg,thresh),axis=0)
    return (outimg,(maxL[0] + x1,
                    maxL[1]+y1),pupil_pos,
            (short_axis/2.,long_axis/2.),
            (short_axis,long_axis,phi))

class MPTracker(object):
    def __init__(self,parameters = None, drawProcessedFrame=False):
        if parameters is None:
            self.parameters = {
                'contrast_clipLimit':10,
                'contrast_gridSize':5,
                'gaussian_filterSize':5,
                'open_kernelSize':0,
                'close_kernelSize':4,
                'threshold':40,
                'minPupilArea': 0.01,
                'maxPupilArea': 0.75,
                'crApprox':None,
                'points':[],
                'invertThreshold':False,
                'eye_radius_mm':2.4, #this was set to 3*0.8 in the matlab version
                'number_frames':0,
            }
        else:
            self.parameters = parameters
        print(self.parameters)
        self.drawProcessedFrame = drawProcessedFrame
        self.set_clhe()
        self.ROIpoints = []
        if 'points' in self.parameters.keys():
            self.setROI(self.parameters['points'])
        self.concatenateBinaryImage=False
    def setROI(self, points):
        self.ROIpoints = points
    def set_clhe(self):
        if (self.parameters['contrast_gridSize']>1):
            self.clahe = cv2.createCLAHE(
                clipLimit=self.parameters['contrast_clipLimit'],
                tileGridSize=(self.parameters['contrast_gridSize'],
                              self.parameters['contrast_gridSize']))
        else:
            class equalizeHist(object):
                def __init__(self,par):
                    self.parameters = par
                    self.gamma = self.parameters['contrast_clipLimit']
                def apply(self,img):
                    if self.gamma>1:
                        return cv2.equalizeHist(img)
                    else:
                        return img
            self.clahe = equalizeHist(self.parameters)

    def apply(self,img):
        res = extractPupilShapeAnalysis(img,
                                        ROIpoints = self.ROIpoints,
                                        params = self.parameters,
                                        clahe = self.clahe,
                                        concatenateBinaryImage = self.concatenateBinaryImage,
                                        drawProcessedFrame=self.drawProcessedFrame)
        self.img = res[0]
        return res[1:]


'''    def applyStarburst(self,img):
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
        return img,cr_position,pupil_position[::-1],max(pupil_radius),pupil_ellipse_par
'''
