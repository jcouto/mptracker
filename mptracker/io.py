#! /usr/bin python
# IO utilities for mptracker.
# Aim is to know the number of frames in advance and have a common interface for asking image frames
# Supported formats:
#    - multipage TIFF
#    - streamPIX seq files
#    - avi files (on demmand)
# November 2016 - Joao Couto

import sys
import os
import numpy as np
from glob import glob
from tifffile import TiffFile
from .norpix import SeqFile

class TiffFileSequence(object):
    def __init__(self,targetpath = None,extension='tif'):
        '''Lets you access a sequence of TIFF files without noticing...'''
        self.path = os.path.dirname(targetpath)
        self.filenames = np.sort(glob(targetpath + '*.' + extension))
        if not len(self.filenames):
            print('Wrong target path: ' + targetpath + '*.' + extension)
            raise
        self.files = [TiffFile(f) for f in self.filenames]
        framesPerFile = []
        for f in self.files:
            N,h,w = f.series[0].shape
            framesPerFile.append(N)
            if 'h' in dir(self):
                if not self.h == h:
                    print('Wrong height value on one of the files.')
                    raise
            else:
                self.h = h
                self.w = w
        self.framesPerFile = np.array(framesPerFile)
        self.framesOffset = np.hstack([0,np.cumsum(self.framesPerFile[:-1])])
        self.nFrames = sum(framesPerFile)

    def getFrameIndex(self,frame):
        '''Computes the frame index from multipage tiff files.'''
        fileidx = np.where(self.framesOffset <= frame)[0][-1]
        return fileidx,frame - self.framesOffset[fileidx]
        
    def getDescripion(self,frame):
        '''Gets image description tag from tiff page'''
        frameidx = self.getFrameIndex(frame)
        tags = self.files[fileidx][frameidx].tags
        if 'image_description' in tags.keys():
            return tags['image_description']
        else:
            return None

    def get(self,frame):
        '''Returns an image given the frame ID.
        Useful attributes are nFrames, h (frame height) and w (frame width)
        '''
        fileidx,frameidx = self.getFrameIndex(frame)
        return self.files[fileidx].asarray(frameidx)

    def close(self):
        for fd in self.files:
            fd.close()

class NorpixFile(object):
    def __init__(self,targetpath = None,extension='tif'):
        '''Wrapper to norpix seq files'''
        self.path = os.path.dirname(targetpath)
        self.filenames = [self.targetpath]
        self.files = [SeqFile(f) for f in self.filenames]
        framesPerFile = []
        for f in self.files:
            N = len(f)
            framesPerFile.append(N)
            if not 'h' in dir(self):
                h,w = (f.height, f.width)
                self.h = h
                self.w = w
        self.framesPerFile = np.array(framesPerFile)
        self.framesOffset = np.hstack([0,np.cumsum(self.framesPerFile[:-1])])
        self.nFrames = sum(framesPerFile)

    def getFrameIndex(self,frame):
        '''Computes the frame index from multipage tiff files.'''
        fileidx = np.where(self.framesOffset <= frame)[0][-1]
        return fileidx,frame - self.framesOffset[fileidx]
        
    def get(self,frame):
        '''Returns an image given the frame ID.
        Useful attributes are nFrames, h (frame height) and w (frame width)
        '''
        fileidx,frameidx = self.getFrameIndex(frame)
        return self.files[fileidx][frameidx]

    def close(self):
        for fd in self.files:
            fd.close()
