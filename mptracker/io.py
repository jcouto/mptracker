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
from os.path import join as pjoin
import numpy as np
from glob import glob
from tifffile import TiffFile
from .norpix import SeqFile
import h5py as h5

def createResultsFile(filename,nframes,MPIO = False):
    if MPIO:
        f = h5.File(filename, 'w', driver='mpio', comm=MPI.COMM_WORLD)
    else:
        f = h5.File(filename, 'w')
    f.create_dataset('diameter',dtype = np.float32,
                     shape=(nframes,),compression = 'gzip')
    f.create_dataset('azimuth',dtype = np.float32,
                     shape=(nframes,),compression = 'gzip')
    f.create_dataset('elevation',dtype = np.float32,
                     shape=(nframes,),compression = 'gzip')
    f.create_dataset('theta',dtype = np.float32,
                     shape=(nframes,),compression = 'gzip')
    f.create_dataset('ellipsePix',dtype = np.float32,
                     shape=(nframes,5),compression = 'gzip')
    f.create_dataset('positionPix',dtype = np.int,
                     shape=(nframes,2),compression = 'gzip')
    f.create_dataset('crPix',dtype = np.int,
                     shape=(nframes,2),compression = 'gzip')
    f.create_dataset('pointsPix',dtype = np.int,
                     shape=(4,2),compression = 'gzip')
    return f

class TiffFileSequence(object):
    def __init__(self,targetpath = None,extension='tif'):
        '''Lets you access a sequence of TIFF files without noticing...'''
        self.path = os.path.dirname(targetpath)
        self.basename = os.path.splitext(os.path.basename(targetpath))[0]
        for f in range(len(self.basename)):
            if not self.basename[-f].isdigit():
                break
        if not -f+1 == 0:
            f = -f+1
        else:
            f = -1
        self.basename = self.basename[:f]
        filtered_filenames = []
        filenames = np.sort(glob(pjoin(self.path,'*.' + extension)))
        self.filenames = []
        for f in filenames:
            if self.basename in f:
                self.filenames.append(f)
        if not len(self.filenames):
            print('Wrong target path: ' + self.path + '*.' + extension)
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
