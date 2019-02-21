import cv2
from tifffile import imread
from multiprocessing import Pool,cpu_count
from functools import partial
from time import time as tic
import numpy as np

def process_tiff(fname,parameters = None):
    '''
Process a tiff file with mptracker given parameters
    '''
    from .tracker import MPTracker
    cv2.setNumThreads(0)
    tracker = MPTracker(parameters)
    dat = imread(fname)
    if len(dat.shape) < 3:
        dat = [dat]
    res = []
    for frame in dat:
        res.append(tracker.apply(frame))
    del tracker
    return res


def par_process_tiff(filenames, parameters,
                     verbose = True,
                     nprocesses = int(cpu_count())):
    '''
Process a set of tiff files in parallel given tracker parameters
    '''
    if verbose:
        ts = tic()
        print('\t - Processing {0} files.'.format(len(filenames)))
        
    with Pool(nprocesses) as pool:
        res = pool.map(partial(
            process_tiff,
            parameters = parameters),
                       filenames)
    res = np.vstack(res)
    if verbose:
        toc = tic() - ts
        print('\t - Analysed {0} frames in {1:.1f} s [{2:4.0f} fps]'.format(
            len(res), toc, len(res)/toc))
    return res
