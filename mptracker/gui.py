#!/usr/bin python
# GUI to define parameters and track the mouse pupil
import sys
import os
import numpy as np
import time
import atexit
import argparse
from glob import glob
from time import sleep
import cv2
#import matplotlib.pyplot as plt
# Qt imports
try:
    from PyQt5.QtWidgets import (QWidget,
                                 QApplication,
                                 QGridLayout,
                                 QFormLayout,
                                 QCheckBox,
                                 QTextEdit,
                                 QSlider,
                                 QLabel,
                                 QGraphicsView,
                                 QGraphicsScene,
                                 QGraphicsItem,
                                 QGraphicsLineItem,
                                 QGroupBox)
    from PyQt5.QtGui import QImage, QPixmap,QBrush,QPen,QColor
    from PyQt5.QtCore import Qt,QSize,QRectF,QLineF,QPointF
except:
    print('PyQt5 not installed??')
import pylab as plt
# Local imports
from .utils import *
from .io import *
from .tracker import *

description = ''' GUI to define parameters and track the pupil.'''

class MPTrackerWindow(QWidget):
    def __init__(self,targetpath = None,app = None):
        super(MPTrackerWindow,self).__init__()
        self.app = app
        if targetpath is None:
            print('No target path specified... in the future this will ask the filename.')
            tiffiles = np.sort(glob(os.environ['HOME']+
                                    '/temp/develop/pupil_tracking_mice'+
                                    '/160927_JC021_lgnmov'+
                                    '/*.tif'))
            targetpath = tiffiles[0]
        self.imgstack = TiffFileSequence(targetpath)
        self.tracker = MPTracker()
        self.parameters = self.tracker.parameters
        self.parameters['number_frames'] = self.imgstack.nFrames
        self.parameters['points'] = []
        self.initUI()
        
    def initUI(self):
        grid = QGridLayout()
        paramGrid = QFormLayout()
        paramGroup = QGroupBox()
        paramGroup.setTitle("Eye tracking parameters")
        
        paramGroup.setLayout(paramGrid)
        self.setLayout(grid)
        # parameters, buttons and options
        self.wRegion = QTextEdit('')
        self.wRegion.setMaximumHeight(25)
        self.wRegion.setMaximumWidth(200)
        paramGrid.addRow(QLabel('Region:'),self.wRegion)

        self.wGaussianFilterSize = QTextEdit()
        self.wGaussianFilterSize.setMaximumHeight(25)
        self.wGaussianFilterSize.setMaximumWidth(40)
        paramGrid.addRow(QLabel('Gaussian filter:'),self.wGaussianFilterSize)

        self.wContrastGridSize = QTextEdit()
        self.wContrastGridSize.setMaximumHeight(25)
        self.wContrastGridSize.setMaximumWidth(40)
        paramGrid.addRow(QLabel('Contrast grid size:'),self.wContrastGridSize)

        self.wContrastLim = QTextEdit('')
        self.wContrastLim.setMaximumHeight(25)
        self.wContrastLim.setMaximumWidth(40)
        paramGrid.addRow(QLabel('Contrast limit:'),self.wContrastLim)

        self.wEyeRadius = QTextEdit('')
        self.wEyeRadius.setMaximumHeight(25)
        self.wEyeRadius.setMaximumWidth(40)
        paramGrid.addRow(QLabel('Approximate eye radius (mm):'),self.wEyeRadius)

        self.wNFrames = QLabel('')
        self.wNFrames.setMaximumHeight(25)
        self.wNFrames.setMaximumWidth(200)

        paramGrid.addRow(QLabel('Number of frames:'),self.wNFrames)

        grid.addWidget(paramGroup,0,0,3,1)
        
        self.wFrame = QSlider(Qt.Horizontal)
        self.wFrame.valueChanged.connect(self.processFrame)
        self.wFrame.setMaximum(self.imgstack.nFrames-1)
        self.wFrame.setMinimum(0)
        self.wFrame.setValue(0)
        grid.addWidget(self.wFrame,0,2,1,3)
        # images and plots
        img = self.imgstack.get(int(self.wFrame.value()))
        img,cr_position,pupil_pos,pupil_radius,pupil_ellipse_par = self.tracker.apply(img)
        self.scene = QGraphicsScene()
        self.view = QGraphicsView(self.scene)
        self.setImage(img)
        self.view.mouseReleaseEvent = self.selectPoints
        grid.addWidget(self.view,1,2,6,5)
        ####################
        # window geometry
#        self.setGeometry(100, 100, img.shape[1]*0.8, img.shape[0]*0.8)
        self.setWindowTitle('mOUSEpUPILtracker')
        self.show()
        self.updateGUI()
        self.running = False

    def selectPoints(self,event):
        pt = self.view.mapToScene(event.pos())
        x = pt.x()
        y = pt.y()
        self.parameters['points'].append([int(round(x)),int(round(y))])
        img = self.imgstack.get(int(self.wFrame.value()))
        height,width = img.shape
        self.tracker.setROI(self.parameters['points'])
        img,cr_position,pupil_pos,pupil_radius,pupil_ellipse_par = self.tracker.apply(img)
        self.setImage(img)
        
    def setImage(self,image):
        self.scene.clear()
        frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Draw region of interest
        #pts = self.parameters['points']
        #if len(self.parameters['points']) > 2:
        #    pts = np.array(pts).reshape((-1,1,2))
        #cv2.polylines(frame,[pts],True,(0,255,255))
        self.qimage = QImage(frame, frame.shape[1], frame.shape[0], 
                             frame.strides[0], QImage.Format_RGB888)
        self.scene.addPixmap(QPixmap.fromImage(self.qimage))
        self.view.fitInView(QRectF(0,0,
                                   frame.shape[1],
                                   frame.shape[0]),
                            Qt.KeepAspectRatio)
        self.scene.update()

    def updateGUI(self):
        if not self.parameters['region'] is None:
            self.wRegion.setText(' '.join(
                [str(p) for p in self.parameters['region']]))
        self.wEyeRadius.setText(str(self.parameters['eye_radius_mm']))
        self.wNFrames.setText(str(self.parameters['number_frames']))
        self.wContrastLim.setText(str(self.parameters['contrast_clipLimit']))
        self.wContrastGridSize.setText(
            str(self.parameters['contrast_gridSize']))
        self.wGaussianFilterSize.setText(
            str(self.parameters['gaussian_filterSize']))
        self.results = {}
        self.results['diamPix'] = np.empty((self.parameters['number_frames'],5))
        self.results['diamPix'].fill(np.nan)
        self.results['pupilPix'] = np.empty((self.parameters['number_frames'],2))
        self.results['pupilPix'].fill(np.nan)
        self.results['crPix'] = np.empty((self.parameters['number_frames'],2))
        self.results['crPix'].fill(np.nan)

    # Update
    def processFrame(self,val = 0):
        f = int(val)
        img = self.imgstack.get(f)
        img,cr_pos,pupil_pos,pupil_radius,pupil_ellipse_par = self.tracker.apply(img)
        self.results['diamPix'][f,:2] = pupil_radius
        self.results['diamPix'][f,2:] = pupil_ellipse_par
        self.results['pupilPix'][f,:] = pupil_pos
        self.results['crPix'][f,:] = cr_pos
        self.setImage(img)
        self.app.processEvents()
        
    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Escape:
            self.close()
        elif e.key() == 80:
            plt.ion()
            fig = plt.figure()
            fig.add_subplot(3,1,1)
            plt.plot(self.results['diamPix'][:,1],'k')
            plt.plot(medfilt(self.results['diamPix'][:,1]),'b')
            fig.add_subplot(3,1,2)
            plt.plot(self.results['pupilPix'][:,0])
            plt.plot(self.results['pupilPix'][:,1])
            fig.add_subplot(3,1,3)
            plt.plot(self.results['crPix'][:,0])
            plt.plot(self.results['crPix'][:,1])
            plt.show()
        elif e.key() == 82:
            if not self.running:
                self.runDetectionVerbose()
            else:
                self.running = False
        else:
            print e.key()

    def runDetectionVerbose(self):
        self.running = True
        ts = time.time()
        if not len(self.parameters['points']) == 4:
            print('You did not specify the region..')
            return
        for f in xrange(self.parameters['number_frames']):
            self.wFrame.setValue(f)
            if not self.running:
                break
        self.results['reference'] = [self.parameters['points'][0],self.parameters['points'][2]]
        print('Done {0} frames in {1:3.1f} min'.format(f,
                                                       (time.time()-ts)/60.))
        import cPickle as pickle
        pickle.dump(self.results, open( "tmp_results.p", "wb" ) )
        print("Saved to tmp_results.p")
def main():
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('target',
                        metavar = 'target',
                        type = str,
                        help = 'Target experiment to process (path).')
    args = parser.parse_args()

    app = QApplication(sys.argv)
    target = None
    if os.path.isfile(args.target):
        target = args.target
    w = MPTrackerWindow(target,app)
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
