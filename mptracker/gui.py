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
                                 QGroupBox,
                                 QFileDialog)
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
    def __init__(self,targetpath = None,resfile = None, app = None):
        super(MPTrackerWindow,self).__init__()
        self.app = app
        if targetpath is None:
            self.targetpath = QFileDialog(self).getOpenFileName()[0]
            if not os.path.isfile(self.targetpath):
                print('Selected non file.')
                sys.exit()
        else:
            self.targetpath = os.path.abspath(targetpath)    
        self.imgstack = TiffFileSequence(self.targetpath)
        self.tracker = MPTracker()
        self.parameters = self.tracker.parameters
        self.parameters['number_frames'] = self.imgstack.nFrames
        self.parameters['points'] = []
        self.resultfile = resfile
        self.results = {}
        self.results['ellipsePix'] = np.empty((self.parameters['number_frames'],5),
                                           dtype = np.float32)
        self.results['ellipsePix'].fill(np.nan)
        self.results['pupilPix'] = np.empty((self.parameters['number_frames'],2),
                                            dtype=np.int)
        self.results['pupilPix'].fill(np.nan)
        self.results['crPix'] = np.empty((self.parameters['number_frames'],2),
                                         dtype = np.int)
        self.results['crPix'].fill(np.nan)

        self.initUI()
        
    def initUI(self):
        grid = QGridLayout()
        paramGrid = QFormLayout()
        paramGroup = QGroupBox()
        
        
        paramGroup.setTitle("Eye tracking parameters")
        
        paramGroup.setLayout(paramGrid)
        self.setLayout(grid)
        

        self.wContrastLim = QSlider(Qt.Horizontal)
        self.wContrastLim.setValue(15)
        self.wContrastLim.setMinimum(0)
        self.wContrastLim.setMaximum(200)
        self.wContrastLimLabel = QLabel('Contrast limit [{0}]:'.format(
            self.wContrastLim.value()))
        self.wContrastLim.valueChanged.connect(self.setContrastLim)
        paramGrid.addRow(self.wContrastLimLabel,self.wContrastLim)

        self.wContrastGridSize = QSlider(Qt.Horizontal)
        self.wContrastGridSize.setValue(5)
        self.wContrastGridSize.setMaximum(100)
        self.wContrastGridSize.setMinimum(3)
        self.wContrastGridSizeLabel = QLabel('Contrast grid size [{0}]:'.format(
            self.wContrastGridSize.value()))
        self.wContrastGridSize.valueChanged.connect(self.setContrastGridSize)
        paramGrid.addRow(self.wContrastGridSizeLabel,self.wContrastGridSize)

        self.wGaussianFilterSize = QSlider(Qt.Horizontal)
        self.wGaussianFilterSize.setValue(7)
        self.wGaussianFilterSize.setMaximum(31)
        self.wGaussianFilterSize.setMinimum(1)
        self.wGaussianFilterSize.setSingleStep(2)
        self.wGaussianFilterSizeLabel = QLabel('Gaussian filter [{0}]:'.format(
            self.wGaussianFilterSize.value()))
        self.wGaussianFilterSize.valueChanged.connect(self.setGaussianFilterSize)
        paramGrid.addRow(self.wGaussianFilterSizeLabel, self.wGaussianFilterSize)

        self.wBinThreshold = QSlider(Qt.Horizontal)
        self.wBinThreshold.setValue(40)
        self.wBinThresholdLabel = QLabel('Binary contrast [40]:')
        self.wBinThreshold.setMinimum(0)
        self.wBinThreshold.setMaximum(255)
        self.wBinThreshold.valueChanged.connect(self.setBinThreshold)
        paramGrid.addRow(self.wBinThresholdLabel,self.wBinThreshold)
        
        self.wEyeRadius = QTextEdit('')
        self.wEyeRadius.setMaximumHeight(25)
        self.wEyeRadius.setMaximumWidth(40)
        self.wEyeRadius.textChanged.connect(self.setEyeRadius)
        paramGrid.addRow(QLabel('Approximate eye radius (mm):'),self.wEyeRadius)
        
        self.wNFrames = QLabel('')
        self.wNFrames.setMaximumHeight(25)
        self.wNFrames.setMaximumWidth(200)
        paramGrid.addRow(QLabel('Number of frames:'),self.wNFrames)

        # parameters, buttons and options
        self.wPoints = QTextEdit('')
        self.wPoints.setMaximumHeight(25)
        self.wPoints.setMaximumWidth(200)
        paramGrid.addRow(QLabel('Points:'),self.wPoints)

        self.wDisplayBinaryImage = QCheckBox()
        self.wDisplayBinaryImage.setChecked(False)
        self.wDisplayBinaryImage.stateChanged.connect(self.updateTrackerOutputBinaryImage)
        paramGrid.addRow(QLabel('Display binary image:'),self.wDisplayBinaryImage)
        
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

    def updateTrackerOutputBinaryImage(self,state):
        self.tracker.concatenateBinaryImage = state
        self.processFrame(self.wFrame.value())

    def setBinThreshold(self,value):
        self.parameters['threshold'] = int(value)
        self.wBinThresholdLabel.setText('Binary contrast [{0}]:'.format(int(value)))
        self.processFrame(self.wFrame.value())

    def setContrastLim(self,value):
        self.parameters['contrast_clipLimit'] = int(value)
        self.wContrastLimLabel.setText('Contrast limit [{0}]:'.format(int(value)))
        self.tracker.set_clhe()
        self.processFrame(self.wFrame.value())

    def setContrastGridSize(self,value):
        self.parameters['contrast_gridSize'] = int(value)
        self.wContrastGridSizeLabel.setText('Contrast grid size [{0}]:'.format(int(value)))
        self.tracker.set_clhe()
        self.processFrame(self.wFrame.value())

    def setGaussianFilterSize(self,value):
        if not np.mod(value,2) == 1:
            value += 1
        
        self.parameters['gaussian_filterSize'] = int(value)
        self.wGaussianFilterSizeLabel.setText('Gaussian filter size [{0}]:'.format(int(value)))
        self.processFrame(self.wFrame.value())
        
    def setEyeRadius(self):
        value = self.wEyeRadius.toPlainText()
        try:
            self.parameters['eye_radius_mm'] = float(value)
            print(self.tracker.parameters['eye_radius_mm'])
        except:
            print('Need to insert a float in the radius.')
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

    def updateGUI(self,value=0):
        if not self.parameters['points'] is None:
            self.wPoints.setText(' '.join(
                [str(p) for p in self.parameters['points']]))
        self.wEyeRadius.setText(str(self.parameters['eye_radius_mm']))
        self.wNFrames.setText(str(self.parameters['number_frames']))
        self.wContrastLim.setValue(int(self.parameters['contrast_clipLimit']))
        self.wContrastGridSize.setValue(
            int(self.parameters['contrast_gridSize']))
        self.wGaussianFilterSize.setValue(
            int(self.parameters['gaussian_filterSize']))

    # Update
    def processFrame(self,val = 0):
        f = int(val)
        img = self.imgstack.get(f)
        img,cr_pos,pupil_pos,pupil_radius,pupil_ellipse_par = self.tracker.apply(img)
        self.results['ellipsePix'][f,:2] = pupil_radius
        self.results['ellipsePix'][f,2:] = pupil_ellipse_par
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
        if self.resultfile is None:
            self.resultfile = QFileDialog().getSaveFileName()[0]
        if not os.path.isfile(self.resultfile):
            fd = createResultsFile(self.resultfile,
                                   self.parameters['number_frames'])
            fd['ellipsePix'][:] = self.results['ellipsePix']
            fd['positionPix'][:] = self.results['pupilPix']
            fd['crPix'][:] = self.results['crPix']
            diam = computePupilDiameterFromEllipse(self.results['ellipsePix'],
                                                   computeConversionFactor(
                                                       self.results['reference']))
            az,el,theta = convertPixelToEyeCoords(self.results['pupilPix'],
                                                  self.results['reference'],
                                                  self.results['crPix'])
            fd['diameter'][:] = diam
            fd['azimuth'][:] = az
            fd['elevation'][:] = el
            fd['theta'][:] = theta
            fd['pointsPix'][:] = np.array(self.parameters['points'])
            fd.close()
            print("Saved to " + self.resultfile)
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
    w = MPTrackerWindow(target,app = app)
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
