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
                                 QTableWidget,
                                 QFileDialog)
    from PyQt5.QtGui import QImage, QPixmap,QBrush,QPen,QColor
    from PyQt5.QtCore import Qt,QSize,QRectF,QLineF,QPointF
except:
    from PyQt4.QtGui import (QWidget,
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
                             QTableWidget,
                             QFileDialog,
                             QImage,
                             QPixmap)
    from PyQt4.QtCore import Qt,QSize,QRectF,QLineF,QPointF
   
import pylab as plt
plt.matplotlib.style.use('ggplot')

# Local imports
from .utils import *
from .io import *
from .tracker import *

description = ''' GUI to define parameters and track the pupil.'''

class MPTrackerWindow(QWidget):
    def __init__(self,targetpath = None,
                 resfile = None, 
                 app = None, usetmp = False):
        super(MPTrackerWindow,self).__init__()
        self.app = app
        if targetpath is None:
            self.targetpath = QFileDialog(self).getOpenFileName()[0]
            if not os.path.isfile(self.targetpath):
                print('Selected non file:'+str(self.targetpath))
                sys.exit()
        else:
            self.targetpath = os.path.abspath(targetpath)
        if usetmp:
            self.tmptarget = copyFilesToTmp(self.targetpath)
            print('WARNING: At exit is not implemented yet to delete this folder. User is responsible for that.')
            target = self.tmptarget
        else:
            target = self.targetpath
        if os.path.splitext(target)[1] in ['.tif','.tiff']:
            self.imgstack = TiffFileSequence(target)
        elif os.path.splitext(self.targetpath)[1] in ['.seq']:
            self.imgstack = NorpixFile(target)
        else:
            print('Unknown extension for:'+target)
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
        self.wContrastGridSize.setValue(7)
        self.wContrastGridSize.setMaximum(200)
        self.wContrastGridSize.setMinimum(1)
        self.wContrastGridSizeLabel = QLabel('Contrast grid size [{0}]:'.format(
            self.wContrastGridSize.value()))
        self.wContrastGridSize.valueChanged.connect(self.setContrastGridSize)
        paramGrid.addRow(self.wContrastGridSizeLabel,self.wContrastGridSize)

        self.wGaussianFilterSize = QSlider(Qt.Horizontal)
        self.wGaussianFilterSize.setValue(3)
        self.wGaussianFilterSize.setMaximum(61)
        self.wGaussianFilterSize.setMinimum(1)
        self.wGaussianFilterSize.setSingleStep(2)
        self.wGaussianFilterSizeLabel = QLabel('Gaussian filter [{0}]:'.format(
            self.wGaussianFilterSize.value()))
        self.wGaussianFilterSize.valueChanged.connect(self.setGaussianFilterSize)
        paramGrid.addRow(self.wGaussianFilterSizeLabel, self.wGaussianFilterSize)

        self.wOpenKernelSize = QSlider(Qt.Horizontal)
        self.wOpenKernelSize.setValue(0)
        self.wOpenKernelSize.setMaximum(61)
        self.wOpenKernelSize.setMinimum(0)
        self.wOpenKernelSize.setSingleStep(1)
        self.wOpenKernelSizeLabel = QLabel('Morph open size [{0}]:'.format(
            self.wOpenKernelSize.value()))
        self.wOpenKernelSize.valueChanged.connect(self.setOpenKernelSize)
        paramGrid.addRow(self.wOpenKernelSizeLabel, self.wOpenKernelSize)

        self.wCloseKernelSize = QSlider(Qt.Horizontal)
        self.wCloseKernelSize.setValue(3)
        self.wCloseKernelSize.setMaximum(61)
        self.wCloseKernelSize.setMinimum(0)
        self.wCloseKernelSize.setSingleStep(1)
        self.wCloseKernelSizeLabel = QLabel('Morph close size [{0}]:'.format(
            self.wCloseKernelSize.value()))
        self.wCloseKernelSize.valueChanged.connect(self.setCloseKernelSize)
        paramGrid.addRow(self.wCloseKernelSizeLabel, self.wCloseKernelSize)

        
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
        self.wPoints = QLabel('nan,nan \n nan,nan \n nan,nan \n nan,nan\n')
        paramGrid.addRow(QLabel('ROI points:'),self.wPoints)
        self.wPoints.mouseDoubleClickEvent = self.clearPoints
        self.wDisplayBinaryImage = QCheckBox()
        self.wDisplayBinaryImage.setChecked(False)
        self.wDisplayBinaryImage.stateChanged.connect(self.updateTrackerOutputBinaryImage)
        paramGrid.addRow(QLabel('Display binary image:'),self.wDisplayBinaryImage)
        self.wInvertThreshold = QCheckBox()
        self.wInvertThreshold.setChecked(False)
        self.wInvertThreshold.stateChanged.connect(self.setInvertThreshold)
        paramGrid.addRow(QLabel('White pupil:'),self.wInvertThreshold)
        
        grid.addWidget(paramGroup,0,0,3,1)
        
        self.wFrame = QSlider(Qt.Horizontal)
        self.wFrame.valueChanged.connect(self.processFrame)
        self.wFrame.setMaximum(self.imgstack.nFrames-1)
        self.wFrame.setMinimum(0)
        self.wFrame.setValue(0)
        grid.addWidget(self.wFrame,0,2,1,4)
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
        self.setWindowTitle('mOUSEpUPILtracker')
        self.show()
        self.updateGUI()
        self.running = False

    def clearPoints(self,event):
        self.tracker.ROIpoints = []
        self.putPoints()
        
    def putPoints(self):
        points = self.tracker.ROIpoints
        self.wPoints.setText(' \n'.join([','.join([str(w) for w in p]) for p in points]))
        
    def updateTrackerOutputBinaryImage(self,state):
        self.tracker.concatenateBinaryImage = state
        self.processFrame(self.wFrame.value())

    def setInvertThreshold(self,value):
        self.parameters['invertThreshold'] = value
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

    def setCloseKernelSize(self,value):
        self.parameters['close_kernelSize'] = int(value)
        self.wCloseKernelSizeLabel.setText('Morph close size [{0}]:'.format(int(value)))
        self.processFrame(self.wFrame.value())

    def setOpenKernelSize(self,value):
        self.parameters['open_kernelSize'] = int(value)
        self.wOpenKernelSizeLabel.setText('Morph open size [{0}]:'.format(int(value)))
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
        self.putPoints()
        
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
#        if not self.parameters['points'] is None:
#            self.wPoints.setText(' '.join(
#                [str(p) for p in self.parameters['points']]))
        self.wEyeRadius.setText(str(self.parameters['eye_radius_mm']))
        self.wNFrames.setText(str(self.parameters['number_frames']))
        self.wContrastLim.setValue(int(self.parameters['contrast_clipLimit']))
        self.wContrastGridSize.setValue(
            int(self.parameters['contrast_gridSize']))
        self.wGaussianFilterSize.setValue(
            int(self.parameters['gaussian_filterSize']))
        self.wOpenKernelSize.setValue(int(self.parameters['open_kernelSize']))
        self.wCloseKernelSize.setValue(int(self.parameters['close_kernelSize']))
    # Update
    def processFrame(self,val = 0):
        f = int(val)
        try:
            img = self.imgstack.get(f)
        except:
            img = None
            print("Failed loading frame {0}".format(f))
        if not img is None:
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
            results = self.results.copy()
            clahe = cv2.createCLAHE(7,(10,10))
            ii = 10
            img = clahe.apply(self.imgstack.get(ii))
            fig = plt.figure(figsize = [10,3])
            ax = fig.add_axes([0.025,0.05,0.25,0.95],aspect='equal')
            ax.imshow(img,cmap='gray',aspect='equal')
            if not 'reference' in results.keys():
                results['reference'] = [self.parameters['points'][0],self.parameters['points'][2]]
            eyeCorners  = results['reference']
            reference = [eyeCorners[0][0] + np.diff([eyeCorners[0][0],eyeCorners[1][0]])/2.,
                         eyeCorners[0][1] + np.diff([eyeCorners[0][1],eyeCorners[1][1]])/2.]
            ax.plot(reference[0],reference[1],'g+',alpha=0.8,markersize=10,lw=1)
            ax.plot([results['reference'][0][0],results['reference'][1][0]],
                    [results['reference'][0][1],results['reference'][1][1]],'-|y',
                    alpha=0.8,markersize=25,lw=1)
            ax.plot(results['pupilPix'][ii,0],results['pupilPix'][ii,1],'r.',alpha=0.8)
            ax.plot(results['crPix'][ii,0],results['crPix'][ii,1],'bo',alpha=0.8)            
            s1 = ellipseToContour(results['pupilPix'][ii,:],results['ellipsePix'][ii,2],
                                  results['ellipsePix'][ii,3],
                                  results['ellipsePix'][ii,4],np.linspace(0,2*np.pi,200))

            ax.plot(np.hstack([s1[:,0,1],s1[0,0,1]]),
                    np.hstack([s1[:,0,0],s1[0,0,0]]),'-',color='orange',alpha=0.8)
            ax.grid('off')
            ax.axis('off');ax.axis('tight');

            axel = fig.add_axes([0.36,0.16,0.6,0.2])
            axdiam = fig.add_axes([0.36,0.76,0.6,0.2]) #,sharex=axel
            axaz = fig.add_axes([0.36,0.46,0.6,0.2])

            diam = computePupilDiameterFromEllipse(results['ellipsePix'],
                                                   computeConversionFactor(results['reference']))
            az,el,theta = convertPixelToEyeCoords(results['pupilPix'],
                                                  results['reference'],results['crPix'])
            
            axdiam.plot(medfilt(diam));axdiam.set_xticklabels([])
            axaz.plot(medfilt(az));axaz.set_xticklabels([]);

            axaz.set_ylabel('Azimuth \n [deg]',color='black')
            axel.plot(medfilt(el));axel.set_ylabel('Elevation \n [deg]',color='black')

            axdiam.set_ylabel('Diameter \n [mm]',color='black')
            
            def cleanAx(ax1):
                ax1.locator_params(axis='y',nbins=3)
                ax1.spines['right'].set_visible(False)
                ax1.spines['top'].set_visible(False)
                # Only show ticks on the left and bottom spines
                ax1.yaxis.set_ticks_position('left')
                ax1.xaxis.set_ticks_position('bottom')
                ax1.spines['bottom'].set_color('black')
                ax1.spines['left'].set_color('black')
                ax1.tick_params(axis='both', colors='black')
            for a in [axdiam,axaz,axel]:
                cleanAx(a)
                a.axis('tight')
            axel.set_ylim(np.array([-0.25,0.25]) + np.nanmedian(el))
            axaz.set_ylim(np.array([-1,1]) + np.nanmedian(az))
            axdiam.set_ylim(np.array([0,1 + np.nanmedian(diam)]))
#            axdiam.set_ylim([0,2.])
#            axaz.set_ylim([0,3.7])
            axel.set_xlabel('Frame number',color='black')
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
            try:
                self.resultfile = QFileDialog().getSaveFileName()
            except:
                self.resultfile = ''
        self.resultfile = str(self.resultfile)
        print self.resultfile
        fname,ext = os.path.splitext(self.resultfile)
        if len(ext)==0:
            print('File has no extension:'+self.resultfile)
            return
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
        else:
            print("File already exists.")
def main():
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('target',
                        metavar = 'target',
                        type = str,
                        help = 'Experiment data path.')
    parser.add_argument('-o','--output',
                        type = str,
                        default=None,
                        help = 'Output data path.')
    parser.add_argument('--usetmp',
                        default = False,
                        action = 'store_true',
                        help = 'Copy data to a temporary folder.')
    args = parser.parse_args()

    app = QApplication(sys.argv)
    target = None
    if os.path.isfile(args.target):
        target = args.target
    w = MPTrackerWindow(target,app = app,
                        resfile = args.output,
                        usetmp=args.usetmp)
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
