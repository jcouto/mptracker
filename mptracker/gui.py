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
                                 QPushButton,
                                 QLabel,
                                 QGraphicsView,
                                 QGraphicsScene,
                                 QGraphicsItem,
                                 QGraphicsLineItem,
                                 QGroupBox,
                                 QDockWidget,
                                 QVBoxLayout,
                                 QMainWindow,
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
                             QPushButton,
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
from .widgets import MptrackerParameters,MptrackerDisplay
from .tracker import cropImageWithCoords

description = ''' GUI to define parameters and track the pupil.'''

class MPTrackerWindow(QMainWindow):
    def __init__(self,targetpath = None,
                 resfile = None,
                 params = None,
                 app = None,
                 usetmp = False):
        super(MPTrackerWindow,self).__init__()
        self.app = app
        if targetpath is None:
            self.targetpath = QFileDialog(self).getOpenFileName()[0]
            if not os.path.isfile(self.targetpath) or not '*' in self.targetpath:
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
        elif os.path.splitext(self.targetpath)[1] in ['.avi']:
            self.imgstack =  AVIFileSequence(target)
        else:
            print('Unknown extension for:'+target)
        self.tracker = MPTracker(parameters = params,
                                 drawProcessedFrame = True)
        self.tracker.apply(self.imgstack.get(0))
        self.unet_data = None
        self.parameters = self.tracker.parameters
        self.parameters['number_frames'] = self.imgstack.nFrames
        self.parameters['crTrack'] = True
        self.parameters['sequentialCRMode'] = False
        self.parameters['sequentialPupilMode'] = False
        self.resultfile = resfile
        self.results = {}
        self.results['ellipsePix'] = np.empty((self.parameters['number_frames'],5),
                                           dtype = np.float32)
        self.results['pupilPix'] = np.empty((self.parameters['number_frames'],2),
                                            dtype=np.float32)
        self.results['crPix'] = np.empty((self.parameters['number_frames'],2),
                                         dtype = np.float32)
        self.results['ellipsePix'].fill(np.nan)
        self.results['pupilPix'].fill(np.nan)
        self.results['crPix'].fill(np.nan)
        if not len(self.tracker.ROIpoints) >= 4:
            self.cropPoints = []
        else:
            self.updateCropPoints()
        self.startFrame = 0
        #self.endFrame = self.imgstack.nFrames
        self.initUI()
        self.update()

    def initUI(self):

        # Menu
#        bar = self.menuBar()
#        editmenu = bar.addMenu("Experiment")
#        editmenu.addAction("New")
#        editmenu.triggered[QAction].connect(self.experimentMenuTrigger)
 #       self.setWindowTitle("LabCams")
        self.tabs = []
        self.tabs.append(QDockWidget("Parameters",self))
        layout = QVBoxLayout()
        self.paramwidget = MptrackerParameters(self.tracker,self.imgstack.get(0))
        self.tabs[-1].setWidget(self.paramwidget)
        self.tabs[-1].setFloating(False)
        self.addDockWidget(
            Qt.RightDockWidgetArea and Qt.TopDockWidgetArea,
            self.tabs[-1])
        self.tabs.append(QDockWidget("Frame",self))
        self.display = MptrackerDisplay(self.tracker.img)
        self.tabs[-1].setWidget(self.display)
        self.tabs[-1].setFloating(False)
        self.addDockWidget(
            Qt.RightDockWidgetArea and Qt.TopDockWidgetArea,
            self.tabs[-1])

#        grid = QGridLayout()

#        self.wNFrames = QLabel('')
#        self.wNFrames.setMaximumHeight(25)
#        self.wNFrames.setMaximumWidth(200)
#        paramGrid.addRow(QLabel('Number of frames:'),self.wNFrames)

#        self.wFrame = QSlider(Qt.Horizontal)
#        self.wFrame.valueChanged.connect(self.processFrame)
#        self.wFrame.setMaximum(self.imgstack.nFrames-1)
#        self.wFrame.setMinimum(0)
#        self.wFrame.setValue(0)

#        grid.addWidget(self.wFrame,0,2,1,4)
#        # images and plots
#        img = self.imgstack.get(int(self.wFrame.value()))
#        cr_position,pupil_pos,pupil_radius,pupil_ellipse_par = self.tracker.apply(img)
#        self.scene = QGraphicsScene()
#        self.view = QGraphicsView(self.scene)
#        self.setImage(self.tracker.img)
        self.display.view.mouseReleaseEvent = self.selectPoints
#        grid.addWidget(self.view,1,2,6,5)
        ####################
        self.wFrame = self.display.wFrame
        self.wFrame.setMaximum(self.imgstack.nFrames-1)
        self.wFrame.valueChanged.connect(self.processFrame)
        self.wFrame.mouseDoubleClickEvent = self.setStartFrame
        self.paramwidget.update = self.updateParam
        # window geometry
        self.setWindowTitle('mOUSEpUPILtracker')
        self.show()
#        self.updateGUI()
        self.running = False

    def setStartFrame(self,event):
        self.startFrame = int(self.wFrame.value())
        print('StartFrame set [{0}].'.format(self.wFrame.value()))

    def clearPoints(self,event):
        self.tracker.ROIpoints = []
        self.parameters['points'] = []
        self.putPoints()
        self.processFrame(self.wFrame.value())

    def putPoints(self):
        points = self.tracker.ROIpoints
        self.tracker.parameters['pupilApprox'] = None
        self.paramwidget.wPoints.setText(' \n'.join([','.join([str(w) for w in p]) for p in points]))

    def selectPoints(self,event):
        pt = self.display.view.mapToScene(event.pos())
        if event.button() == 1:
            x = pt.x()
            y = pt.y()
            self.parameters['points'].append([int(round(x)),int(round(y))])
            img = self.imgstack.get(int(self.wFrame.value()))
            height,width = img.shape
            self.tracker.setROI(self.parameters['points'])
            self.putPoints()
            self.updateCropPoints()
        elif event.button() == 2:
            x = pt.x()
            y = pt.y()
            img,(x1, y1, w, h) = cropImageWithCoords(self.tracker.ROIpoints,
                                                     self.tracker.img)
            pts = [int(round(x))+x1,int(round(y))+y1]
            self.tracker.parameters['crApprox'] = pts
        self.processFrame(self.wFrame.value())

    def updateCropPoints(self):
        if len(self.tracker.ROIpoints) >= 4:
            img,(x1, y1, w, h) = cropImageWithCoords(self.tracker.ROIpoints,
                                                     self.tracker.img)
            self.cropPoints = [x1,x1+w,y1,y1+h]
        else:
            self.cropPoints = []

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
        self.display.view.fitInView(QRectF(0,0,
                                   frame.shape[1],
                                   frame.shape[0]),
                            Qt.KeepAspectRatio)
        self.scene.update()
    def updateParam(self):
        self.processFrame(int(self.wFrame.value()))
    # Update
    def processFrame(self,val = 0):
        f = int(val)
        try:
            img = self.imgstack.get(f)
        except Exception as err:
            img = None
            print("Failed loading frame {0}".format(f))
            print(err)
        if not img is None:
            cr_pos,pupil_pos,pupil_radius,pupil_ellipse_par = self.tracker.apply(img)
            self.results['ellipsePix'][f,:2] = pupil_radius
            self.results['ellipsePix'][f,2:] = pupil_ellipse_par
            self.results['pupilPix'][f,:] = pupil_pos
            self.results['crPix'][f,:] = cr_pos
            #self.wNFrames.setText(str(f) +
            #                      '//' + str(self.parameters['number_frames']))
            if (len(self.tracker.ROIpoints)>=4  and
                len(self.cropPoints) == 4 and
                not self.tracker.concatenateBinaryImage):
                x1,x2,y1,y2 = self.cropPoints
                image =  cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                image[y1:y2,x1:x2,:] = self.tracker.img
            else:
                image = self.tracker.img
            self.display.setImage(image)
        self.app.processEvents()

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Escape or e.key() == ord('Q'): # ESC
            self.close()
        elif e.key() == 72:
            print('''
+++++++++++++++++++++++++++++++++++++++++
+++++++++++MousePupilTracker+++++++++++++
+++++++++++++++++++++++++++++++++++++++++
+  Esc/Q - Quit                         +
+    R   - run analysis                 +
+    P   - plot                         +
+    F   - run analysis and save output +
+    A   - get augmented output         +
+    M   - run tracker in parallel      +
+    H   - print this message           +
+++++++++++++++++++++++++++++++++++++++++

''')
        elif e.key() == ord("M"):
            self.runParallel()
        elif e.key() == ord("P"):
            results = self.results.copy()
            img = self.imgstack.get(100)
            from .plotutils import plot_results
            plot_results(results,self.parameters,img,100)
        elif e.key() == 82: # R
            if not self.running:
                self.runDetectionVerbose()
            else:
                self.running = False
        elif e.key() == 70: # F
            if not self.running:
                self.runDetectionVerbose(saveOutput = True)
            else:
                self.running = False
        elif e.key() == 65: # A
            # add to keras model
            if self.unet_data is None:
                self.unet_data = []
            img = self.imgstack.get(self.wFrame.value())
            img = self.tracker.getAugmented(img)
            self.display.setImage(img)
        else:
            print(e.key())

    def runParallel(self):
        from .io import TiffFileSequence
        seq = self.imgstack
        if not type(seq) is TiffFileSequence:
            print('This mode only works with TiffFileSequence.')
            return
        if not len(seq.filenames)>1:
            print('There is only one file. Run sequencial [R key] .')
            return
        if not self._initResults():
            return
        from .parutils import par_process_tiff
        print('Starting the parallel run.')
        res = par_process_tiff(seq.filenames,self.tracker.parameters)
        self.results['ellipsePix'][:,:2] = np.array([r[2] for r in res])
        self.results['ellipsePix'][:,2:] = np.array([r[3] for r in res])
        self.results['pupilPix'][:,:] = np.array([r[1] for r in res])
        self.results['crPix'][:,:] = np.array([r[0] for r in res])
        if self.resultfile is None:
            try:
                self.resultfile = str(QFileDialog().getSaveFileName()[0])
            except:
                self.resultfile = None
        self.resultfile = str(self.resultfile)
        res = exportResultsToHDF5(self.resultfile,
                                  self.parameters,
                                  self.results)            
        return 
    def _initResults(self):
        if not len(self.parameters['points']) == 4:
            print('''You did not specify a region... 
Please select 4 points around the eye.
The order matters, the first and third points are the edges of the eye.''')
            return False
        self.results['reference'] = [self.parameters['points'][0],
                                     self.parameters['points'][2]]
        self.results['ellipsePix'].fill(np.nan)
        self.results['pupilPix'].fill(np.nan)
        self.results['crPix'].fill(np.nan)

        return True

    def runDetectionVerbose(self,saveOutput = False):
        self.running = True
        ts = time.time()
        if not self._initResults():
            return 
        if saveOutput:
            # get a filename (tiff to save output)...
            saveOutputFile = QFileDialog().getSaveFileName()
            if type(saveOutputFile) is tuple:
                saveOutputFile = saveOutputFile[0]
            from tifffile import TiffWriter
            with TiffWriter(saveOutputFile) as fd:
                for f in range(self.startFrame,self.parameters['number_frames']):
                    self.wFrame.setValue(f)
                    fd.save(self.tracker.img)
                    if not self.running:
                        break
        else:
            # Just run it.
            for f in range(self.startFrame,self.parameters['number_frames']):
                self.wFrame.setValue(f)
                if not self.running:
                    break
        print('Done {0} frames in {1:3.1f} min'.format(f-self.startFrame,
                                                       (time.time()-ts)/60.))
        if self.resultfile is None:
            try:
                self.resultfile = str(QFileDialog().getSaveFileName()[0])
            except:
                self.resultfile = None
        self.resultfile = str(self.resultfile)
        res = exportResultsToHDF5(self.resultfile,
                                  self.parameters,
                                  self.results)            
        if not res is None:
            print("Saved to " + self.resultfile)

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
    parser.add_argument('-p','--param',
                        type = str,
                        default=None,
                        help = 'Parameter file.')
    parser.add_argument('--usetmp',
                        default = False,
                        action = 'store_true',
                        help = 'Copy data to a temporary folder.')
    args = parser.parse_args()

    app = QApplication(sys.argv)
    target = None
    if os.path.isfile(args.target) or '*' in args.target:
        target = args.target
    params = None

    if not args.param is None:
        import json
        with open(args.param,'r') as fd:
            params = json.load(fd)
            if 'crApprox' in params.keys():
                params['crApprox'] = np.array(params['crApprox'])
    w = MPTrackerWindow(target,app = app,
                        params = params,
                        resfile = args.output,
                        usetmp=args.usetmp)
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
