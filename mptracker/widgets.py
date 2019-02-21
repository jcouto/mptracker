from .tracker import MPTracker
import numpy as np
import cv2
import os
from PyQt5.QtWidgets import (QWidget,
                             QApplication,
                             QVBoxLayout,
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
                             QTabWidget,
                             QFileDialog)
from PyQt5.QtGui import QImage, QPixmap,QBrush,QPen,QColor
from PyQt5.QtCore import Qt,QSize,QRectF,QLineF,QPointF

class MptrackerDisplay(QWidget):
    def __init__(self,img = None):
        super(MptrackerDisplay,self).__init__()

        grid = QFormLayout()
        
        self.wNFrames = QLabel('')
        self.wNFrames.setMaximumHeight(25)
        self.wNFrames.setMaximumWidth(200)
        grid.addRow(QLabel('Number of frames:'),self.wNFrames)
        
        self.wFrame = QSlider(Qt.Horizontal)
        self.wFrame.valueChanged.connect(self.processFrame)
        self.wFrame.setMaximum(1)
        self.wFrame.setMinimum(0)
        self.wFrame.setValue(0)
        grid.addRow(self.wFrame)
        # images and plots
        self.scene = QGraphicsScene()
        self.view = QGraphicsView(self.scene)
        self.setImage(img)
        grid.addRow(self.view)
        self.setLayout(grid)
    def processFrame(self):
        pass
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
                                   frame.shape[0],
                                   frame.shape[1]),
                            Qt.KeepAspectRatio)
        self.scene.update()

class MptrackerParameters(QWidget):
    def __init__(self,tracker,image = np.ones((75,75),dtype=np.uint8)):
        super(MptrackerParameters,self).__init__()
        self.tracker = tracker
        self.parameters = tracker.parameters
        # Set the layout and the tabs widget
        #     - Tab "Tracker parameters"
        #     - Tab "ROI selection"
        #     - Tab "Display and save"
        layout = QVBoxLayout(self)
        self.tabs = QTabWidget()
        self.tabTrackParameters = QWidget()
        self.tabROI = QWidget()
        self.tabDisplay = QWidget()
        layout.addWidget(self.tabs)
        self.tabs.addTab(self.tabTrackParameters,'Tracker parameters')
        self.tabs.addTab(self.tabROI,'Tracker ROI selection')
        self.tabs.addTab(self.tabDisplay,'Display and save')

        # Image parameters
        self.tabTrackParameters.layout = QGridLayout()
        pGroup = QGroupBox(self)
        pGrid = QFormLayout()
        pGroup.setLayout(pGrid)
        pGroup.setTitle("Image enhancement")  
        self.tabTrackParameters.setLayout(self.tabTrackParameters.layout)

        # Gamma
        self.wGamma = QSlider(Qt.Horizontal)
        self.wGamma.setValue(self.parameters['gamma']*10)
        self.wGamma.setMaximum(30)
        self.wGamma.setMinimum(1)
        self.wGamma.setSingleStep(5)
        self.wGammaLabel = QLabel('Gamma [{0}]:'.format(
            self.wGamma.value()/10.))
        self.wGamma.valueChanged.connect(self.setGamma)
        pGrid.addRow(self.wGammaLabel, self.wGamma)
        
        # Gaussian blur
        self.wGaussianFilterSize = QSlider(Qt.Horizontal)
        self.wGaussianFilterSize.setValue(self.parameters['gaussian_filterSize']*10)
        self.wGaussianFilterSize.setMaximum(61)
        self.wGaussianFilterSize.setMinimum(1)
        self.wGaussianFilterSize.setSingleStep(2)
        self.wGaussianFilterSizeLabel = QLabel('Gaussian filter [{0}]:'.format(
            self.wGaussianFilterSize.value()/10.))
        self.wGaussianFilterSize.valueChanged.connect(self.setGaussianFilterSize)
        pGrid.addRow(self.wGaussianFilterSizeLabel, self.wGaussianFilterSize)

        # Contrast equalization (adaptative or full)
        self.wContrastLim = QSlider(Qt.Horizontal)
        self.wContrastLim.setValue(self.parameters['contrast_clipLimit'])
        self.wContrastLim.setMinimum(0)
        self.wContrastLim.setMaximum(200)
        self.wContrastLimLabel = QLabel('Contrast limit [{0}]:'.format(
            self.wContrastLim.value()))
        self.wContrastLim.valueChanged.connect(self.setContrastLim)
        pGrid.addRow(self.wContrastLimLabel,self.wContrastLim)

        self.wContrastGridSize = QSlider(Qt.Horizontal)
        self.wContrastGridSize.setValue(self.parameters['contrast_gridSize'])
        self.wContrastGridSize.setMaximum(200)
        self.wContrastGridSize.setMinimum(1)
        self.wContrastGridSize.setSingleStep(1)
        self.wContrastGridSizeLabel = QLabel('Contrast grid size [{0}]:'.format(
            self.wContrastGridSize.value()))
        self.wContrastGridSize.valueChanged.connect(self.setContrastGridSize)
        pGrid.addRow(self.wContrastGridSizeLabel,self.wContrastGridSize)
        
        # Morphological operations
        pGroup2 = QGroupBox()
        pGrid2 = QFormLayout()
        pGroup2.setLayout(pGrid2)
        pGroup2.setTitle("Morphological operations")  

        self.wOpenKernelSize = QSlider(Qt.Horizontal)
        self.wOpenKernelSize.setValue(self.parameters['open_kernelSize'])
        self.wOpenKernelSize.setMaximum(61)
        self.wOpenKernelSize.setMinimum(0)
        self.wOpenKernelSize.setSingleStep(1)
        self.wOpenKernelSizeLabel = QLabel('Morph open size [{0}]:'.format(
            self.wOpenKernelSize.value()))
        self.wOpenKernelSize.valueChanged.connect(self.setOpenKernelSize)
        pGrid2.addRow(self.wOpenKernelSizeLabel, self.wOpenKernelSize)

        self.wCloseKernelSize = QSlider(Qt.Horizontal)
        self.wCloseKernelSize.setValue(self.parameters['close_kernelSize'])
        self.wCloseKernelSize.setMaximum(61)
        self.wCloseKernelSize.setMinimum(0)
        self.wCloseKernelSize.setSingleStep(1)
        self.wCloseKernelSizeLabel = QLabel('Morph close size [{0}]:'.format(
            self.wCloseKernelSize.value()))
        self.wCloseKernelSize.valueChanged.connect(self.setCloseKernelSize)
        pGrid2.addRow(self.wCloseKernelSizeLabel, self.wCloseKernelSize)

        pGroup3 = QGroupBox()
        pGrid3 = QFormLayout()
        pGroup3.setLayout(pGrid3)
        pGroup3.setTitle("Detection")  
        
        self.wBinThreshold = QSlider(Qt.Horizontal)
        self.wBinThreshold.setValue(self.parameters['threshold'])
        self.wBinThresholdLabel = QLabel('Binary contrast [40]:')
        self.wBinThreshold.setMinimum(0)
        self.wBinThreshold.setMaximum(255)
        self.wBinThreshold.valueChanged.connect(self.setBinThreshold)
        pGrid3.addRow(self.wBinThresholdLabel,self.wBinThreshold)

        self.wRoundIndex = QTextEdit(str(self.parameters['roundIndex']))
        self.wRoundIndex.setMaximumHeight(25)
        self.wRoundIndex.setMaximumWidth(40)
        self.wRoundIndex.textChanged.connect(self.setRoundIndex)
        pGrid3.addRow(QLabel('Circle threshold:'),self.wRoundIndex)

        self.wInvertThreshold = QCheckBox()
        self.wInvertThreshold.setChecked(self.parameters['invertThreshold'])
        self.wInvertThreshold.stateChanged.connect(self.setInvertThreshold)
        pGrid3.addRow(QLabel('White pupil:'),self.wInvertThreshold)

        self.wDisableCRtrack = QCheckBox()
        self.wDisableCRtrack.setChecked(self.parameters['crTrack'])
        self.wDisableCRtrack.stateChanged.connect(self.setCRTrack)
        pGrid3.addRow(QLabel('Track corneal reflection:'),self.wDisableCRtrack)

        self.wSequentialCrMode = QCheckBox()
        self.wSequentialCrMode.setChecked(self.parameters['sequentialCrMode'])
        self.wSequentialCrMode.stateChanged.connect(self.setSequentialCrMode)
        pGrid3.addRow(QLabel('Sequential refraction mode:'),self.wSequentialCrMode)

        self.wSequentialPupilMode = QCheckBox()
        self.wSequentialPupilMode.setChecked(self.parameters['sequentialPupilMode'])
        self.wSequentialPupilMode.stateChanged.connect(self.setSequentialPupilMode)
        pGrid3.addRow(QLabel('Sequential pupil mode:'),self.wSequentialPupilMode)

        self.wResetSequentialPupil = QPushButton('Reset sequential pupil')
        self.wResetSequentialPupil.clicked.connect(self.resetSequentialPupil)
        pGrid3.addRow(self.wResetSequentialPupil)

        self.tabTrackParameters.layout.addWidget(pGroup)
        self.tabTrackParameters.layout.addWidget(pGroup2)
        self.tabTrackParameters.layout.addWidget(pGroup3)

        self.tabDisplay.layout = QGridLayout()
        pGroup = QGroupBox()
        self.pGridSave = QFormLayout()
        pGroup.setLayout(self.pGridSave)
        pGroup.setTitle("Display parameters")
        self.tabDisplay.setLayout(self.tabDisplay.layout)
        
        self.wEyeRadius = QTextEdit(str(self.parameters['eye_radius_mm']))
        self.wEyeRadius.setMaximumHeight(25)
        self.wEyeRadius.setMaximumWidth(40)
        self.wEyeRadius.textChanged.connect(self.setEyeRadius)
        self.pGridSave.addRow(QLabel('Approximate eye radius (mm):'),self.wEyeRadius)


        self.wDisplayBinaryImage = QCheckBox()
        self.wDisplayBinaryImage.setChecked(False)
        self.wDisplayBinaryImage.stateChanged.connect(self.updateTrackerOutputBinaryImage)
        self.pGridSave.addRow(QLabel('Display binary image:'),self.wDisplayBinaryImage)

        self.wDrawProcessed = QCheckBox()
        self.wDrawProcessed.setChecked(self.tracker.drawProcessedFrame)
        self.wDrawProcessed.stateChanged.connect(self.setDrawProcessed)
        self.pGridSave.addRow(QLabel('Draw processed frame:'),self.wDrawProcessed)

        self.wNFrames = QLabel('')
        self.wNFrames.setMaximumHeight(25)
        self.wNFrames.setMaximumWidth(200)
        self.pGridSave.addRow(QLabel('Number of frames:'),self.wNFrames)

        self.saveParameters = QPushButton('Save tracker parameters')
        self.saveParameters.clicked.connect(self.saveTrackerParameters)
        self.pGridSave.addRow(self.saveParameters)

        
        self.tabDisplay.layout.addWidget(pGroup)

        
        self.tabROI.layout = QFormLayout()
        pGrid = self.tabROI.layout
        self.tabROI.setLayout(self.tabROI.layout)
        
        # parameters, buttons and options
        self.wPoints = QLabel('nan,nan \n nan,nan \n nan,nan \n nan,nan\n')
        pGrid.addRow(QLabel('ROI points:'),self.wPoints)
        self.wPoints.mouseDoubleClickEvent = self.clearPoints
        
        self.wROIscene = QGraphicsScene()
        self.wROIview = QGraphicsView(self.wROIscene)
        self.setROIImage(cv2.equalizeHist(image))
        self.wROIview.setGeometry(0,0,75,75)
        self.wROIview.scale(0.3,0.3)
        self.wROIview.mouseReleaseEvent = self.selectPoints
        pGrid.addRow(self.wROIview)
#         self.tabROI.layout.addWidget(pGrid)

        self.setLayout(layout)
        self.show()

    def setROIImage(self,image):
        self.wROIscene.clear()
        if len(image.shape) == 2:
            frame  = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)
        else:
            frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Draw region of interest
        #pts = self.parameters['points']
        #if len(self.parameters['points']) > 2:
        #    pts = np.array(pts).reshape((-1,1,2))
        #cv2.polylines(frame,[pts],True,(0,255,255))
        qimage = QImage(frame, frame.shape[1], frame.shape[0], 
                               frame.strides[0], QImage.Format_RGB888)
        self.wROIscene.addPixmap(QPixmap.fromImage(qimage))
        #self.wROIview.fitInView(QRectF(0,0,
        #                               2000,
        #                               2000),
        #                        Qt.KeepAspectRatio)
        self.wROIscene.update()

        
    def clearPoints(self,event):
        self.tracker.ROIpoints = []
        self.parameters['points'] = []
        self.putPoints()
        self.update()

        
    def putPoints(self):
        points = self.tracker.ROIpoints
        self.tracker.parameters['pupilApprox'] = None
        self.wPoints.setText(' \n'.join([','.join([str(w) for w in p]) for p in points]))
        
    def updateTrackerOutputBinaryImage(self,state):
        self.tracker.concatenateBinaryImage = state
        self.update()

    def setInvertThreshold(self,value):
        self.parameters['invertThreshold'] = value
        self.update()

    def setCRTrack(self,value):
        self.parameters['crTrack'] = value
        self.update()
            
    def setRoundIndex(self):
        value = self.wRoundIndex.toPlainText()
        try:
            self.parameters['roundIndex'] = float(value)
        except:
            print('Need to insert a float in the radius.')
        self.update()

    def setGamma(self,value):
        self.parameters['gamma'] = float(value)/10
        self.wGammaLabel.setText('Gamma [{0}]:'.format(self.parameters['gamma']))
        self.update()

    def setSequentialCrMode(self,value):
        self.parameters['sequentialCrMode'] = value

    def setSequentialPupilMode(self,value):
        self.parameters['sequentialPupilMode'] = value

    def setDrawProcessed(self,value):
        self.tracker.drawProcessedFrame = value
        self.update()

    def setBinThreshold(self,value):
        self.parameters['threshold'] = int(value)
        self.wBinThresholdLabel.setText('Binary contrast [{0}]:'.format(int(value)))
        self.update()

    def setContrastLim(self,value):
        self.parameters['contrast_clipLimit'] = int(value)
        self.wContrastLimLabel.setText('Contrast limit [{0}]:'.format(int(value)))
        self.tracker.set_clhe()
        self.update()

    def setContrastGridSize(self,value):
        self.parameters['contrast_gridSize'] = int(value)
        self.wContrastGridSizeLabel.setText('Contrast grid size [{0}]:'.format(int(value)))
        self.tracker.set_clhe()
        self.update()

    def setGaussianFilterSize(self,value):
        if not np.mod(value,2) == 1:
            value += 1
        
        self.parameters['gaussian_filterSize'] = int(value)
        self.wGaussianFilterSizeLabel.setText('Gaussian filter size [{0}]:'.format(int(value)))
        self.update()

    def setCloseKernelSize(self,value):
        self.parameters['close_kernelSize'] = int(value)
        self.wCloseKernelSizeLabel.setText('Morph close size [{0}]:'.format(int(value)))
        self.update()

    def setOpenKernelSize(self,value):
        self.parameters['open_kernelSize'] = int(value)
        self.wOpenKernelSizeLabel.setText('Morph open size [{0}]:'.format(int(value)))
        self.update()

    def resetSequentialPupil(self):
        self.tracker.parameters['pupilApprox'] = None

    def setEyeRadius(self):
        value = self.wEyeRadius.toPlainText()
        try:
            self.parameters['eye_radius_mm'] = float(value)
            print(self.tracker.parameters['eye_radius_mm'])
        except:
            print('Need to insert a float in the radius.')
    def selectPoints(self,event):
        pt = self.wROIview.mapToScene(event.pos())
        if event.button() == 1:
            x = pt.x()
            y = pt.y()
            self.parameters['points'].append([int(round(x)),int(round(y))])
            #img = self.imgstack.get(int(self.wFrame.value()))
            #height,width = img.shape
            self.tracker.setROI(self.parameters['points'])
            self.putPoints()
        elif event.button() == 2:
            x = pt.x()
            y = pt.y()
            from .tracker import cropImageWithCoords
            img,(x1, y1, w, h) = cropImageWithCoords(self.tracker.ROIpoints, self.tracker.img)
            pts = [int(round(x))+x1,int(round(y))+y1]
            self.tracker.parameters['crApprox'] = pts
    def update(self):
        print('Pass...')

    def saveTrackerParameters(self,resultfile = None):
        if resultfile is None or resultfile == False:
            try:
                paramfile = QFileDialog().getSaveFileName()
            except:
                paramfile = ''
        else:
            paramfile = str(resultfile)
        if type(paramfile) is tuple:
            paramfile = paramfile[0]
        from .io import saveTrackerParameters
        res = saveTrackerParameters(paramfile,parameters)
        print('Saved parameters [{0}].'.format(paramfile))        

        
