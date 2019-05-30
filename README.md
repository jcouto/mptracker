mOUSEpUPILtracker
================

A mouse pupil tracker using Python and OpenCV.
This can process 5 min of recording in around 10 seconds (on a laptop).

Help making it work for your dataset by letting me know cases where it fails/works.

![picture](images/mptrackerExample.png)

Supported file formats:
-----------------------
   - Multipage TIFF sequence
   - Norpix seq files
   - AVI

Output file format:
-------------------

The output is an HDF5 file that can be read pretty much anywhere.

The file is organized as follows:

- **/diameter** - diameter of the pupil [estimated mm]
- **/azimuth** - azimuth of the pupil [estimated degrees]
- **/elevation** - elevation of the pupil [estimated degrees]
- */theta* - angle 
- */ellipsePix* - ellipse parameters for each frame [in pixels] [short_axis,long_axis,a,b,phi]
- */positionPix* - position of the eye in pixels
- */crPix* - position of the corneal reflexion in pixels
- *points* - points marked by the user (left eye corner, top of the eye, right eye corner, bottom of the eye). These points mark the area to be analysed and define the scale.

In MATLAB do for example: `diam = h5read('filename.something','/diameter')`

Installation:
-------------
Dependencies:

- PyQt5
- pyqtgraph
- opencv 3.0 (cv2)
- h5py
- PIMS (for reading norpix seq files)
- libtiff
- natsort
- numpy, scipy, matplotlib, shutil
### Install instructions:

1. Get [ miniconda ](https://conda.io/miniconda.html) (e.g. Python 3.7 x64) 
2. ``conda install pyqt h5py scipy numpy matplotlib``
3. ``conda install -c conda-forge tifffile``
4. ``conda install -c menpo opencv3``
5. ``pip install pims pyqtgraph tqdm natsort``
6. Clone the repositoty: ``git clone https://bitbucket.org/jpcouto/mptracker.git``
7. Go into that folder``cd mptracker`` and finally ``python setup.py develop``

**Note:** On windows I suggest getting the [ git bash terminal ](https://git-scm.com/downloads); Installing Anaconda as system python will make your life easier...

Usage (GUI):
------------

**NOTE**: ``mptracker-gui --help`` for options.


Launch the GUI from the command line: ``mptracker-gui <filename>``. The filename is that of a seq file or one of the TIFF files in a TIFF sequence.  


### Command line options:

- *-o* <output file path> File where to save the results to (will ask if not specified).
- *-p* <parameter file> Parameter file to load.

### Instructions:

1.   Select left corner; top; right corner and bottom of the eye by dragging the points (keep the arangement between points the same as default).
2.   Adjust the parameters for best pupil contrast.
3.   Press the key *r* to launch the analysis. You will be prompted for a filename where to save  when it finishes. ( *r* stops the analysis also).
4.   Press the key *p* to plot results.


**Please let me know whether this works for you and acknowledge if you use it in a publication.**

**Joao Couto** - *jpcouto@gmail.com*

November 2016
