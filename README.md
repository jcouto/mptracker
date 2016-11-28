mOUSEpUPILtracker
================

A mouse pupil tracker using Python and OpenCV.

On a common laptop can process 13 min in around 4 minutes, however there is a batch mode that uses MPI to launch frames on separate nodes/computers. OpenCV can be compiled to use the graphics card to speed up computations.

![picture](images/mptrackerExample.png)

Supported file formats:
-----------------------
   - Multipage TIFF sequence
   - Norpix seq files [not tested  yet...]

Output file format:
-------------------
*Results are saved as a python pickle for now... load using the unpickle method. The roadmap is bellow.*

The output is an HDF5 file that can be read pretty much anywhere.

The file is organized as follows:

- **/diameter** - diameter of the pupil [in mm]
- **/azimuth** - azimuth of the pupil [in degrees]
- **/elevation** - elevation of the pupil [in degrees]
- */theta* - angle [in degrees]
- */ellipsePix* - ellipse parameters for each frame [in pixels] [short_axis,long_axis,a,b,phi]
- */positionPix* - position of the eye in pixels
- */crPix* - position of the corneal reflexion in pixels
- *points* - points marked by the user (left eye corner, top of the eye, right eye corner, bottom of the eye). These points mark the area to be analysed and define the scale.
   
Installation:
-------------
Install dependencies:

- PyQt5
- opencv 3.0 (cv2)
- h5py
- mpi4py (Not needed if ran on a single computer)

Usage:
------
Launch the GUI from the command line: ``mptracker-gui <filename>``

**NOTE**: Do ``mptracker-gui --help`` for options.


Please let me know whether this works for you and acknowledge if you use it in a publication.

**Joao Couto** - *jpcouto@gmail.com*

November 2016