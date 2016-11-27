mOUSEpUPILtracker
=========

A mouse pupil tracker using python and opencv.

On a common laptop can process 13 min in around 4 minutes, however there is a bach mode for using MPI that can be ran on multiple computers. 

![picture](images/mptrackerExample.png)

Supported file formats:
-----------------------
   - Multipage TIFF sequence
   - Norpix seq files

Output file format:
-------------------
The output is an HDF5 file that can be read pretty much anywhere.

The file is organized as follows:

- */ellipsePix* - ellipse parameters for each frame [in pixels] [short_axis,long_axis,a,b,phi]
- */positionPix* - position of the eye in pixels
- */crPix* - position of the corneal reflexion in pixels
- *points* - points marked by the user (left eye corner, top of the eye, right eye corner, bottom of the eye). These points mark the area to be analysed and define the scale.
- **/diameter** - diameter of the pupil [in mm]
- **/azimuth** - azimuth of the pupil [in degrees]
- **/elevation** - elevation of the pupil [in degrees]
- **/theta** - angle of the pupil [in degrees]
   
Installation:
-------------
Install dependencies:

- PyQt5
- opencv 3.0 (cv2)
- h5py
- mpi4py (Not needed if ran on a single computer)

Usage:
------
Launch the GUI from the command line: ``mptracker-gui``

**NOTE**: Do ``mptracker-gui --help`` for options.


Please let me know whether this works for you and acknowledge if you use it in a publication.

**Joao Couto** - *jpcouto@gmail.com*

November 2016