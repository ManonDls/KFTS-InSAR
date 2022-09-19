#############################################################################
# GIAnT Utilities for reading input data for time-series InSAR analysis.
#
# author:
#     Piyush Agram <piyush@gps.caltech.edu>
#     Modified for KFTS : Manon Dalaison
#############################################################################
    
import numpy as np
import h5py
import time
import sys
import logging
import re
import os
import collections
import lxml.objectify as ob
import scipy.io.netcdf as netcdf

###############################File I/O Utils ############################

###Create a memory map using numpy's memmap module.
def load_mmap(fname, nxx, nyy, map='BSQ', nchannels=1, channel=1, datatype=np.float32, quiet=False, conv=False):
    '''Create a memory map to data on file.

    Args:

        * fname   -> File name
        * nxx     -> Width
        * nyy     -> length

    KwArgs:
        * map       -> Can be 'BIL', 'BIP', 'BSQ' 
        * nchannels -> Number of channels in the BIL file. 
        * channel   -> The channel needed in the multi-channel file
        * datatype  -> Datatype of data in file (float 32 default)
        * quiet     -> Suppress logging outputs
        * conv      -> Switch from low-endian to big-endian

    Returns:

        * fmap     -> The memory map.'''

    if quiet==False:
        logging.info('Reading input file: %s'%(fname))

    ####Get size in bytes of datatype
    ftemp = np.zeros(1, dtype=datatype)
    fsize = ftemp.itemsize

    if map.upper() == 'BIL':  #Band Interleaved by Line 
        nshape = (nchannels*nyy-channel+1, nxx) 
        noffset = (channel-1)*nxx*fsize

        try:
            omap = np.memmap(fname, dtype=datatype, mode='r', shape=nshape, offset = noffset)
        except:
            raise Exception('Could not open BIL style file or file of wrong size: ' + fname)

        if conv:
            gmap = omap.byteswap(False)
        else:
            gmap = omap

        nstrides = (nchannels*nxx*fsize, fsize)
        fmap = np.lib.stride_tricks.as_strided(gmap, shape=(nyy,nxx), strides=nstrides)

    elif map.upper() == 'BSQ': #Band Sequential
        noffset = (channel-1)*nxx*fsize*nyy
        try:
            gmap = np.memmap(fname, dtype=datatype, mode='r', shape=(nyy,nxx), offset=noffset)
        except:
            raise Exception('Could not open BSQ style file or file of wrong size: ' + fname)

        if conv:
            fmap = gmap.byteswap(False)
        else:
            fmap = gmap

    elif map.upper() == 'BIP': #Band interleaved by pixel
        nsamps = nchannels * nyy * nxx  - (channel-1)
        noffset = (channel-1)*fsize

        try:
            gmap = np.memmap(fname, dtype=datatype,mode='r', shape = (nsamps), offset = noffset)
        except:
            raise Exception('Could not open BIP style file or file of wrong size: ' + fname)

        if conv:
            omap = gmap.byteswap(False)
        else:
            omap = gmap

        nstrides = (nchannels*nxx*fsize, nchannels*fsize)
        fmap = np.lib.stride_tricks.as_strided(omap, shape=(nyy,nxx), strides=nstrides)
    
    else:
        assert False, "do not recognize format of map {}, should be: 'BIL', 'BIP' or 'BSQ'".format(map.upper())

    return fmap



#####Load a simple float file
def load_flt(fname, nxx, nyy, scale=1.0, datatype=np.float32, quiet=False, conv=False):
    '''Load a FLAT BINARY file.
    
    Args:
    
        * fname         Name of the file
        * nxx           Width of the image
        * nyy           Length of the image
        
    Kwargs:
    
        * scale        Scales by the this factor.
        * datatype     Numpy datatype, FLOAT32 by default.
    
    Returns:
    
        * phs          Scaled array (single channel)'''

    if quiet == False:
        logging.info('READING FLOAT FILE: %s'%(fname))

    phs = np.zeros(nxx * nyy, dtype=datatype)
    try:
        fin = open(fname, 'rb') #Open in Binary format
        phs = np.fromfile(file=fin, dtype=datatype, count=nxx * nyy)
    except:
        raise Exception('Could not open RMG file or file of wrong size: ' + fname)

    fin.close()

    if conv:
        phs.byteswap(True)

    phs = np.reshape(phs, (nyy, nxx))
    phs = phs * scale
    return phs


#######Load a GMT grd file
def load_grd(fname, var='z', shape=None):
    '''Load a GMT grd file.

    Args:
        * fname         Name of the file

    Kwargs:
        * var           Variable name to be loaded


    Returns:
        * data          2D array of the data'''

    try:
        fin = netcdf.netcdf_file(fname)
    except:
        raise Exception('Could not open GMT file: ' + fname)

    z = fin.variables[var]
    if shape is not None:
        if z.shape != shape:
            raise ValueError('Shape mismatch in GRD file: '%fname)

    return z.data




