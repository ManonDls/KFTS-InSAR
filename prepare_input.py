# -*- coding: utf-8 -*-
##############################################################################
# Read and Prepare data for KFTS
# Heavily inspired by GIAnT v1.0 (P. Agram & R. Jolivet)
# 
# Construct H5 files containing interfero 
# and auxiliary files :
# lon.flt, lat.flt, inc.flt, head.flt, hgt.flt 
#
# Manon Dalaison
# Sept 2022
#############################################################################

import numpy as np
import os
import h5py
import datetime as dt
import time as TIME 
import kf.utils.tsio as ts
#import kfts

# To Parametrize
import configparser
import argparse
from ast import literal_eval

# Only for deramping 
from matplotlib import path
import scipy.linalg 


class GetConfig(object):
    '''
    Class to run read config file and keep everything in object
    '''
    
    def __init__(self, configfile):
        
        config = configparser.ConfigParser(interpolation = configparser.ExtendedInterpolation())
        config.read(configfile)
        
        #----------------------------------------------
        secIN = config['INPUT']
        self.loc = secIN.get('workdir', fallback='./')
        self.baselinedir = self.loc + secIN.get('baselinedir', fallback='')
        self.igramsdir = self.loc + secIN.get('igramsdir', fallback='')
        
        self.unwfmt = secIN.get('unwfmt', fallback='ISCE')

        #optional arguments, None otherwise
        self.lonfile = secIN.get('lonfile', fallback=None)
        self.latfile = secIN.get('latfile', fallback=None)
        self.losfile = secIN.get('losfile', fallback=None)
        self.demfile = secIN.get('demfile', fallback=None)
        
        if self.lonfile is not None:
            self.lonfile = os.path.join(self.loc,self.lonfile)
        if self.latfile is not None:
            self.latfile = os.path.join(self.loc,self.latfile)
        if self.losfile is not None:
            self.losfile = os.path.join(self.loc,self.losfile)
        if self.demfile is not None:
            self.demfile = os.path.join(self.loc,self.demfile)
                 
        ilistfile = secIN.get('ilistfile',fallback='')
        clistfile = secIN.get('clistfile',fallback='')
        
        listfile = os.path.join(self.loc, ilistfile)
        if os.path.isfile(listfile):
            print("Will load interferograms listed in",listfile )
            self.ilist = np.genfromtxt(listfile, dtype=str)
        else : 
            print("WARNING : Could not find predefined list of interferograms",listfile)
            self.ilist = None 
        
            
        #----------------------------------------------
        secOU = config['OUTPUT']
        self.verbose = secOU.getboolean('verbose', fallback = False)
        self.outdir = secOU.get('outdir', fallback='./')
        self.outfile = self.outdir + secOU.get('outfile', fallback ='PROC-STACK.h5')
        
        if self.verbose:
            print("Input directory is {}, output directory is {}".format(self.loc,self.outdir))
            
        #----------------------------------------------
        secPA  = config['PARAMETERS']
        self.ylen = secPA.getint('ylen',fallback=0)
        self.xlen = secPA.getint('xlen',fallback=0)
        
        self.yreflim = literal_eval(secPA.get('yreflim',fallback = (0,0)))
        self.xreflim = literal_eval(secPA.get('xreflim',fallback = (0,0)))
        
        self.ylim = literal_eval(secPA.get('ylim',fallback = (0,-1)))
        self.xlim = literal_eval(secPA.get('xlim',fallback = (0,-1)))

        self.cthresh = secPA.getfloat('cthresh',fallback=0.0)
        assert (self.cthresh >=0.0) & (self.cthresh <1.0), "Coherence threshold must be between 0 and 1"
        if self.cthresh > 0.0 :
            listfile = os.path.join(self.loc, clistfile)
            if os.path.isfile(listfile):
                print("Will load coherence maps listed in",listfile )
                self.clist = np.genfromtxt(listfile, dtype=str)
            else : 
                print("WARNING : Could not find predefined list of coherence maps",listfile)
                self.clist = None 
            
        
        self.deramp = secPA.getboolean('deramp', fallback=False)
        if self.deramp:
            #Apex of polygon defining the reference region for deramping
            self.dref = literal_eval(secPA.get('rampzone',fallback=[(0,0)]))
         
        # wavelength (in meters) or sensor    
        self.wvl = secPA.get('wavelength',fallback='S1')
        if self.wvl == 'S1':
            self.wvl = 0.05546576 #number from master/IW1.xml in ISCE as radarwavelength
            print("Assume Sentinel 1 sensor")
        elif isinstance(literal_eval(self.wvl),float) :
            print("Sensor wavelength defined to {} METERS".format(self.wvl))
        else : 
            assert False, "Format of wavelength parameter not understood"
        
        # List for changing endianness. Possible Entries:UNW,COR,DEM,LAT,LON,INC,MASK
        self.endianlist = literal_eval(secPA.get('endianlist',fallback=[]))
        
        #End of Class 
    

#-------------------------------------------------------------------------- 
#Create IFG list

def getPairs(igramsDir):
    '''
    Get the two dates of each interferogram
    produced by ISCE, in Igram directory.'''
    
    # Get pairs
    dates1,dates2 = [], []
    print("Get Dates from directory names in {}".format(igramsDir))
    for dirs in os.listdir(igramsDir):
        if os.path.isdir(igramsDir+'/'+dirs) and (len(dirs)==17) and (dirs[8]=='_'):
            di1 = dirs[0:8]
            di2 = dirs[9:17]
            dates1.append(di1)
            dates2.append(di2)
        
    # All done
    return np.array(dates1), np.array(dates2)
    
def getBaselines(baselineDir, masterDate):
    '''Get baseline values from baseline files.'''
    
    # Initialize lists
    baselineFiles = []
    datestrings = []
    dateList = []
    baselineList = []

    # List all baseline files
    print("Looking for baselines in",baselineDir)
    for dates in os.listdir(baselineDir):
        dirs = os.path.join(baselineDir,dates)
        if os.path.isdir(dirs):
            for fil in os.listdir(dirs):
                if fil.endswith('.txt'):
                    if os.stat(os.path.join(dirs,fil)).st_size != 0: # file not empty
                        baselineFiles.append(os.path.join(dirs,fil))
                        datestrings.append(dates)
    
    print("found {} baseline files".format(len(datestrings)))

    # Calculate baseline
    for i in range(len(baselineFiles)):
            
        # Get date 1 and 2 of the pair
        baselineFile = baselineFiles[i]
        datestring = datestrings[i]
      
        dp2 = datestring[9:17]
        dateList.append(dp2)

        # Read by line
        linestring = open(baselineFile, 'r').read()
        
        # Initialize swath baselines list
        Bp = []

        # Get baseline of the first swath
        Bpiw1 = linestring.split('\n')[1]
        Bpiw1 = float(Bpiw1[17:31])
        Bp.append(Bpiw1)

        # Get the second swath baseline if exists
        if linestring.split('\n')[3]:
            Bpiw2 = linestring.split('\n')[4]
            Bpiw2 = float(Bpiw2[17:31])
            Bp.append(Bpiw2)

            # Get the third baseline swath
            if linestring.split('\n')[6]:
                Bpiw3 = linestring.split('\n')[7]
                Bpiw3 = float(Bpiw3[17:31])
                Bp.append(Bpiw3)

        # Calculate mean
        Bperp = np.mean(Bp)
        baselineList.append(Bperp)

    # Add master to lists
    dateList = ["{}".format(masterDate)] + dateList
    baselineList = np.concatenate((np.array([0.00]), baselineList))

    # Transform lists into arrays
    return np.array(dateList), np.array(baselineList)


def save2file(dates1, dates2, outfile, dateList = None, baselineList = None, satnameList = None):
    '''
    Save baselines to a file.
    It can be used for GIAnT time series analysis.
    '''
    if baselineList is not None:
        # Initialize list
        bp_couple = []
    
        # Get baseline of the pair
        for i in range(0,len(dates1),1):
            dt1_ind=np.where(dateList ==  dates1[i])
            dt2_ind=np.where(dateList ==  dates2[i])
            bp = str(baselineList[dt1_ind] - baselineList[dt2_ind])
            if len(bp.strip("[]"))==0 :
                bp_couple.append(np.NaN)
            else :
                bp_couple.append(bp.strip("[]"))
    
        # Prepare data for saving
        dates1List = np.array(dates1)
        dates2List = np.array(dates2)
        bp_coupleList = np.array(bp_couple)
        #satnameList = ['S1']*(len(dates1))
        alldata = (np.array([dates1List, dates2List, bp_coupleList])).T
        np.savetxt(outfile, alldata, fmt=['%s','%s','%s'], delimiter=' ')
   
    else : 
        alldata = np.column_stack((np.array(dates1), np.array(dates2)))
        np.savetxt(outfile, alldata, fmt=['%s','%s'], delimiter=' ')

#--------------------------------------------------------------------------     
#Find path of interferograms 

def makefnames(dirname, dates1,dates2):
    '''
    Generates paths to the files needed for creating the stack.
    .. Args:
        * dates1     - Date of master 
        * dates2     - Date of slave
    .. Returns:
        * iname      - Path to the unwrapped interferogram
        * cname      - Path to the coherence file'''

    if os.path.exists('%s/%s_%s/filt_ECMWF_ERA-5_fine_corrected.unw'%(dirname,dates1,dates2)):
        iname = '%s/%s_%s/filt_ECMWF_ERA-5_fine_corrected.unw'%(dirname,dates1,dates2)
        cname = '%s/%s_%s/filt_ECMWF_ERA-5_fine.cor'%(dirname,dates1,dates2)

    elif os.path.exists('%s/%s_%s/filt_ECMWF_ERA-5_fine.unw'%(dirname,dates1,dates2)):
        print('No unwrapping error correction for dates: {}-{}'.format(dates1,dates2))
        iname = '%s/%s_%s/filt_ECMWF_ERA-5_fine.unw'%(dirname,dates1,dates2)
        cname = '%s/%s_%s/filt_ECMWF_ERA-5_fine.cor'%(dirname,dates1,dates2)

    elif os.path.exists('%s/%s_%s/filt_fine.unw'%(dirname,dates1,dates2)):
        print('No atmospheric corection for dates: {}-{}'.format(dates1,dates2))
        iname = '%s/%s_%s/filt_fine.unw'%(dirname,dates1,dates2)
        cname = '%s/%s_%s/filt_fine.cor'%(dirname,dates1,dates2)

    else :
        print('%s/%s_%s/filt_fine.unw'%(dirname,dates1,dates2))
        print('No igram detected for dates: {}-{}'.format(dates1,dates2))
        
    return iname,cname

#--------------------------------------------------------------------------   
#--------------------------------------------------------------------------     

class BuildStack(object):
    ''' Quick class to deal with stack of interferogram'''
    def __init__(self,verbose,ylims,xlims):
        
        self.verbose = verbose
        self.ylims = ylims
        self.xlims = xlims
        if verbose: 
       
            print("Initialize stack")

    #--------------------------------------
    #Create RAW-STACK
    def datenum(self, datelist):
        '''Converts list of dates in yymmdd/yyyymmdd format to ordinal array.'''
    
        Uts = np.zeros(len(datelist))
        for k in range(len(datelist)):
            dint = int(datelist[k])
            if dint < 1e7:
                dint = dint+2e7
            yy = int(dint/10000)
            mm = int((dint - yy*10000)/100)
            dd = int(dint-(yy*10000+mm*100))
            dstr = dt.date(yy,mm,dd)
            Uts[k] = dstr.toordinal()      
        return Uts
        
    def ConnectMatrix(self, dates):
        '''
        Gets the connectivity matrix for given set of IFGs.
        Args:
            * dates        ->  List of pairs of strings 
        Returns:
            * Uts          -> Unique dates of acquisitions
            * connMat      -> Connectivity matrix (1,-1,0)'''
        
        Nifg = dates.shape[0]
        scenes = np.zeros((Nifg*2),dtype=('str',16))
        scenes[:] = np.hstack((dates[:,0],dates[:,1]))
        
        #####Get unique date 
        uscenes =  np.unique(scenes)
        Nsar = uscenes.shape[0]
        Uts = np.zeros(Nsar)
        
        ######Convert to ordinal
        Uts = self.datenum(uscenes) #ordinal dates
        ind = np.argsort(Uts) #Increasing order of time
        uscenes = uscenes[ind]
        
        if dates.dtype != uscenes.dtype: #check if bytes 
            dates.astype(str)
            uscenes.astype(str)

        ConnMat = np.zeros((Nifg,Nsar))
        for k in range(Nifg):
            mind = np.where((uscenes == dates[k,0]))
            sind = np.where((uscenes == dates[k,1]))
            ConnMat[k,mind[0]] = 1.0
            ConnMat[k,sind[0]] = -1.0
        Uts = Uts[ind]
        
        self.Jmat = ConnMat
        self.days = Uts
    
    #-------------------------------------
    #Create PROC-STACK
    def referenceIgram(self,phs):
        ''' remove for each single 2D interfero phs'''
        Ry0,Ry1 = self.ylims
        Rx0,Rx1 = self.xlims
        
        #Compute reference phase 
        sub = phs[Ry0:Ry1,Rx0:Rx1]
        refph = np.nanmean(sub)

        if np.isnan(refph):
            if not np.isnan(np.nanmean(phs[Ry0-10:Ry1+10,Rx0-10:Rx1+10])):
                print("Reference region is empty, expand by 10x10")
                refph = np.nanmean(phs[Ry0-10:Ry1+10,Rx0-10:Rx1+10])
            elif np.isnan(np.nanmean(phs)):
                raise ValueError('Interferogram is empty (all zeros).')
            else : 
                raise ValueError('Reference region has no pixels.')
            
        phs = phs - refph
        phs = phs.astype(np.float32)
        return phs
        
    def buildAmat(self,Nxy,x,y):
        if self.poly==1:
            A = np.ones(Nxy)
        elif self.poly==3:
            A = np.column_stack((np.ones(Nxy),x,y))
        elif self.poly==4:
            A = np.column_stack((np.ones(Nxy),x,y,x*y))
        return A 
        
    def findramp(self, phs,mask):
        '''
        Estimate the best-fitting ramp parameters for a matrix.
        Args:
            * phs     -  Input unwrapped phase matrix.
            * mask    -  Mask of reliable values.
            * poly    -  Integer flag for type of correction.
                            1 - Constant
                            3 - Plane
                            4 - Plane + cross term
        Returns:
            * ramppoly - Polynomial corresponding to the rank'''
    
        #On masked grid
        [ii,jj] = np.where((mask != 0) & np.isfinite(phs))
        y = ii/(1.0 *self.ny)
        x = jj/(1.0 *self.nx)
        gval = phs[ii,jj] #Good values
        
        A = self.buildAmat(len(ii),x,y)
    
        ramp = scipy.linalg.lstsq(A,gval, cond=1.0e-8)
        ramppoly = ramp[0]
    
        return ramppoly
    
    def removeramp(self, phs,ramppoly):
        '''
        Deramp a matrix with the given ramp polynomial.
        .. Args:
            * phs             Input matrix
            * ramppoly        Ramp polynomial
        .. Returns:
            * dphs            Deramped phase matrix.'''
    
        #On full grid 
        X,Y = np.meshgrid(np.arange(self.nx)*1.0,np.arange(self.ny)*1.0)
        X = X.flatten()/(self.nx*1.0)
        Y = Y.flatten()/(self.ny*1.0)
    
        assert self.poly == len(ramppoly), ""
        
        A = self.buildAmat(self.ny*self.nx,X,Y)
        dphs = np.dot(-A,ramppoly)
        del A
       
        mask = (phs == 0.0) #save where zeros as NaN
        dphs = np.reshape(dphs,(self.ny,self.nx))
        dphs = phs + dphs
        dphs[mask] = 0.0 

        return dphs
    
    def deramp(self, data, out, network=True, poly=3, dref=[(0,0)]):
        '''
        Network deramping of the stack of interferograms. Used when no GPS 
        is available. 
        
        Args:
            * data      Input stack object (interferograms)
            * out       Output stack object (dataset will be overwritten)
        
        Kwargs:
            * network   Network deramping or individual deramping
            * poly      Polynomial code for deramping
            * dref      Coordinates of points delimiting a polygon taken 
                        as the reference region for estimating a ramp
                        (list of 2-tuple) '''
        
        self.nslice,self.ny,self.nx = data.shape
        self.poly = poly
        
        # Define reference region for deramping
        if len(dref) < 3 :
            # not enough points
            # means all interfero as reference for deramping
            maskref = np.ones((self.ny,self.nx))
    
        elif len(dref) >= 3 and type(dref[0])==tuple :
            # polygon of reference 
            polygon = path.Path(dref)
    
            #geometry of data
            xv,yv = np.meshgrid(np.array(list(range(self.nx))),np.array(list(range(self.ny))))
    
            #create mask for ref
            maskref = polygon.contains_points(np.hstack((xv.flatten()[:,np.newaxis],yv.flatten()[:,np.newaxis])))
            maskref = np.reshape(maskref,(self.ny,self.nx))
            
        else :
            assert False, "Points defining reference zone for deramping should be a list of 2-tuples"
    
        # Get Ramp parameter for each interferogram
        ramparr = np.zeros((self.nslice, poly))
        
        if self.verbose:
            print("Get ramp for each interferogram")
        for kkk in range(self.nslice):
            dat = data[kkk, :, :]
            dat[dat==0.0] = np.nan

            mask = np.isfinite(dat)
            masktmp = mask *maskref
    
            ramp = self.findramp(dat, masktmp)
            ramparr[kkk, :] = ramp
        
        if network:
            if self.verbose:
                print("Check network consistency")
            jmat = self.Jmat
            umat, svec, vmat = np.linalg.svd(jmat, compute_uv=1,
                    full_matrices=False)
            rnk = np.linalg.matrix_rank(jmat)
            svec[rnk:] = 0.
            jtilde = np.dot(umat, np.dot(np.diag(svec), vmat))
        else:
            jmat = np.diag(np.ones(self.nslice))
            jtilde = jmat.copy()
    
        jinv = scipy.linalg.pinv(jtilde)
        sarramp = np.dot(jinv, ramparr)
        rampest = np.dot(jmat, sarramp)
        
        #clean
        del ramparr
        del sarramp
        del jtilde

        if self.verbose:
            print('Apply network deramp for each interferogram')
        for kkk in range(self.nslice):
            ramp = self.removeramp(data[kkk, :, :], rampest[kkk, :])
            out[kkk, :, :] = self.referenceIgram(ramp)

        return out 
        
#-------------------------------------------------------------------------- 

if __name__ == '__main__':
    
    # Get inline config file name
    parser = argparse.ArgumentParser( description = 'Collect and prepare data for InSAR TSA')
    parser.add_argument('-c', type=str, dest = 'config', default = None,
        help = 'Specify INI config file containing at least sections INPUT,OUTPUT,PARAMETERS')
    argparser = parser.parse_args()
    while (not argparser.config):
        print('Please choose a config file')
        exit()
    
    # extract content of config and store in class
    args = GetConfig(argparser.config)
    
    # Get master date
    baselineName = os.listdir(args.baselinedir)[1]
    masterDate = baselineName.split("_",2)[0]

    # Get date pairs and baselines
    dates1, dates2 = getPairs(args.igramsdir)

    # Get baselines and save to txt 
    if args.baselinedir is not None:
        dateList, baselineList = getBaselines(args.baselinedir, masterDate)
        save2file(dates1, dates2, "Ifg_dates_baselines.txt", dateList, baselineList)
    else :
        print('Baseline directory not defined, will not keep track of baselines') 
        save2file(dates1, dates2, "Ifg_dates.txt")
    
    # Load function to find interferogram file properly
    dates = np.hstack((dates1[:,None],dates2[:,None]))
    
    # Spatial definitions
    flen,fwid =  args.ylen,args.xlen  #former shape
    x0,x1    = args.xlim
    y0,y1    = args.ylim
    nlen,nwid = y1-y0, x1-x0 #new shape 
    if args.verbose:
        print('------------------------------------------------')
        print("Spatially Resize all interferograms from {} to {}".format((flen,fwid),(nlen,nwid)))
        print("Reference zone along Y {} and X {}".format(args.yreflim,args.xreflim))

    stack = BuildStack(args.verbose,args.yreflim,args.xreflim)
    stack.ConnectMatrix(dates)
    Nifg, Nsar = stack.Jmat.shape
    nIslands = np.min(stack.Jmat.shape) - np.linalg.matrix_rank(stack.Jmat)
    
    if args.verbose : 
        print('------------------------------------------------')
        print('Number of interferograms = ', Nifg)
        print('Number of unique SAR scenes = ', Nsar)
        print('Number of connected components in network: ', nIslands)
    if nIslands > 1:
        print('WARNING: The network appears to contain disconnected components')
        
    #######Setting first scene to reference.
    tims = (stack.days-stack.days[0])/365.25    #conversion to years
    
   
    #####Preparing for reading in IFGs.
    outname = args.outfile
    print('Output h5file:',outname)
    if os.path.exists(outname):
        os.remove(outname)
        print('Deleting previous',outname)
    
    if args.deramp:
        outtmp = os.path.join(args.outdir,"NODERAMP-STACK.h5")
        if os.path.exists(outtmp):
            os.remove(outtmp)
        if args.verbose:
            print("open temporary H5file for storage of raw interferograms before deramping {}".format(outtmp))
        ftmp = h5py.File(outtmp,"w")
        ifgsraw = ftmp.create_dataset('igram',(Nifg,nlen,nwid),'f')
    
    else:
        f = h5py.File(outname,"w")
        f.attrs['help'] = 'All the raw data read from individual interferograms into a single location for fast access.'
        ifgs = f.create_dataset('igram',(Nifg,nlen,nwid),'f')
        ifgs.attrs['help'] = 'Unwrapped IFGs read straight from files.'
    
    print('Reading in IFGs')
    unwfmt = args.unwfmt
    endianlist = args.endianlist
    
    #Single sensor being used.
    scl = 1000 *args.wvl/(4*np.pi) ####Converting to mm.
                
    for k in range(10):
        if args.ilist is None:
            #####File names not provided as input
            iname,cname = makefnames(args.igramsdir,dates[k,0],dates[k,1]) #cname may not exist at this stage
        else:
            #####File names provided in input file
            iname = args.ilist[k]
            if args.cthresh > 0.0:
                cname = args.clist[k]
        if args.verbose: 
            print("Interferogram {}/{}: {} {}".format(k, Nifg, dates1[k], dates2[k]))

        #----------------------------------------------------------------
        # READ IGRAM
        if unwfmt in  ['GMT','GRD']:
            phs = ts.load_grd(iname, shape=(flen,fwid))
        
        elif unwfmt in ['ISCE','RMG', 'FLT']: 
            if unwfmt in ['ISCE','RMG']:
                mapname = 'BIL' #Band Interleaved by Line adapted for ISCEstackS1
            elif unwfmt=='FLT' : 
                mapname = 'BSQ' #Band Sequential
            phs = ts.load_mmap(iname, fwid, flen, quiet=args.verbose, map=mapname,
                            nchannels=2, channel=2, conv = ('UNW' in endianlist))
            
        else:
            raise ValueError('Undefined format for unw files.')
        
        #----------------------------------------------------------------
        # READ COHERENCE
        if args.cthresh > 0.0: #Load only if needed 
            if unwfmt in  ['GMT','GRD']:
                cor = ts.load_grd(cname, shape=(flen,fwid))
            elif unwfmt == 'ISCE': 
                cor = ts.load_mmap(cname, fwid, flen, quiet=args.verbose, map='BIP',
                        nchannels=1, channel=1, conv = ('COR' in endianlist))
            elif unwfmt == 'RMG':
                cor = ts.load_mmap(cname, fwid, flen, quiet=args.verbose, map='BIL',
                        nchannels=2, channel=2, conv = ('COR' in endianlist))
            elif unwfmt == 'FLT':
                cor = ts.load_mmap(cname, fwid, flen, quiet=args.verbose,map='BSQ',
                        conv = ('COR' in endianlist))
            else:
                raise ValueError('Undefined format for cor files. Maybe different from UNW')
        
        #----------------------------------------------------------------
        # MASK AND REFERENCING
        mask = np.ones((nlen,nwid)) 
        if args.cthresh > 0.0:
            # Coherence threshold
            mask[cor[y0:y1,x0:x1] >= args.cthresh] = 1.0
        
        mask[phs[y0:y1,x0:x1] == 0.0] = np.nan  # zeros to NaN
        phs = phs[y0:y1,x0:x1]*mask*scl         # cut and convert to millimeters !!
        
        #Reference and Save interferogram to HDF5 file 
        if args.deramp:
            ifgsraw[k,:,:] = stack.referenceIgram(phs)
        else :
            ifgs[k,:,:] = stack.referenceIgram(phs)
        
        #END iteration over interferograms
        
    #--------------------------------------------------------------------
    # READ and SAVE LATITUDE and LONGITUDE FILES
    if args.latfile is not None :
        if unwfmt in  ['GMT']:
            lat = ts.load_grd(args.latfile, shape=(flen,fwid))
            lon = ts.load_grd(args.lonfile, shape=(flen,fwid))
        else :
            lat = ts.load_mmap(args.latfile, fwid, flen, quiet=args.verbose, datatype=np.float64,
                                conv = ('LAT' in endianlist))
            lon = ts.load_mmap(args.lonfile, fwid, flen, quiet=args.verbose, datatype=np.float64,
                                conv = ('LON' in endianlist))

        if args.verbose:
            print('------------------------------------------------')
            print("mean LAT", np.mean(lat))
            print("mean LON", np.mean(lon))

        #reshape
        lat = lat[y0:y1, x0:x1]
        lon = lon[y0:y1, x0:x1]
        lat = lat.astype(np.float32)
        lon = lon.astype(np.float32)
        
        #save in compact form into a Binary file
        lout = open(os.path.join(args.outdir,"lat.flt"),'wb')
        lat.tofile(lout)
        lout.close()                        
        
        lout = open(os.path.join(args.outdir,"lon.flt"),'wb')
        lon.tofile(lout)
        lout.close()
    
    #--------------------------------------------------------------------
    # READ and SAVE Line of Sight
    if args.losfile is not None:
        if unwfmt in  ['GMT']:
            ang = ts.load_grd(args.losfile, shape=(flen,fwid)) #### never tested 
        else : 
            ang = ts.load_mmap(args.losfile, fwid, flen, quiet=args.verbose, datatype=np.float32,map='BIL',
                                    nchannels=2, channel=1) # LOS incidence
            azi = ts.load_mmap(args.losfile, fwid, flen, quiet=args.verbose, datatype=np.float32,map='BIL',
                                    nchannels=2, channel=2) # LOS azimuth
            
        ang = ang[y0:y1, x0:x1]
        ang = ang.astype(np.float32)
        azi = azi[y0:y1, x0:x1]
        azi = azi.astype(np.float32)
        
        if args.verbose:
            print("mean LOS incidence", np.mean(ang),np.std(ang))
            print("mean LOS azimuth", np.mean(azi),np.std(azi))
            print("save files inc.flt and azi.flt in ",args.outdir)

        lout = open(os.path.join(args.outdir,"inc.flt"),'wb')
        ang.tofile(lout)
        lout.close()
        
        lout = open(os.path.join(args.outdir,"azi.flt"),'wb')
        azi.tofile(lout)
        lout.close()
    
    #--------------------------------------------------------------------
    # READ and SAVE DEM
    if args.demfile is not None:
        if unwfmt  == 'GMT':
            hgt = ts.load_grd(args.demfile, shape=(flen,fwid))
        elif unwfmt == 'ISCE': ## adapted for ISCEstackS1
            hgt = ts.load_mmap(args.demfile, fwid, flen, datatype=np.float64, quiet=args.verbose, map='BIP',
                        nchannels=1, channel=1, conv = ('DEM' in endianlist))
        elif unwfmt=='RMG':
            hgt = ts.load_mmap(args.demfile, fwid, flen, quiet=args.verbose, map='BIL',
                        nchannels=2, channel=2, conv = ('DEM' in endianlist))
        elif unwfmt=='FLT':
            hgt = ts.load_mmap(args.demfile, fwid, flen, quiet=args.verbose, map='BSQ', 
                        conv = ('DEM' in endianlist))
        else:
            raise ValueError('Unknown DEM format.')
        
        hgt = hgt[y0:y1, x0:x1]
        hgt = hgt.astype(np.float64)
        
        if args.verbose:
            print("DEM min, max", np.min(hgt),np.max(hgt))
            print("save files hgt.rdr in ",args.outdir)

        lout = open(os.path.join(args.outdir,"hgt.rdr"),'wb')
        hgt.tofile(lout)
        lout.close()
    
    #--------------------------------------------------------------------
    # Deramp interferograms
    if args.deramp:
        print('------------------------------------------------')
        print("WARNING: deramping takes time")
        if args.verbose:
            if args.dref == [(0,0)]:
                print("Deramping on whole interferogram's surface")
            else:
                print("Points defining the polygon on which deramping is performed :",args.dref)
        
        #Create 
        f = h5py.File(outname,"w")
        f.attrs['help'] = 'All the raw data read from individual interferograms into a single location for fast access.'
        ifgs = f.create_dataset('igram',(Nifg,nlen,nwid),'f')
        ifgs.attrs['help'] = 'Unwrapped IFGs after deramping'
        
        #Parse raw and deramp interferogram stacks as H5py dataset to save memory
        stack.deramp(ifgsraw, ifgs, network=True, poly=3, dref=args.dref) 
        ftmp.close() 
        
    #--------------------------------------------------------------------
    # Finilize storage 
    
    if args.baselinedir is not None:   
        g = f.create_dataset('bperp',data=np.array(baselineList))
        g.attrs['help'] = 'Array of baseline[:]s.'
    
    g = f.create_dataset('Jmat',data=stack.Jmat)
    g.attrs['help'] = 'Connectivity matrix [-1,1,0]'
    
    g = f.create_dataset('tims',data=tims)
    g.attrs['help'] = ' Array of SAR acquisition times.'
    
    g = f.create_dataset('dates',data=stack.days)
    g.attrs['help'] = 'Ordinal[:]s of SAR acquisition dates.'
    f.close()
