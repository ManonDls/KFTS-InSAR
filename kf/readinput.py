from __future__ import print_function
############################################################################@
# Read files with interferograms, select study area, reshape for Kalamn filter
#
#Date : July 2018
#Author : Manon Dalaison 
###########################################################################@

from builtins import zip
from builtins import range
from builtins import object
import numpy as np
import h5py, sys
import datetime as dt

#local load only to convert MintPy format
from prepare_input import BuildStack

#os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

###########################################################################
# WORK FROM DATA
class Subregion():
    def __init__(self, x1, x2, y1, y2):
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2

class SetupKF(object):

    def __init__(self, h5file, fmt='ISCE', comm=False, mpiarg=0, utime = 'years',
                        verbose=True, subregion=None, cohTh=None, refyx=None):
        '''
        Class for reading and modifying interferograms for Kalman filtering.
            :h5file:    .h5 file path containing
                
                - time  : decimal dates relative to first acquisition (usually in years)
                - dates : absolute data
                - igram : interferograms
                - links : connection between phases to build interfero (M x N), 0 1 and -1
                - bperp : perpendicular baseline between aquisitions (not exploited yet)
        Opts:
            :fmt:       format of H5file input, default is 'ISCE' (also 'RAW' 'MintPy')
            :comm:      do you use parallel features of mpi4py (True or False, default False)
            :mpiarg:    precise rank and size of communicator (tuple used if mpi=True)
            :utime:     time unit as string (years or days)
            :verbose:   print stuff? (True or False)
            :subregion: subregion class instance containing x and y bounds as pixel numbers
            :cohTh:     minimum coherence used to mask pixels in each interferogram
        '''

        if comm == False:
            mpi=False
        else :
            mpi=True

        self.comm = comm
        self.verbose = verbose
        
        ## Import and read Data
        if fmt.upper() == 'ISCE':
            intrfname = 'figram'
        elif fmt.upper() == 'RAW':
            intrfname = 'igram'
        elif fmt.upper() == 'MINTPY':
            #Select info from ifgramStack
            intrfname = 'unwrapPhase'
        else :
            assert False,"Specified format {} of H5file not known".format(fmt)
        
        if not mpi :
            fin = h5py.File(h5file,'r') #dictionary
        else : 
            fin = h5py.File(h5file,'r',driver='mpio',comm=self.comm)

        self.Ny, self.Nx = np.shape(fin[intrfname])[1:]
        self.igram       = fin[intrfname]
        
        if subregion is None :
            self.spatial_grid()
        else :
            self.spatial_grid(xmin = subregion.x1, xmax = subregion.x2, 
                        ymin = subregion.y1, ymax = subregion.y2, truncate = True)
        
        #Slice data between workers along Y (axis 1)
        self.dividepxls(mpi,mpiarg)
        
        if (refyx is not None) or (cohTh is not None):
           #modification of input igram is required 
           self.copydata2file(fin)
           if refyx is not None:
               self.rereference(refyx)

        if  fmt.upper() == 'MINTPY':
            self.bperp       = fin['bperp'][:] 
            self.mintpy2kfts(fin)
        else : 
            self.time        = fin['tims'][:]                       #time in decimal year wrt start 
            self.links       = fin['Jmat'][:]                       #2D (interf,time)
            self.bperp       = fin['bperp'][:]                      #perpendicular baseline (interf)
            self.orddates    = fin['dates'][:].astype(int)          #ordinal dates
        
        if subregion is None: 
            self.igram   = self.igram[:,self.miny:self.maxy,:]     #3D (interf,y,x)
        else :
            self.igram   = self.igram[:,subregion.y1:subregion.y2,
                                            subregion.x1:subregion.x2]
        # Apply coherence threshold
        if cohTh is not None:
            self.filter_by_coherence(fin,cohTh)
        
        # Ordinal date to decimal year
        init      = dt.datetime.fromordinal(self.orddates[0])
        yr_start  = dt.date(init.year,1,1).toordinal()
        yr_len    = dt.date(init.year+1,1,1).toordinal() - yr_start
        day_inyr  = dt.date(init.year,init.month,init.day).toordinal() - yr_start
        t0        = init.year + day_inyr/yr_len 
        self.date =  t0 + self.time[:]

        if utime == 'years':
            if self.verbose: 
                print('Time expressed in years')  
        elif utime == 'days':
            # Ordinal dates in days already
            if self.verbose :
                print('WARNING: time is in number of days')
            self.time   = self.orddates - self.orddates[0]
        else : 
            assert False, "unit of time {} not understood".format(utime)
        self.unit = utime 


    def dividepxls(self, mpi, mpiarg):
        '''
        Check if MPI used and divide pxls into subsets for different workers
        '''
        # store total amount of lines
        self.Ntot = self.Ny
        
        if mpi :
            self.mpi = mpi
            self.rank, size = mpiarg
            
            #select number of line per worker
            if self.Ntot > size :
                Yslice = int(self.Ny/size)
            else :
                Yslice = 1

            if self.verbose:
                print('There are {} columns, each worker will deal with about {} columns'.format(
                                                      self.Ntot,Yslice))
            #select subset along latitudes (y)
            miny = self.rank * Yslice
            assert (miny < self.Ntot),"Worker {} has no columns to work with, STOP".format(self.rank)
            
            if self.rank < (size-1):
                maxy = miny + Yslice
            else:
                maxy = self.Ntot

            self.Ny = maxy -miny
            self.miny,self.maxy = miny,maxy
            
            if self.rank in [0,size]: 
                print('Worker {} working on {} to {}'.format(self.rank, miny, maxy))
            
        else :
            self.mpi = False
            self.rank, size = 0,1
            self.miny,self.maxy = 0, self.Ny
                
            
    def trunc_time(self,t_num) :
        ''' 
        Select subset of dates.
            :t_num: the number of dates to keep at the end of array
        '''
        
        if self.verbose:
            print('Truncate time :keep last', t_num, 'aquisitions') 
        
        self.time  = self.time[-t_num:]
        self.links = self.links[:,-t_num:]

        #remove lines not containing both 1 and -1
        mask = np.logical_and( (self.links==1).any(axis=1), (self.links==-1).any(axis=1) )

        self.links = self.links[mask,:]
        self.bperp = self.bperp[mask]
        self.igram = self.igram[mask,:,:]
    
    def spatial_grid(self, xmin=0, xmax=0, ymin=0, ymax=0, truncate=False) :
        '''
        Create spatial grid with possibility of choosing a spatial subset.
        
        Opts:
            :xmin,xmax,ymin,ymax: indexes delimiting the study area (integers)
                            default are all pixels (0,Nx,0,Ny)   
            :truncate: True or False (for quick testing)

        Return: 
            * meshgrid in x and y (2 2D arrays)
        '''
        
        if (xmax==0) and (ymax==0) :
            xmax = self.Nx
            ymax = self.Ny
        
        nx,ny           = xmax-xmin, ymax-ymin
        y,x             = np.array(list(range(ny))),np.array(list(range(nx)))    
        self.yv,self.xv = np.meshgrid( ymin+y, xmin+x, indexing='ij')
        
        if (truncate and (nx*ny < self.Nx*self.Ny)):
            self.yv,self.xv = np.meshgrid(y, x, indexing='ij')
            self.Ny,self.Nx = ny, nx 
            self.Ntot = ny

        return x,y


    def mintpy2kfts(self,fin):
        '''
        Read and translate information in the hDF5 output of MintPy named ifgramStack.h5
        Essentially reformat dates and build matrix mapping interferogram to time
            :fin:  open HDF5 file 
        '''
        
        # initialize class 
        ylims,xlims = (0,-1), (0,-1)
        stack = BuildStack(self.verbose,ylims,xlims)
        
        # read date strings 
        datestrings = fin['date'][:]
 
        stack.ConnectMatrix(datestrings.astype(str))
        self.links = stack.Jmat     # get matrix
        self.orddates = stack.days.astype(int)  # ordinal dates 
        self.time = (stack.days-stack.days[0])/365.25    # decimal years 
    
    def copydata2file(self,fin):
        '''
        Create new file to copy and edit data (interferograms) 
        in a safe and memory-saving way before splitting tasks between workers
        '''
        # create new file for iterative writing and storage
        newstackfile = fin.filename[:-3]+'_copy_igram.h5'
        if not self.mpi :
            fout = h5py.File(newstackfile,'w')
        else : 
            fout = h5py.File(newstackfile,'w', driver='mpio', comm=self.comm)
            
        if self.verbose:
            print('Create standardized source file {}'.format(newstackfile))  
        
        #dataset
        maskigram = fout.create_dataset('igrams',(self.igram.shape),'f')
        maskigram[:,self.miny:self.maxy,:] = self.igram[:,self.miny:self.maxy,:]
        self.finmask = fout
    
    def rereference(self,ref):
        '''
        Remove the interferoùetric value on a given pixel to all interferograms
        (optional if interferogram stack already referenced)
            :ref: (y,x) pixel index of reference
        '''
        if self.verbose:
            print('-- Re-reference all interferograms to a single pixel {} --'.format(ref))
        
        if hasattr(self, 'finmask'): 
            maskigram = self.finmask['igrams']
        else:
           assert False, "Need to run copydata2file BEFORE filter_by_coherence"
        
        for j in range(self.igram.shape[0]): 
           maskigram[j,:,:] -= maskigram[j,ref[0],ref[1]]
        
        self.igram = maskigram
        
    def filter_by_coherence(self,fin,cohTh):
        '''
        Remove points below a given threshold in all interferograms
            :fin: open HDF5 file containing a 'coherence' dataset
        '''
        
        if self.verbose:
            print('-- Masking low coherence pixel in each interferogram --')
            print('Coherence threshold is {}'.format(cohTh))
        
        if hasattr(self, 'finmask'): 
            maskigram = self.finmask['igrams']
        else:
           assert False, "Need to run copydata2file BEFORE filter_by_coherence"

        for j in range(0, self.Ny): #each worker iterates over its own Y
            # load coherence
            cohsub = fin['coherence'][:,self.miny+j,:]
            igrsub = self.igram[:,j,:]
            
            # replace  by NaN
            mask = (cohsub < cohTh)
            igrsub[mask] = np.nan
            
            # write to file
            maskigram[:,self.miny+j,:] = igrsub

        self.igram = maskigram[:,self.miny:self.maxy,:]

    def apply_input_mask(self,maskfile):
        '''
        * maskfile:  hdf5 file containing a 'mask' dataset of (Y,X) shape
        '''
        
        if self.verbose:
            print('-- Masking pixels according to a loaded mask --')
            print('Input mask is in {}'.format(maskfile))

        # open file
        if not self.mpi :
            fin = h5py.File(maskfile,'r')
        else :
            fin = h5py.File(maskfile,'r',driver='mpio',comm=self.comm)
        
        # create mask of right size (worker specific if MPI)
        mask = np.zeros(self.igram.shape[1:]) 
        mask[:] = fin['mask'][self.miny:self.maxy,:] 
        fin.close()
        
        if hasattr(self,'mask'):
            self.mask *= mask
        else:
            self.mask = mask


    def pxl_with_nodata(self,thres=30, chunks=200, plot=False) :
        '''
        Check for pixels with little data and build mask.
            :xv,yv: grid element 
        Opts:
            :thres:  minimum number of interferogram required to be available for one pixel
            :chunks: chunks of columns to be processed simultaneously
        Return:
            * pairs of indices in list 
        '''
        
        # Make mask
        if self.verbose:
            print('-- Masking elements with little data --')
        
        mask = np.zeros(self.igram.shape[1:]) #(dy,x)
        
        for j in range(0, self.Nx, chunks):    #itterate through blocks of 
            end = np.min([self.igram.shape[2], j+chunks])
            mask[:,j:end] = np.sum(np.isfinite(self.igram[:,:,j:end]), axis=0)  #sum counts number of True
        
        if plot==True:
            import matplotlib.pyplot as plt
            fig,ax = plt.subplots(1,1)
            plt.pcolormesh(mask,vmin=200)
            plt.colorbar()
            plt.savefig('numb_data_map.png',dpi=250)
            plt.close()
            
        mask[mask<thres] = 0
        mask[mask>=thres] = 1
        
        if hasattr(self,'mask'):
            self.mask *= mask
        else:
            self.mask = mask
        
        print('Selected pixels :',int(np.sum(self.mask)),'so',\
                        round(np.sum(self.mask)/(float(self.Nx*self.Ny))*100.,1),'%',
                        "-for worker",self.rank)
                                
        return 
    
    def select_pxl_band(self,x,y,slope,Xoff1,Xoff2,xmin=0) :
        '''
        Chose subset around fault between two parallel lines (X = slope*Y + off).
            :slope: slope of bounds for X as a function of Y (float)
            :Xoff1: offset in minimum line (float)
            :Xoff2: offset in maximum line (Xoff2 > Xoff1)
        Opt:
            :xmin: if nonzero in spatial_grid function need to specify its value 
        
        NOT ADAPTED FOR MPI USE YET
        '''
        
        if self.verbose:
            print('-- Selecting fault zone --')
        
        mask = np.zeros(self.igram.shape[1:])
        
        for jj in y: #itterate over rows
            Xmin = slope*jj +Xoff1
            Xmax = slope*jj +Xoff2
            mask[jj,x+xmin > Xmin] = 1
            mask[jj,x+xmin > Xmax] = 0
        
        # update mask
        if hasattr(self, 'mask'):
            self.mask *= mask
        else:
            self.mask = mask
        
        if self.verbose:
            print(self.rank,'Selected pixels :',int(np.sum(mask)),'so',\
                            round(np.sum(mask)/(float(self.Nx*self.Ny))*100.,2),'%')

        return 
        
    def get_interf_pairs(self):
        ''' 
        Extract indices of phases substracted together to build interfero.
            * imoins : indice of phases substracted to iplus (M)
            * iplus  : indices of phases added  (M) 
        '''  
        Connect = np.array([[np.flatnonzero(self.links[i,:]==-1.)[0], \
                            np.flatnonzero(self.links[i,:]== 1.)[0]] for i in range(self.links.shape[0])])
    
        self.imoins = Connect[:,0].astype(int)
        self.iplus  = Connect[:,1].astype(int)
        
        self.max_tsep = int(max(((self.imoins-self.iplus)**2)**(1/2.)))
        
        if self.verbose:
            print('max step separation btwn interfero',self.max_tsep)
        
        return 

    def create_R(self, rr):
        ''' 
        Create covariance matrix of data 
            :rr: variance in observation (=interferograms) noise (float)
        '''
        
        imoins,iplus = self.imoins,self.iplus
        R = np.eye(len(imoins))* rr      
        #for q in range(0,len(imoins)):
        #    R[q,-1] = 1/2.*rr*(kdelta(imoins[q],imoins[-1])  
        #                        +kdelta(iplus[q],iplus[-1])
        #                        -kdelta(imoins[q],iplus[-1])
        #                        -kdelta(iplus[q],imoins[-1]))
        #    R[-1,q] = R[q,-1]           
        
        self.R = R

        return  

    def summary(self):
        '''Print data properties '''

        if self.verbose:
            init = dt.datetime.fromordinal(self.orddates[0])
            last = dt.datetime.fromordinal(self.orddates[-1])
            print("-- Data summary --")
            print('starting date is {}/{}/{} and last acquisition {}/{}/{}'.format(
                        init.year, init.month, init.day, last.year, last.month, last.day))
            print('timespan of measurements : ',self.time[-1] - self.time[0],'years')
            print('number of days with acquisitions : ', len(self.time))
            print('number of interferograms : ', len(self.bperp))
        
        print('total number of pixels : ',np.shape(self.igram)[1:],' -for worker',self.rank)
        
        #set first phase to zero
        #self.links[:,0] = 0  #can't write on h5py  
                                 
        return 
    
    
##########################################################################
def kdelta(i,j):
    '''kronecker delta function'''
    if i == j : 
        return 1.0
    else :
        return 0.0
        

def initiatefileforKF(statefile, phasefile, L, data, model, store, 
                           updtfile=None, comm=False, toverlap=0, tshift=1, t_sep= None):
    '''
    Open h5py file for kalman filter.
        :statefile: file name and location 
        :phasefile: file name and location 
        :L:         number of parameters 
        :model:     model description in tuple used for timefunction.py
        :store:     is a tuple of things to store
        :updtfile:  file name to store additional statistics about KF analysis
        :comm:      communicator if MPI used (e.g. MPI.COMM_WORLD)
        :toverlap:  number of overlaping timesteps with past solution (only if restart KF)
        :tshift:    number of previously estimated phases (may be updated)
    '''
    
   
    lent = data.time.shape[0] + toverlap #length of saved record for latter update
    newt = lent - tshift                 #length of new optimized time steps
    Ny, Nx = data.Ntot, data.Nx
    m_err, sig_eps, sig_gam  = store
    
    if comm==False :
        fstates = h5py.File(statefile, 'w')
        fphases = h5py.File(phasefile, 'w')
        if updtfile is not None:
            fupdt = h5py.File(updtfile, 'w')
    else : 
        fstates = h5py.File(statefile, 'w', driver='mpio', comm=comm)
        fphases = h5py.File(phasefile, 'w', driver='mpio', comm=comm)
        if updtfile is not None:
            fupdt = h5py.File(updtfile, 'w', driver='mpio', comm=comm)

    if t_sep is None:
        tsep = data.max_tsep
    else:
        tsep = t_sep
        
    ## Create output dataset (name,shape,datatype)
    # Elements to restart the kalman filter
    state = fstates.create_dataset('state',(Ny,Nx,L+tsep),'float64')
    state.attrs['help'] = 'State m comprising the model parameters and the last retrieved phases \n for model '+str(model)

    state_cov = fstates.create_dataset('state_cov',(Ny,Nx,L+tsep,L+tsep),'float64')
    state_cov.attrs['help'] = 'Covariance P of state m'
    
    indx = fstates.create_dataset('indx',(tsep,),'i8')
    indx.attrs['help'] = 'Indexes of the phases still within the state m (in parms)'

    subtime = fstates.create_dataset('tims',(tsep,),'float64')
    subtime.attrs['help'] = 'times corresponding to phases in state in decimal years with respect to first phase with ti= '+str(data.date[0])

    pn = fstates.create_dataset('processnoise',data=m_err)
    pn.attrs['help'] = 'Process noise for functional model parameters'

    mm = fstates.create_dataset('mismodeling',data=sig_gam)
    mm.attrs['help'] = 'Mismodeling error added as process noise on last phase estimate'

    mc = fstates.create_dataset('misclosure',data=sig_eps)
    mc.attrs['help'] = 'Misclosure error included in data covariance'

    # Save time series of deformation 
    rawts = fphases.create_dataset('rawts',(Ny,Nx,lent),'f')
    rawts.attrs['help'] = 'Reconstructed phases'

    rastd = fphases.create_dataset('rawts_std',(Ny,Nx,lent),'f')
    rastd.attrs['help'] = 'Reconstructed phases standard deviation (sqrt of diag(P))'
    
    tims = fphases.create_dataset('tims',(lent,),'f')
    tims.attrs['help'] = 'Decimal years of the time series with ti= '+str(data.date[0])
    
    dates = fphases.create_dataset('dates',data=data.orddates)
    dates.attrs['help'] = 'Ordinal values of the SAR acquisition dates'

    idx = fphases.create_dataset('idx0',data=0)
    idx.attrs['help'] = 'Index of first phase in file with respect to first reference date of time series'
    
    # Save part of the innovation and Gain to have information 
    # about the predictive power of the model 
    innv = fupdt.create_dataset('mean_innov',(Ny,Nx,newt),'f')
    innv.attrs['help'] = 'Mean innovation (or residual) for the last phase estimate at each time step'
    
    # about the convergence and sensitivity to data of model parameters
    gain = fupdt.create_dataset('param_gain',(Ny,Nx,newt,L),'f')
    gain.attrs['help'] = 'Norm of the gain for the L model parameters at each time step'

    if updtfile is not None:
        return fstates,fphases,fupdt
    else :
        return fstates,fphases


def reopenfileforKF(statefile, phasefile, comm=False):
    '''Function to reopen file after closure to test readability'''
    
    if comm==False :
        fstates = h5py.File(statefile, 'r+')
        fphases = h5py.File(phasefile, 'r+')
    else :
        fstates = h5py.File(statefile, 'r+', driver='mpio', comm=comm)
        fphases = h5py.File(phasefile, 'r+', driver='mpio', comm=comm)
    
    return fstates,fphases
