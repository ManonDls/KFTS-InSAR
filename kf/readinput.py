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

###########################################################################
# WORK FROM DATA
class Subregion():
    def __init__(self, x1, x2, y1, y2):
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2

class SetupKF(object):

    def __init__(self, h5file, fmt='ISCE', mpi=False, mpiarg=0, verbose=True, subregion=None):
        '''
        Class for reading and modifying interferograms for Kalman filtering.
            :h5file: .h5 file path containing
                
                - time  : decimal dates relative to first acquisition (usually in years)
                - dates : absolute data
                - igram : interferograms
                - links : connection between phases to build interfero (M x N), 0 1 and -1
                - bperp : perpendicular baseline between aquisitions (not exploited yet)
        Opts:
            :fmt:    format of H5file input, default is 'ISCE'
            :mpi:    do you use parallel features of mpi4py (True or False, default False)
            :mpiarg: precise rank and size of communicator (tuple used if mpi=True)
        '''
        
        self.verbose = verbose
        
        ## Import and read Data
        if fmt == 'ISCE':
            intrfname = 'figram'
        elif fmt == 'RAW':
            intrfname = 'igram'
        elif fmt == 'Etna':
            intrfname = 'igram_aps'
        
        else :
            assert False,'Format of H5file not known'
        
        fin = h5py.File(h5file,'r') #dictionary
        self.Ny, self.Nx = np.shape(fin[intrfname])[1:]
        self.igram       = fin[intrfname]
        
        if subregion is None :
            self.spatial_grid()
        else :
            self.spatial_grid(xmin = subregion.x1, xmax = subregion.x2, 
                        ymin = subregion.y1, ymax = subregion.y2, truncate = True)
        
        self.dividepxls(mpi,mpiarg)

        self.time        = fin['tims'][:]
        self.links       = fin['Jmat'][:]                              #2D (interf,time)
        self.bperp       = fin['bperp'][:]                             #perpendicular baseline (interf)
        self.orddates    = fin['dates'][:].astype(int)
        
        if subregion is None: 
            self.igram   = fin[intrfname][:,self.miny:self.maxy,:]     #3D (interf,y,x)
        else :
            self.igram   = fin[intrfname][:,subregion.y1:subregion.y2,
                                            subregion.x1:subregion.x2]
        
        #Ordinal date to decimal year
        init      = dt.datetime.fromordinal(self.orddates[0])
        yr_start  = dt.date(init.year,1,1).toordinal()
        yr_len    = dt.date(init.year+1,1,1).toordinal() - yr_start
        day_inyr  = dt.date(init.year,init.month,init.day).toordinal() - yr_start
        t0        = init.year + day_inyr/yr_len 

        self.date =  t0 + self.time[:]


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

            if self.verbose and self.rank==0:
                print('There are {} columns, each worker will deal with {} columns'.format(
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
            
            if self.verbose:
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
        if self.rank==0 and self.verbose:
            print('-- Masking elements with little data --')
        
        mask = np.zeros(self.igram.shape[1:]) #(interf,y,x)
        
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
        
        if self.verbose :
            print('Selected pixels :',int(np.sum(mask)),'so',\
                        round(np.sum(mask)/(float(self.Nx*self.Ny))*100.,1),'%',
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
        
        if self.rank==0 and self.verbose:
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
        
        if self.verbose and self.rank == 0:
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
        
        if self.rank==0 and self.verbose:
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
        ''' Print data properties'''

        if self.rank==0 and self.verbose:
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
                           updtfile=None, comm=False, toverlap=0, t_sep= None):
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
    '''
        
    lent = data.time.shape[0]
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
    
    if updtfile is not None:
        # Save part of the innovation and Gain to have information 
        # about the predictive power of the model 
        innv = fupdt.create_dataset('mean_innov',(Ny,Nx,lent-toverlap),'f')
        innv.attrs['help'] = 'Mean innovation (or residual) for the last phase estimate at each time step'

        # about the convergence and sensitivity to data of model parameters
        gain = fupdt.create_dataset('param_gain',(Ny,Nx,lent-toverlap,L),'f')
        gain.attrs['help'] = 'Norm of the gain for the L model parameters at each time step'

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
