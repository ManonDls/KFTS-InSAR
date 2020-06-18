#!/usr/bin/env python

# Global stuff
import numpy as np
import matplotlib.pyplot as plt
import operator
import os 
import sys  
import h5py
import time as TIME 

# Local stuff
import kf
from kf.KF_class import *
import kf.readinput as infmt
from kf.timefunction import TimeFct

# To Parametrize
import configparser
import argparse
from ast import literal_eval

# MPI 
from mpi4py import MPI

class RunKalmanFilter(object):
    '''
    Class to run the full KFTS-InSAR processing chain
    Read configuration and data. Setup parameters. Run KF for each pixel.
    '''
    
    def __init__(self, config):
        plt.close('all')           # close all figures
        self.start_time = TIME.time()   # record running time
        self.initMpi()
        self.setConfiguration(config)

    def initMpi(self):
        ''' Initiate communication of the Message Passing Interface (MPI)'''
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        

    def setConfiguration(self, config):
        '''Read configuration file and convert to python objects.
            :config: open config file (.ini format)
        '''
        
        loc     = config['INPUT'].get('workdir', fallback='./')
        self.infile  = loc + config['INPUT'].get('infile')
        self.fmtfile = config['INPUT'].get('fmtfile', fallback='ISCE')
        self.instate = config['INPUT'].get('instate', fallback=None)
        self.eqinfo  = config['INPUT'].get('eqinfo', fallback=None)

        if self.instate is not None:
            self.instate = loc + self.instate
        if self.eqinfo is not None:
            self.eqinfo = loc + self.eqinfo

        self.outdir = loc + config['OUTPUT'].get('outdir', fallback='')
        self.figdir = loc + config['OUTPUT'].get('figdir', fallback='')

        secMS      = config['MODEL SETUP']
        self.EQ    = secMS.getboolean('EQ', fallback = False)
        freq       = secMS.getfloat('freq', fallback = 2*np.pi)
        self.sig_y = secMS.getfloat('sig_y', fallback = 10.0)
        self.sig_i = secMS.getfloat('sig_i', fallback = 0.01)
        
        try :
            self.model = literal_eval(secMS.get('model'))
        except : 
            if self.rank == 0:
                print("WARNING: model unreadable in config section [MODEL SETUP], use default")
            self.model = [('POLY',1),('SIN',freq),('COS',freq)]

        self.sig_a = literal_eval(secMS.get('sig_a'))
        self.sig_a = list(self.sig_a)
        self.sig_eq = secMS.getfloat('sig_eq', fallback=30.)
        
        secKFS       = config['KALMAN FILTER SETUP']
        self.VERBOSE = secKFS.getboolean('VERBOSE', fallback = False)
        self.PLOT    = secKFS.getboolean('PLOT', fallback = False)
        self.UPDT    = secKFS.getboolean('UPDT', fallback = False)
        self.pxlTh   = secKFS.getint('pxlTh', fallback = 1)
        
        if self.isTraceOn():
            print("Functional model string is: {}".format(self.model))
            print("Exclude pixels is less than {} interferograms".format(self.pxlTh)) 

        ## Optional section
        # initialize
        self.subregion = None
        
        # read if exists
        if config.has_section('FOR TESTING'):
            secFT = config['FOR TESTING']
            SUBREGION = secFT.getboolean('SUBREGION', fallback = False)
            if SUBREGION:
                x1,x2,y1,y2 = literal_eval(secFT.get('limval',fallback = '0,0,0,0'))
                self.subregion = infmt.Subregion(x1, x2, y1, y2)
                if self.rank == 0:
                    print("WARNING: select subregion", x1, x2, y1, y2)
    
    def readData(self):
        '''Initiate the data class dealing with interferograms, time,
        spatial grid and mask. It
            #. reads data
            #. divide the grid between workers (if MPI)
            #. build data covariance and
            #. store information
        '''
        
        # Read data file
        data = infmt.SetupKF(
            self.infile, 
            mpi = True, 
            mpiarg = (self.rank,self.size), 
            fmt = self.fmtfile, 
            verbose = self.VERBOSE,
            subregion = self.subregion
            )

        # Chose subset around fault 
        #data.select_pxl_band(x,y,0.48,-750,-650) 
        
        # Record indexes of empty pixels
        data.pxl_with_nodata(thres = self.pxlTh, plot = self.PLOT)
        
        # Get indices pairs used to build interferograms
        data.get_interf_pairs()
        
        # Build matrix of uncertainty on data
        data.create_R(np.square(self.sig_i))  
        
        # Print properties of data and keep it stored in there
        data.summary()
        
        return data

    def earthquakeIntegration(self):
        ''' Add step function to the functional model of deformation
        to model coseismic displacement due to earthquakes.
        Require a file containing earthquake properties in the track
        reference frame (see earthquake2step.py).
        '''
        
        def twoD_Gauss(x, y, amp0, center = (0.,0.), sig = 1.0):
            '''Compute a gaussian at coord x,y'''
            x0, y0 = center
            Function = amp0 *np.exp(-((x0-x)**2 + (y0-y)**2)/(2. * sig**2))
            return Function

        # Get earthquake properties in actual reference frame
        Xeq,Yeq,teq,Rinf,sx,sy = np.loadtxt(self.eqinfo, unpack=True)
        Xeq = Xeq.astype('int32')
        Yeq = Yeq.astype('int32')
        
        if isinstance(teq,float):
            #specific case of one earthquake only
            Leq = 1
            teq = [teq]
            Xeq,Yeq = [Xeq],[Yeq]
            Rinf,sx,sy = [Rinf],np.array(sx),np.array(sy)
        elif isinstance(teq,np.array):
            teq = teq.tolist()
            Leq = len(teq)
        else:
            assert False, "Verify content of {}".format(eqinfo)
        
        teq.insert(0,'STEP')
        self.model.insert(len(self.model),tuple(teq))
        if self.isTraceOn():
            print('New functional model with earthquakes :', self.model)

        self.sig_a.extend(np.zeros(Leq))

        # Error per earthquake
        P0eq = np.zeros((data.Ny,data.Nx,Leq))
        for i in range(Leq):
            width = Rinf[i]/(np.mean([np.mean(sx),np.mean(sy)]))
            fct   = twoD_Gauss(data.xv[data.miny:data.maxy,:], data.yv[data.miny:data.maxy,:],
                            self.sig_eq**2, center=(Xeq[i],Yeq[i]), sig=width)
            fct[fct < 1.] = 0.
            P0eq[:,:,i] = fct

    def initCovariances(self, L):
        '''Create arrays for the initial state Covariance matrix (P0)
        and the process noise covariance (Q).
            :L: Initial length of the state vector
                (number of model parameter + 1 (for reference phase))
        '''
        
        P_par   = np.square(self.sig_a)
        m_err   = np.zeros(L)           # Incertitude sur parametres
        phi_err = 0.0                   # Incertitude sur interfero
        add_err = (self.sig_y)**2       # Uncertainty on newly added phase
        return P_par, m_err, phi_err, add_err

    def initPlot(self, data, figdir):
        '''Draw quick plots to visualize input data:
            * Data plot
            * baseline plot
            * spatial mask on pixel
        '''
            
        import kf.makeplot as mplt
        mplt.view_data(data.igram, range(np.shape(data.igram)[0]), figdir)
        mplt.view_mask(data.mask, figdir)
        mplt.plot_baselines(data.time, data.bperp, data.imoins, data.iplus, figdir)

    def isTraceOn(self):
        '''Print only if verbose activated and first parallel worker'''
        return self.VERBOSE and self.rank == 0 

    def launch(self):
        '''Combine procedures to prepare data and model, then
        launch the Kalman filter class on each pixel row by row
        '''
        
        ## Setup
        # Start class dealing with data
        data = self.readData()
        
        # Add earthquake to model
        if self.EQ == True:
            self.earthquakeIntegration()
        
        # Start class dealing with the functional model
        tfct = TimeFct(data.time,self.model)
        tfct.check_model(verbose = self.VERBOSE)
        L = len(self.sig_a)
        
        # Check right number of model a priori error
        if tfct.L == L:
            # Add zero to apriori uncertainty as first phase is certain
            self.sig_a.extend([0])
        elif tfct.L == (L-1): 	
            if self.sig_a[-1] == 0: 
                L -= 1
            else : 
                assert False, "sig_a has an unexpected extra value, reduce length by one"
        elif tfct.L<(L-1) :
            assert False, "Too many uncertainties specified (%d) in sig_a compare to model requirement (%d)"%(L,tfct.L)
        elif tfct.L>L:
            assert False, "Too few uncertainties specified (%d) in sig_a compare to model requirement (%d)"%(L,tfct.L)
        
        P_par, m_err, phi_err, add_err = self.initCovariances(L)
        
        if self.PLOT : 
            self.initPlot(data, self.figdir)
            
        #---------------------------------------------------------------------------
        ## Kalman Filter
        kf_start = TIME.time()
        m0 = np.zeros(L)
        P0 = P_par*np.eye(L+1)
        
        toverl = 0
        if self.UPDT == True:
            # Import common things to all pixels to gain time 
            finstate  = h5py.File(self.instate,'r')
            statetime = finstate['tims'][:]     #contains as many dates than state contains phases
            indxs     = finstate['indx'][:].tolist()
            
            # Identify new dates with respect to former state
            newdate = [t for t in data.time if t not in statetime]
            
            # Check consistency 
            assert (len(newdate)  > 0), "No supplementary date in file {} wrt existing {}".format(infile,instate)
            assert (statetime[0] <= data.time[0]), "New data involves timesteps older than what was retained in the former state"
            
            # Set number of phases to keep for latter update
            data.max_tsep = np.max([data.max_tsep,len(statetime)])
            toverl = len(statetime) - len(data.time) + len(newdate)
            
            if self.isTraceOn():
                print('Data contains {} new time steps'.format(len(newdate)))

        if self.isTraceOn():
            print('-- Open H5 file for storage --')

        fstates, fphases = infmt.initiatefileforKF(
            self.outdir + 'States.h5', 
            self.outdir + 'Phases.h5', 
            L, 
            data, 
            self.model, 
            (m_err,self.sig_i,self.sig_y), 
            comm = self.comm, 
            toverlap = toverl
            )

        if self.isTraceOn():
            print('-- START Kalman Filter --')

        # Loop on each line
        for j in range(data.Ny): 
            
            if self.rank == 0:
                sys.stdout.write('\r {} / {}'.format(j,data.Ny))
                sys.stdout.flush()

            # Create the kalman filters for all columns
            kalmans = [Kalman(data, tfct, j = j, i = jj, verbose = self.VERBOSE) for jj in range(data.Nx) if data.mask[j,jj]>0.]
               
            # Create the filters for all columns of this line
            for kal,jj in zip(kalmans,range(data.Nx)):
                if self.EQ == True :
                    diagP0 = np.diag(P0).copy()
                    diagP0[-(Leq+1):-1] = P0eq[j,jj,:]
                    P0 = np.diag(diagP0)

                if self.UPDT == True : 
                    kal.restart_from_file(finstate, statetime, indxs)
                else :
                    kal.start_new(m0, P0)
                    
                kal.kf(m_err, phi_err, add_err, plots = False, t_sep = data.max_tsep)
                    
                # Store outputs
                kal.write_output(fstates, fphases)

        # Close file
        if self.isTraceOn():
            print('-- Close H5 files in {} --'.format(self.outdir))
        fstates.close()
        fphases.close()
        if self.UPDT == True:
            finstate.close()
        
        print('Time for Kalman filter : {}'.format(TIME.time() - kf_start))


if __name__ == '__main__':
    
    #First get file name from inline argument
    parser = argparse.ArgumentParser( description = 'InSAR Time series analysis with a Kalman Filter')
    parser.add_argument(
        '-c', 
        type=str, 
        dest = 'config', 
        default = None,
        help = 'Specify INI config file containing at least sections INPUT,OUTPUT, MODEL SETUP, KALMAN FILTER SETUP'
        )
    args = parser.parse_args()
    while (not args.config):
        print('Please choose a config file')
        exit()
    config = configparser.ConfigParser(interpolation = configparser.ExtendedInterpolation())
    config.read(args.config)
    runner = RunKalmanFilter(config)
    runner.launch()
