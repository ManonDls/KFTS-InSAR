############################################################################@
# Create interferograms, reshape for Kalman filter
# For 1D and 2D synthetic testing
# works in python 2.7 and 3.6
#
#Date : July-Nov 2018
#Author : Manon Dalaison 
###########################################################################@

#global import 
from builtins import str
from builtins import range
from builtins import object
import numpy as np

#Local import
from kf.timefunction import TimeFct
from kf.utils.generateRnAtmo import generateAtmo

class SynteticKF(object):
    def __init__(self, time):
        ''' 
        Class to initialise synthetic InSAR data to use with the Kalman Filter. 
        It mirrors the functions of the class ``kf.readinput.SetupKF`` 
        used for real observations. 
            * time : array 
                    aquisitions times
        '''
        self.time = time
        self.date = time
        self.orddates = time
        
        #useless parameter here, only to match MPI with real data
        self.rank = 0
        self.miny = 0
        
    def create_timeseries(self, model, m_r, sig_y, origintime=0.0, atmo=False):
        '''
        Create synthetic time series 
        
        Parameters
        ----------
            * model : list of tuples
                    a model with the appropriate shape for the class TimeFct
            * m_r : array/list
                    multiplicating factors for each element of the functional model
            * sig_y : float      
                    std of noise on time series
            * origintime : float, optional
                    specify initial time if not zero
            * atmo : boolean, optional       
                    add atmospheric noise? works for 2D only
        Returns:
            * the reference phases (without noise) 
            * the noisy phases 
            * the array of reference parameters with sin and cos amplitudes 
            transformed into amplitude and phase shift of oscillation
        '''
            
        #N = len(self.time)     #number of dates
        #L = len(m_r)         #number of parameters
        m_r = np.array(m_r)
        
        fct = TimeFct(self.time, model, origintime=origintime)
        fct.check_model()
        m_ph,tmp = fct.comp_phase_shift(m_r)
        
        Ref   = fct.draw_model(m_r)       
        Phas  = Ref +np.random.normal(0,sig_y,np.shape(Ref)) 
        
        if Ref.ndim > 1 : 
            if atmo :
                #Add atmospheric noise to acquisitions 
                atmnoise = generateAtmo((Ref.shape[1],Ref.shape[0]),\
                                    sig_y,8.,nt=len(self.time))
                Phas += atmnoise 
                
        # Set first phase to Zero
        shift        = -Phas[...,0] 
        m_r[...,0]   = shift 
        m_ph[...,0]  = shift 
        
        if (isinstance(shift,float) or isinstance(shift,int)):
            Phas  += shift
            Ref   += shift
        elif shift.ndim == 2 : 
            phas  = [Phas[:,:,i]+shift for i in range(np.shape(Phas)[-1])]
            ref   = [Ref[:,:,i]+shift for i in range(np.shape(Ref)[-1])]
            Phas  = np.transpose(phas, (1,2,0))    #put back defo on last index
            Ref   = np.transpose(ref, (1,2,0))
                
        else :
            assert False, "Format not understood"+ str(shift)
        
        
        self.ref, self.phase = Ref, Phas
        
        return self.ref, self.phase, m_ph
        
    
    def create_interfero(self, rr, sig_i, t_sep=5, perp_dist=200, fmt='diag'):
        '''
        Create interferograms for synthetic phases with temporal and 
        perpendicular baseline constraints.
            * rr    : float       
                    variance in data (std**2)
            * sig_i : float    
                    std of noise on phase differences
            * t_sep : integer, optional     
                    time separation allowed
            * perp_dist : float, optional
                    separation of perp baseline (generated randomly with std 200) authorise
            * fmt : string, optional      
                    define format of matrix R ('diag','common_dates','num_links')
        Returns:
            * interferograms (difference of phases) 
            * covariance matrix of interferograms 
            * the matrix of links between phases (# interf x # functional parameters and phases)
        '''
        Links = []      
        self.pair = []
        N = len(self.time)
        
        self.max_tsep = t_sep
        
        # Generate synthetic perpendicular baseline
        self.bperp  = np.random.normal(0,200,N)
        
        for k in range(0,N):
            for j in range(0,k):
                if abs(k-j) <= t_sep :            #timesep diff max
                    dist = abs(self.bperp[k]-self.bperp[j])
                    if dist <= perp_dist :        #if dist between tracks not too big 
                        H_line = np.zeros(N)  
                        H_line[k] = 1
                        H_line[j] = -1
                        if len(Links) == 0 :
                            Links = [H_line]
                        else : 
                            Links = np.vstack((Links,H_line))
                        
                        self.pair.extend([[j,k]])
        
        '''# Add long-temporal connections
        i = 20          #start at phase ...
        i_amp = 20      #go back by ... dt
        i_step = 10     #every ... dt
        while i < Links.shape[1] : 
            new_interf = np.zeros(Links.shape[1])
            new_interf[i] = 1
            new_interf[i-i_amp] = -1                 
            Links = np.vstack((Links,new_interf))
            self.pair.extend([[i-i_amp,i]])
            i += i_step  '''                              
        
        # To set first phase to zero
        self.links = Links
        #self.links[:,0] = 0 
        
        self.pair = np.array(self.pair)
        self.build_R(rr,fmt)   
        
        # Build interferograms
        if self.phase.ndim == 1 :
            interf= np.dot(self.links,self.phase) \
                    + np.random.normal(0,sig_i,np.shape(self.links)[0])
        
        if self.phase.ndim == 3 : 
            ny,nx = self.phase.shape[:2]
            self.phase = np.reshape(self.phase,(ny*nx,len(self.time)))
            interf = [np.dot(self.links,self.phase[j,:]) for j in range(nx*ny)] \
                +np.random.normal(0,sig_i,(nx*ny,np.shape(self.links)[0]))
        
            #reshape interferograms
            self.igram = np.reshape(interf.T,(interf.shape[-1],ny,nx))
            
            #save dimensions
            self.Nx,self.Ny,self.Ntot = nx, ny, ny
        
        #All done
        return interf, self.R, self.links
        
    def build_R(self,rr,fmt):
        '''
            * rr : float  
                variance (std**2) in observation (=interferograms) noise (float)
            * fmt : string
                define format of matrix R ('diag','common_dates','num_links')
        '''
        
        pair = self.pair
        N = np.shape(self.links)[0]
        R = np.zeros((N,N)) 
        
        if fmt =='diag':
            R = rr*np.eye(N)
            
        elif fmt =='common_dates':
            R[-1,-1] = rr
            for j in range(0,N):
                for q in range(0,N):
                    R[q,j] = 1/2.*rr*(kdelta(pair[q][0],pair[j][0])  
                                        +kdelta(pair[q][1],pair[j][1])
                                        -kdelta(pair[q][0],pair[j][1])
                                        -kdelta(pair[q][1],pair[j][0]))
                    R[j,q] = R[q,j]  #symmetry 
                
        elif fmt == 'num_links': 
            indx = np.sort(pair.flatten()).tolist()
            dic  = {x:indx.count(x) for x in indx} 
            rrs  = rr*np.max(list(dic.values())) /np.array(list(dic.values()))  #the more connection, the smaller the uncertainty
            
            for j in range(0,N):
                for q in range(0,N):
                    R[q,j] = rrs[pair[j][0]]*(kdelta(pair[q][0],pair[j][0]) \
                                    -kdelta(pair[q][1],pair[j][0])) \
                                    +rrs[pair[j][1]]*(kdelta(pair[q][1],pair[j][1]) \
                                    -kdelta(pair[q][0],pair[j][1]))
                                        
                    R[j,q] = R[q,j]  #symmetry 
            
            
        else :
            assert False,'do not understand format (fmt) of R'
            
        self.R = R
        
    def get_interf_pairs(self):
        ''' 
        OBSOLETE as use pairs (15/10/2018) ?
        extract indices of phases substracted together to build interfero
            * imoins : indice of phases substracted to iplus (M)
            * iplus  : indices of phases added  (M)
        '''
        Connect = np.array([[np.flatnonzero(self.links[i,:]==-1.)[0], \
                            np.flatnonzero(self.links[i,:]== 1.)[0]] for i in range(self.links.shape[0])])
    
        self.imoins = Connect[:,0].astype(int)
        self.iplus  = Connect[:,1].astype(int)
        
    
    def slice_data(self,N_new): 
        '''
        Slice previously generated array (Optional) 
        (keeps same sequence of random numbers)
        '''
        self.time = self.time[:N_new]
        self.ref, self.phase = self.ref[:N_new],self.phase[:N_new] 
        #self.bperp = self.bperp[:N_new] 
        
        return self.time, self.ref, self.phase
    
##########################################################################
def kdelta(i,j):
    '''kronecker delta function'''
    if i == j : 
        return 1.0
    else :
        return 0.0
    
