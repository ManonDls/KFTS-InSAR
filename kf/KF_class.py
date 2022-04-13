# -*- coding: utf-8 -*-
############################################################################@
#Assimilation of measured phase : Kalman Filter for (synthetic) INSAR data 
#
#
#Date : February 2018
#Author : Manon Dalaison & Romain Jolivet
###########################################################################@
from __future__ import print_function
from __future__ import division
from builtins import str
from builtins import range
import numpy as np
import matplotlib.pyplot as plt
import operator
import os        #using operating system dependent functionality.
import h5py
import time as TIME

# Local stuff
#import insar_fct
#from timefunction import TimeFct

###############################################################################     
# A class for the Kalman filter
    
class Kalman(object):
    
    def __init__(self, data, fctmod, j=0, i=0, verbose=True):
        '''
        Class for a Kalman filter for an InSAR time series analysis
        Initialize the object
        
        * data : object 
                observations/measurements in class from readimput_mpi.py
        * fctmod : object 
                functional model in class from timefunction.py
        * j,i : integers
                indexes for 2-D image used for storage
        * verbose : boolean 
                print stuffs
        '''

        self.verbose = verbose

        ### Extract model from class
        self.modelobj  = fctmod
        self.model     = fctmod.model
        self.L         = fctmod.L
        
        ### Store essential information
        #self.dataobj = data
        self.data    = data.igram[:,j,i]
        self.t       = data.time
        self.link    = data.links[:]
        self.Rdat    = data.R
        
        self.link[:,0] = 0
        
        # For 2D store index of pixels
        self.xi = i
        self.yi = data.miny +j
        
        self.rank = data.rank
        
        assert (np.shape(self.link)[0] == np.shape(data.igram[:,j,i])[-1]), 'rows of links should correspond to numb of interfero'
    
    
    def restart_from_file(self,fin,pasttime,indxs):
        '''
        Extract initial condition from OPENED infile (fin)  which stores previously 
        computed mk and Pk for all pixels including pixel[i,j]
        
        * fin : object  
                opened H5 file containing formely computed state 
        * pasttime : array
                already loaded time array in fin
        * indxs : array
                already loaded index array
        '''
        
        i = self.xi
        j = self.yi
        
        # Load Previously computed state
        self.m        = fin['state'][j,i,:]        #3D (y,x,m)
        self.P        = fin['state_cov'][j,i,:,:]  #3D (y,x,P)
        
        # Keep strack of where we are with respect to first data of time series
        self.m_indxs  = indxs                             #indexes of phases in m after prediction phase 
        self.t        = np.concatenate((pasttime,self.t))
        self.t        = np.unique(self.t)                 

        if len(self.modelobj.t) < len(self.t): 
            self.modelobj.t = self.t

        self.phases   = []                         #phases already computed 
        self.std      = []                         #std of phases already computed
        
        # Check consistency
        len_m0      = self.m.shape[-1] -len(pasttime)     #number of parameters in the m stored, define initial model
        self.Lref   = self.L                              #max number of parameters as predicted by the model
        
        self.get_model_from_num_of_param(len_m0)
               
        if np.shape(self.link)[1] != len(self.t):  
            #WARNING: columns of links and time must be of same length, modify links
            self.link   = self.link[:,-len(self.t):]

    
    def start_new(self,m0, P0):
        '''
        Start from skratches
        
        * m0 : 1D array (N)
                Initial Model. The length of the vector will determine how many 
                element of the model will be kept (in the order given in the model vector) 
        * P0 : 2D array (N,N)
                Initial Covariance
        '''    
        self.m       = np.concatenate((m0,[0.0])) #first phase fixed to zero
        self.P       = P0
        self.m_indxs = [0]   #indexes of phases in m after prediction phase
        self.phases  = []    #store phases that converged and removed from m 
        self.std     = []    #std of phases already computed
                
        self.Lref   = self.L     #max number of model parameters (take it from TimeFct)
                        
        self.get_model_from_num_of_param(len(m0))
        assert (np.shape(self.link)[1] == len(self.t)), 'columns of links and time must be of same length'
         
    def get_model_from_num_of_param(self,N):
        ''' 
        Truncate model if the number of parameters in the input (N)
        is smaller than the maximum number of parameters as predicted 
        by the initial functional model (self.Lref). 
        N (or equivalently self.L) may increase with latter iterations'''
        
        assert N <= self.Lref, 'number of apriori parameters greater than what model predicts'
        
        if N < self.Lref:    
            print('Truncate model')
            self.mod = self.modelobj.cut_model(N) #new truncated model (list of tuples)
                                         #new self.L defined as N
      
    def create_Q(self, m_err, phi_err, add_err, M): 
        '''
        Create process covariance Q from uncertainty on model (m_err)
        and interferograms (phi_err) at kth time. 
        
        * m_err : float or an array of length L 
                model uncertainty
        * phi_err : float
                systematic error on phases (should be zero)
        * add_err : float
                systematic error on last forecast (square of std of mismodeling)
        * M : integer
                the state vector length
        '''
        
        L = self.L 
            
        if (isinstance(m_err,float) or isinstance(m_err,int)) : #check if float or int
            self.Q = np.concatenate((m_err*np.eye(M+1)[:L],phi_err*np.eye(M+1)[L:])) 
        elif len(m_err) >= L :
            Q1 = np.hstack((np.diag(m_err[:L]),np.zeros((L,M+1-L))))
            self.Q = np.concatenate((Q1,phi_err*np.eye(M+1)[L:]))
        else :
            assert False, 'format m_err not understood'
        
        if M >= L+1 : #not first iteration
            self.Q[-1,-1] = add_err  #large error associated to last a priori value
            
    def create_H_R_and_D(self, k, indxs):
        '''
        Produce the measurement vector (D), the measurement matrix (H), and 
        the measurement covariance matrix (R) at a specific timestep (0≤ k <N) 
        --> if len(D)=n for this timestep, then H will be (n x (L+k+1)) and R (n x n)
        
        * k   : integer
                itteration number
        * indx : integer
                indexes (with respect to t0) of phases in self.m[L:]
        '''
        
        
        # find interferogram for time k (line of links) involving past and present dates
        ind_interf = np.array([i for i,hh in enumerate(self.link[:,k]) if abs(hh)==1])
        
        
        if len(ind_interf)> 0 : 
            # check that interferogram does not involve future dates
            ind_interf = np.array([i for i in ind_interf if np.sum(self.link[i,k+1:])==0 ])

            # check for NaN in D[ind_interf]
            if len(ind_interf)> 0 :
                mask_nan  = np.isnan(self.data[ind_interf])
                ind_interf = ind_interf[np.invert(mask_nan)]
        
        if len(ind_interf)> 0 :
            
            #find phase substracted to phase k (column of links)
            ind_phases = np.array([i for i in np.where(self.link[ind_interf,:]==abs(1))[1] if i!=k])
            
            #If phase involved in interferogram not in state (m) anymore
            condition = [i not in indxs for i in ind_phases]
            if any(condition):  
                ind_old,im = ind_phases[condition],np.where(condition)[0]
                
                #expand m and P from phase 
                self.m = np.concatenate((self.m[:self.L],self.phases[ind_old],self.m[self.L:]))
                           
                for i in range(len(ind_old)):
                    var = np.square(self.std[ind_old[i]])
                    row = np.zeros(np.shape(self.P)[0])
                    col = np.zeros(np.shape(self.P)[0]+1)
                
                    #No covariance terms in P gives better results, but approximation
                    col[self.L] = var

                    self.P = np.insert(self.P,self.L+i,row,axis=0)
                    self.P = np.insert(self.P,self.L+i,col,axis=1)
            
                #notify new
                indxs = np.concatenate((ind_old,indxs)) #with respect to idx0
                self.m_indxs = np.concatenate((ind_old +self.idx0,self.m_indxs))
        
            #select relevant D and R
            self.D = self.data[ind_interf]
            self.R = self.Rdat[ind_interf,:][:,ind_interf]
            Hsub = self.link[ind_interf,:k+1]
            
            #resize if phases deleted 
            Hsub = Hsub[:,indxs]
            self.H = np.hstack((np.zeros((len(self.D),self.L)),Hsub))
        
        else : 
            self.D = []
            self.R = []
            self.H = []


    def predict(self,X, P, A, Q):
        '''
        Forecast step

        * X : array (N)
            The mean state estimate of the previous step ( k −1). 
        * P : array (N, N)
            The state covariance of previous step ( k −1).
        * A : array (N+1, N)
            The transition matrix.
        * Q : array (N+1, N+1)
            The process noise covariance matrix.
        '''
        
        Xf = np.dot(A, X)
        Pf = np.dot(A, np.dot(P, A.T)) + Q
        return(Xf,Pf)
    
    def update(self,Xf, Pf): 
        ''' 
        * Xf : array  
            forecast mean of the state
        * Pf : array
            forecast covariance of the state'''
    
        Y = np.array(self.D)  # the measurement vector
        H = np.array(self.H)  # the measurement matrix
        R = np.array(self.R)  # the measurement covariance matrix
        
        if len(Y) == 0 : #no information for this date
            self.inov = np.nan            #innovation has no meaning here
            self.K = np.zeros((self.L,1)) #kalman gain has no meaning here
            return (Xf,Pf)                #return forecast
        else : 
            #Data - predictive distribution of Y
            self.inov = self.innovation(Xf,Y)
            #the Covariance or predictive mean of Y
            IS = R + np.dot(H, np.dot(Pf, H.T)) 
            
            #the Kalman Gain matrix
            if np.linalg.det(IS) == 0:
                print('Pf', Pf)
                print('H', H)
                print('determinant is Zero, not invertible')
            
            self.K = np.dot(Pf, np.dot(H.T, np.linalg.inv(IS)))
            
            #the estimated mean state 
            X = Xf + np.dot(self.K,self.inov)  #Y-IM is residual or "innovation vector"
            
            #the estimated covariance state
            P = Pf - np.dot(self.K,np.dot(H,Pf))
           
            if self.verbose :
                self.check_fit(X,P)

            return(X,P)
    
    def innovation(self, Xf, Y):
        '''
        Compute residual or innovation vector
        Innovation for phases is not informative. After a few steps,
        reflects noise of data around model. '''

        #the Mean of predictive distribution of Y
        IM = np.dot(self.H, Xf)
        return(Y-IM)
    
    def check_fit(self, X, P, eps_interf=10):
        '''
        Check quality of fit of phases if verbose activated. 
        Compute residual weighted by its Covariance for analysed state 
        and print warning if pb
        
        * eps_interf : float
             accepted difference between computed and real interferograms.
        '''
                    
        Cres = self.R + np.dot(self.H, np.dot(P, self.H.T))
        res = np.dot(np.linalg.inv(Cres), self.innovation(X, self.D))

        if abs(np.mean(res)) > eps_interf :
            print('WARNING: post-fit residual too big (mean >' +str(eps_interf)+ 'mm)')
            print(res)

    
    def reduce_sizes_m_P(self, k ):
        '''
        Remove phases in m if not used to build interferograms and has converged
        For ulterior long baseline interferograms, phase and associated standard deviation 
        can be recovered but state Covariance terms are lost (too heavy to store)

        * k : integer 
                number of iteration 
        '''

        t_sep = self.t_sep #number of time step to keep in state vector 

        if k >= t_sep :
            L = self.L   #number of parameters
            
            # apply to phases not in current interferograms (t > t_sep)
            sub_P   = np.diag(self.P[L:-(t_sep),L:-(t_sep)])
            sub_m   = self.m[L:-(t_sep)] 
            indx    = self.m_indxs[:-(t_sep)]
           
            #Look at where to insert old phases    
            relativind = indx -(len(self.phases)+self.idx0)
            ind_mod,im = indx[relativind<0],np.where(relativind<0)[0]
            ia = np.where(relativind>=0)[0]
            
            if len(im)>0:
                #phase already stored need to be updated
                self.phases[ind_mod-self.idx0] = sub_m[im]
                self.std[ind_mod-self.idx0] = abs(sub_P[im])**(1/2.)

            #Append phases
            self.phases  = np.append(self.phases,sub_m[ia])
            self.std     = np.append(self.std,abs(sub_P[ia])**(1/2.))  #sqrt of variance        
            
            #Cut state 
            self.P = np.delete(self.P,list(range(L,len(self.m)-t_sep)),0) #row
            self.P = np.delete(self.P,list(range(L,len(self.m)-t_sep)),1) #column
            self.m = np.concatenate((self.m[:L],self.m[-t_sep:]))
            self.m_indxs = self.m_indxs[-t_sep:]

        assert np.shape(self.P)[0]==np.shape(self.P)[1],'ERROR: Pb in reshape, P not square matrix'
        assert np.shape(self.P)[0]==len(self.m),'ERROR: shape of m and P do not match'
        

    def expend_m_P(self,L,n,PL):
        '''
        Open state vector and covariance (m and P) to add building parameters
        
        * L  : integer
                index at which we open and insert new parameters in m and P
        * n  : integer
                number of parameters to add 
        * PL : float
                apriori variance of the new parameters
        '''
        
        #increase size of m 
        self.m = np.concatenate((self.m[:L],np.zeros(n),self.m[L:]))
        
        #extend P
        for i in range(L,L+n):
            row = np.zeros(np.shape(self.P)[0])
            col = np.zeros(np.shape(self.P)[0]+1)
            col[L] = PL
            self.P = np.insert(self.P,i,row,axis=0)
            self.P = np.insert(self.P,i,col,axis=1)
    
    def detect_event(self,k,kmod,m_all):
        ''' 
        IN PROGRESS TESTED ON SYNTHETIC DATA
        Add model parameter for unexpected events not in model
            
        * k : integer
            iteration
        * kmod : integer
            minimum k at which modification can be applied
        '''
        
        # Test for sharp variations in model parameters
        params = [i[:self.L] for i in m_all[-5:-1]]
        condition = np.sum(abs(np.mean(np.diff(params,axis=0)))) > 1/2. *np.sum(self.m[:self.L])
        
        #sig_y  = 15.0 
        #vel_all = [i[1] for i in m_all[-5:-1]]
        #condition = abs(np.mean(np.diff(vel_all))) > sig_y/self.t[k]*1./(len(vel_all)-1)
        
        if k > kmod and condition:
            print('They may be an unexpected event')
            print(np.sum(abs(np.mean(np.diff(params,axis=0)))))
            print(1/2. *np.sum(self.m[:self.L]))
            
            '''
            if k >= kmod_min-1 and self.Lss > 1 :
                #reevaluate length of Lss
                Amps = self.m[self.L+self.Leq:self.L+self.Leq+self.Lss]
                mask = Amps > 1/100.*max(Amps)            #want to keep significant terms only
                rem_i = np.array(range(len(Amps)))[np.invert(mask)] #indexs to remove
                Amps = Amps[mask]                         #to keep
            
                print 'clean phase'
                print self.st
                self.st = self.st[mask]
                self.m = np.concatenate((self.m[:(self.L+self.Leq)],Amps,self.m[(self.L+self.Leq+self.Lss):]))
                print self.st

                self.P = np.delete(self.P,self.L+self.Leq+rem_i,0) #row
                self.P = np.delete(self.P,self.L+self.Leq+rem_i,1) #column

                self.Lss = len(Amps)
            
            if k > kmod_min and conditions :  #compare velocity terms  
                print 'diff between vel', np.diff(vel_all)
                print 'mean change in vel', abs(np.mean(np.diff(vel_all)))
                print 'max crit', sig_y/self.t[k]*1./(len(vel_all)-1)

                dt = int(round(2*self.swidth/6))         #how early can the ss be detected?
                add_times = [self.t[k+dt-5],self.t[k+dt-3]] #times to test
                self.st = np.append(self.st,add_times) 
                
                L = self.L + self.Leq + self.Lss        #where we split m to insert new param
                
                print 'add slowslip', self.t[k]
                
                self.expend_m_P(L,len(add_times),100.**2.) 
                self.m[1] = m_all[-3][1]
                self.Lss += len(add_times) 
                
                kmod_min = k + dt + 2     #time to wait for adjustement of amplitude before chacking for new ss 
                if kmod_min >= len(self.t):
                    kmod_min = len(self.t)'''

    def kf(self, m_err, phi_err, add_err, t_sep=6, plots=True, cm='jet', ax1=0, ax2=0):
        '''
        Run kalman filter combining other functions of class (i.e. MAIN)
            
        * m_err   : array
            systematic error on model (should be 0)
        * phi_err : float 
            systematic error on interferograms (should be 0)
        * t_sep : integer
            maximum time separation between interferograms, fix the minimum
            number of phases that must be kept in the state vector. Constrain 
            the maximum length of the state vector

        * plots : boolean, optional 
            WARNING - activate only if one instance of KF (=one pixel), 
            then subsequent parameters must be specified
                - ax1 : pyplot axis
                    in which plot evolution of parameters
                - ax2 : pyplot axis
                    in which plot evolution of predicted value and model
                - cm  : string or colormap
                    the colormap of reference later discretised 
        '''

        #Prepare storage array       
        kmod = 13        #int from which parameters can be added
        m_all = []       #store state vector at each k in list of lists
        
        self.t_sep = t_sep #caution if large may be slow and heavy

        #Get where to start itterations
        assert len(self.m_indxs)<= len(self.t),'ERROR: more phases computed than dates to work on'
        assert len(self.m_indxs) < len(self.t),'ERROR: from array size, no NEW phase to compute'
       
        self.idx0 = self.m_indxs[0]   #indx of first phase (in this KF update) wrt the initial reference at t0
        k_start   = len(self.m_indxs) #number of phases in m from start (that will be reanalysed/updated)
        k_end     = len(self.t)       #number of dates in time series (final number of phase estimates)
        
        #Initialize
        self.A = self.modelobj.create_A(k_start-1, len(self.m))
        self.create_Q(m_err, phi_err, add_err, len(self.m))
        
        Innov = []
        Gain = []
        #Loop on time
        for k in range(k_start,k_end): 
            
            self.m_indxs = np.append(self.m_indxs, self.idx0+k) #add last phase index 
            self.create_H_R_and_D(k, self.m_indxs-self.idx0)
           
            #Update matrices
            self.create_Q(m_err,phi_err,add_err,len(self.m))
            self.A = self.modelobj.create_A(k-1,len(self.m))

            (mf,Pf) = self.predict(self.m, self.P, self.A, self.Q)
            (self.m,self.P) = self.update(mf, Pf)
            
            #store info
            Gain.append( np.linalg.norm(self.K[:self.L,:], axis=1) ) #Gain of model parameters
            Innov.extend( [np.mean(self.inov)] )
            
            if (k%5==0) or (k_end-1): #every 5th k (to save time)
                #Reduce size of m (part with phases)
                self.reduce_sizes_m_P(k)
            
            #OPTION ONLY TESTED ON SYNTHETICS
            #Add building parameters based on observations
            #m_all.append(self.m.copy())
            #self.detect_event(k,kmod,m_all)
            
            if k < k_end-1:  #not last step
                
                #if number of parameters smaller than in ref
                if self.L < self.Lref : 
                    #Add model element when getting close to relevant date
                    self.mod,n = self.modelobj.expend_model(k,2,verbose=False)
                    self.expend_m_P(self.L-n,n,70.**2.) #L already increased in timefunction

                #Update matrices
                #self.create_Q(m_err,phi_err,add_err,len(self.m))
                #self.A = self.modelobj.create_A(k,len(self.m))
            
            #Plot
            if plots == True : 
                cmap = [cm(1.*i/k_end) for i in range(k_end)] 
                if ax1.ndim == 2:
                    self.plot_params(k,ax1[0,:],cmap)
                    self.plot_gain(k,ax1[1,:],cmap)
                else:
                    self.plot_params(k,ax1,cmap)
                    
                if k > 5 : #do not plot first poorly fitting models
                    self.plot_model(k,ax2,cmap)
                
            ###END loop
        
        self.phases = np.concatenate((self.phases, self.m[self.L:]))
        std_all = np.concatenate((self.std, np.diag(abs(self.P[self.L:,self.L:]))**(1/2.)))
        self.std = std_all
        
        if plots == True : 
            ax2.errorbar(self.t[:],self.phases,yerr=std_all,\
                            fmt='.',c='pink',label='retrieved phases',markersize=10)
            ax2.set_xlabel('Time (days)')
            ax2.set_ylabel('Displacement (mm)')
            self.title_labels(ax1)
        
        self.Gain = np.array(Gain)
        self.Innov = np.array(Innov)

        return
            
    def write_output(self, outstates, outphase, outupdate=None):
        '''
        Save outputs of kalman filter for next run 
        
        * outstates : h5py file 
                Open h5file for state storage 
        * outphase  : h5py file
                Open h5file for phase storage
        * outupdate : h5py file
                Open h5file for gain and innovation (Optional) 
        '''

        i = self.xi
        j = self.yi

        if np.count_nonzero(outstates['indx'][:]) == 0 :  #check if empty 
            #Store pixel independent information
            outstates['indx'][:]  = self.m_indxs
            outstates['tims'][:]  = self.t[self.m_indxs-self.idx0]
            outphase['idx0'][...] = self.idx0
            outphase['tims'][:]   = self.t
        
        #Fill in with new data
        outstates['state'][j,i,:]       = self.m
        outstates['state_cov'][j,i,:,:] = self.P
        outphase['rawts'][j,i,:]        = self.phases
        outphase['rawts_std'][j,i,:]    = self.std
        
        #Store Gain and Innovation
        outupdate["mean_innov"][j,i,:] = self.Innov
        outupdate["param_gain"][j,i,:,:] = self.Gain

        # All done
        return
        
       
    def plot_params(self,k,ax,cmap) :
        ''' 
        Plot each parameter over time with its uncertainty in 
        subplots of size len(ax)''' 
        
        m_plt,P_plt = self.modelobj.comp_phase_shift(self.m[:self.L],P=self.P[:self.L,:self.L])
        P_plt = abs(np.diag(P_plt))**(1/2.)
        
        #plot
        for i in range(self.L):
            ax[i].errorbar(self.t[k],m_plt[i],yerr= P_plt[i],fmt='o',markersize=4.5,\
                        c=cmap[k],elinewidth=0.9,markeredgecolor='k',markeredgewidth=0.3)
            
    
    def plot_gain(self,k,ax,cmap) :
        '''
        Plot gain for each parameter over time'''
        
        g_l = np.linalg.norm(self.K[:self.L,:],axis=1)
        g_plt,tmp = self.modelobj.comp_phase_shift(g_l)
        
        for i in range(self.L):
            ax[i].plot(self.t[k],g_plt[i],'o',c=cmap[k],markersize=3)
            
    def plot_model(self,k,ax,cmap) :
         '''
         Plot resulting model'''
         
         Model= self.draw_model(self.m[:self.L])
         ax.plot(self.t,Model,'-',c=cmap[k],linewidth=0.8)
    
    def title_labels(self,ax1) :
        '''
        Add axes label and titles for subplots from plot_params and plot gain functions'''
         
        if ax1.ndim == 1:
            ax1[0].set_ylabel('parameters')
            #for i in range(len(ax1)):
            #    ax1[i].set_xlabel('time (days)')
        if ax1.ndim == 2:
            ax1[1,0].set_ylabel('gains')
            ax1[0,0].set_ylabel('parameters')
            for i in range(np.shape(ax1)[1]):
                ax1[1,i].set_xlabel('time (days)')
            ax1 = ax1[0] #select first row of plots
        
        #Titles of subplots in fig1
        tltsize = 12
        labels = self.modelobj.get_label(self.L, 'mm', phase=True)
        
        i=0
        for lab in labels:
            ax1[i].set_title(lab,fontsize=tltsize)
            i +=1
    
#EOF
