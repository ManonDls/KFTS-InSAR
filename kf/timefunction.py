# -*- coding: utf-8 -*-
############################################################################@
#     Class implementing the functional 
#        representation of the phase
#
#Date : July 2018
#Author : Manon Dalaison and Romain Jolivet
###########################################################################@

#Enable compatibility with python 2.7 (optional)
from __future__ import print_function
from __future__ import division
from builtins import zip
from builtins import range
from builtins import object

import numpy as np
from scipy.special import factorial
import sys
import os   #using operating system dependent functionality.

class TimeFct(object):

    def __init__(self, time, model, origintime=0.0):
        '''
        Initialization of the class, which build parametrised function of time. 
        This creates an object that can build functional representations \
        and spit out names, etc.
        
            * *time*       : vector of the time (decimal years )
            * *originTime* : origin of the time used here
            * *model*      : contain string of function and associated parameters \
                             with syntax described in table below
            
        ========================================    ========================
         model =                                     description
        ========================================    ========================                              
          ``[('POLY'   ,deg),``                        polynomial of degree ``deg``
          ``('COS'    ,freq),``                        cosine of frequency ``freq``
          ``('SIN'    ,freq),``                        sine of frequency ``freq``
          ``('STEP'   ,t1,t2,...),``                   earthquake at time ``ti``
          ``('HTAN'   ,t1,w1,t2,w2,...),``             slowslip(s) centred on ``ti``
          ``('EXP'    ,t1,w1),``                       starting and characteristic time
          ``('LOG'    ,t1,w1),``                       starting and characteristic time
          ``('BSPLINE',order,t1,w1,t2,w2,...)``        peak(s) of deformation centred on ``ti``
          ``('ISPLINE',order,t1,w1,t2,w2,...)]``       slowslip(s) centred on ``ti``         
        ========================================    ========================
        
        '''

        # Save
        if type(time).__module__ == np.__name__: #test if numpy array
            self.t = time
        else:
            self.t = time[:]
        
        self.ti = origintime
        if origintime > 0 :
            self.t += self.ti  
        self.model = model         #reference should be kept as it is
        self.mod   = model         #model we will work with, may be truncated/expended
        self.check = False         #has the model been checked? 
    
    def check_model(self, verbose=True):
        ''' 
            * *verbose* : print wordy description of the building block of the model (True or False)
        '''     
        
        if verbose==False:
            original_stdout = sys.stdout
            sys.stdout = open(os.devnull, 'w')
        
        k = 0 #count number of param in m 
        for mod in self.model : 
            if mod[0] in ('POLY','POLYNOMIAL'):
                assert len(mod)==2, "Syntax: ['POLYNOMIAL', degree (int)]"
                print('+ Polynomial function of degree', mod[1])
                k += mod[1]+1
                
            elif mod[0] in ('COS','COSINE'):
                assert len(mod)==2, "Syntax: ['COSINE', frequency (float)]"
                print('+ Cosine function of frequency', mod[1])
                k += +1

            elif mod[0] in ('SIN','SINE'):
                assert len(mod)==2, "Syntax: ['SINE', frequency (float)]"
                print('+ Sine function of frequency', mod[1])
                k += +1

            elif mod[0] in ('STEP','EARTHQUAKE','HEAVISIDE'):
                assert len(mod)>=2, "Syntax: ['STEP', time1, time2, ...]"
                print('+ Heaviside function to model earthquakes at times', mod[1:])
                k += len(mod[1:])
                
            elif mod[0] in ('HTAN'):
                assert len(mod)>=3, "Syntax: ['HTAN', time1, width1, time2, width2,...]"
                print('+ Hyperbolic tangent to model slow slip at times', mod[1::2],'and width',mod[2::2])
                k += len(mod[1::2])
                
            elif mod[0] in ('EXP', 'EXPONENTIAL'):
                assert len(mod)==3, "Syntax: ['EXP', start time, characteristic time]"
                print('+ Exponential function (-exp(-x)) starting at time',mod[1],'with characteristic time',mod[2])
                k += 1

            elif mod[0] in ('LOG', 'LOGARITHM'):
                assert len(mod)==3, "Syntax: ['LOG', start time, characteristic time]"
                print('+ Logarithmic function starting at time',mod[1],'with characteristic time',mod[2])
                k += 1
        
            elif mod[0] in ('Bsp', 'BSPLINE'):
                assert len(mod)>=4, "Syntax: ['BSPLINE', order, time1, width1, time2, width2]"
                print('+ B-Spline of degree',mod[1],'centred on times', mod[2::2],'and with width',mod[3::2])
                k += len(mod[2::2])
                
            elif mod[0] in ('Isp', 'ISPLINE'):
                assert len(mod)>=4, "Syntax: ['ISPLINE', order, time1, width1, time2, width2]"
                print('+ Integrated Spline of degree',mod[1],'centred on times', mod[2::2],'and with width',mod[3::2])
                k += len(mod[2::2])
            
            else:
                assert False, 'Functional form unknown: {}'.format(mod)
        
        self.L = k
        print('There need to be',k,'parameters in m')
        print('**** MODEL OK ****') 
        self.check = True
        
        if verbose==False:
            sys.stdout.close()
            sys.stdout = original_stdout
        
        
    def transition_vect(self,t):
        ''' 
        Build linear transition vector that gives the model
            * *t* : time can be one date or an array of dates'''
        
        if (isinstance(t,float) or isinstance(t,int)) :
            A = np.zeros(self.L)      
        elif t.ndim == 1 :
            A = np.zeros((self.L,len(t)))
        else :
            assert False, "Format time not understood"
        
        k = 0
        for mod in self.mod : 
            if mod[0] in ('POLY','POLYNOMIAL'):
                for i in range(mod[1]+1):
                    A[k] = t**(i)
                    k += 1
                    
            elif mod[0] in ('COS','COSINE'):
                A[k] = np.cos(t*mod[1])
                k += 1
                
            elif mod[0] in ('SIN','SINE'):
                A[k] = np.sin(t*mod[1])
                k += 1
                
            elif mod[0] in ('STEP','EARTHQUAKE','HEAVISIDE'):
                if len(mod) == 2 :  #one earthquake  
                    A[k] = _step(t,[mod[1]])
                    k += 1
                else: 
                    nb_eq = len(mod)-1
                    A[k:k+nb_eq] = _step(t,mod[1:])
                    k += nb_eq
                    
            elif mod[0] in ('HTAN'):
                if len(mod) == 3 : #one slowslip
                    A[k] = _htan(t,[mod[1]],[mod[2]])
                    k += 1
                else: 
                    nb_ss = (len(mod)-1)//2 #floor division
                    A[k:k+nb_ss] = _htan(t,mod[1::2],mod[2::2])
                    k += nb_ss
                    
            elif mod[0] in ('EXP', 'EXPONENTIAL'):
                A[k] = (1 - np.exp(-(t-mod[1])/mod[2]))*(t>=mod[1])
                k += 1

            elif mod[0] in ('LOG', 'LOGARITHM'):
                A[k] = np.log(1.0+ ((t-mod[1])/mod[2])*(t>=mod[1]))
                k += 1
        
            elif mod[0] in ('Bsp', 'BSPLINE'):
                nb_bs = (len(mod)-2)//2
                spline = _bspline(self.t,mod[1],mod[2::2],mod[3::2])
                
                if (isinstance(t,float) or isinstance(t,int)):
                    if spline.ndim ==1:
                        spline = spline[np.where(self.t==t)]
                    if spline.ndim ==2:
                        #remove dim with squeeze to have 1D like A
                        spline = np.squeeze(spline[:,np.where(self.t==t)])
                        
                A[k:k+nb_bs] = spline
                k += nb_bs
    
            elif mod[0] in ('Isp', 'ISPLINE'):
                nb_is = (len(mod)-2)//2
                spline = _ispline(self.t,mod[1],mod[2::2],mod[3::2])
                
                if (isinstance(t,float) or isinstance(t,int)):
                    if spline.ndim ==1:
                        spline = spline[np.where(self.t==t)]
                    if spline.ndim ==2:
                        #remove dim with squeeze to have 1D like A
                        spline = np.squeeze(spline[:,np.where(self.t==t)])
                    
                A[k:k+nb_is] = spline
                k += nb_is
                
            else:
                assert False, 'Functional form unknown: {}'.format(mod)
        
        return A 
    
    def find_coeff_lsq(self,evolv,err):
        '''
        Basic linear least-squares which finds of defined model
            * evolv : time series of phase change (shape of time)
            * err : standard deviation of shage  (shape of time)
        '''
        A = self.transition_vect(self.t)
        y = evolv
        
        ### Weight matrix for covariance in Data
        Cd_inv = np.eye(len(self.t))*(err)**(-1)
        
        ### Least square regression
        Cm = np.linalg.inv(np.dot(np.dot(A.T,Cd_inv),A)) #posterior model covariance
        
        if evolv.ndim ==2:
            ATb  = [np.dot(np.dot(A.T, Cd_inv), y[i,:]) for i in range(np.shape(evolv)[0])]
            m    =  np.array([np.dot(Cm, ATb[i]) for i in range(np.shape(evolv)[0])])
        else :
            ATb  = np.dot(np.dot(A.T, Cd_inv), y)
            m    = np.dot(Cm, ATb)
        
        merr = np.sqrt(np.diag(Cm))
        
        return m, merr
        
    def draw_model(self,coeff):
        ''' 
        Gives f(t) as defined in model
            * *coeff* : contain multiplying coefficients of each function \
                        in the same order as in model \
                        (correspond to first elements of state vector m)'''     
        
        assert np.shape(coeff)[-1]==self.L, "number of parameters do no match functional model"

        A = self.transition_vect(self.t)
        
        '''#tmp
        if coeff.ndim==3:
            fault = (3.,1.,60.)
            for x in range(0,coeff.shape[1]):
                for y in range(0,coeff.shape[0]): 
                    xy_sign = fault_vel_2D(x,y, fault)
                    coeff[y,x,-1] = xy_sign*coeff[y,x,-1]
        #tmp '''
        
        f = np.dot(coeff,A)
    
        return f 
        
    def create_A(self,k,M):
        ''' 
        Create A, the transition n × n matrix which converts m into mf
        Matrix used during prediction in the Kalman Filter
            * *k* : the timestep
            * *M* : the state vector length '''
        
        A_top = np.eye(M)         #keep phases as in state vector (NxN)
        A_row = np.zeros(M)       #compute next phase from model (N)
        
        A_row[:self.L]= self.transition_vect(self.t[k+1])
        A = np.vstack((A_top,[A_row]))
        
        return A
    
    def shift_t0(self,t0,coeff):
        ''' 
            * t0 : shift in t0 (t0_new -t0_old)
            * coeff  : parameter (assuming: offset, slope, sin, cos along last axis)'''
        
        m = coeff.copy()
        m_new = np.zeros(np.shape(m))
        
        #avoid useless computations
        if t0==0.0:
            return m
            
        #keep track of cos sine in model
        cos = False
        sin = False
        
        #count num of parameters
        k=0 
        #count num of functional units (mod)
        j=0 
        for mod in self.mod : 
            if mod[0] in ('POLY','POLYNOMIAL'):
                m_new[...,0] = m[...,0]
                for i in range(mod[1]+1):
                    m_new[...,0] += -m[...,k]*t0**(i)
                    m_new[...,k] = m[...,k]
                    k += 1
                    
            elif mod[0] in ('COS','COSINE'):
                cos = True
                if sin == False:
                    ktrig = k 
                    k += 1
                else : 
                    print("assuming same freq for SIN and COS")
                    m_new[...,ktrig] = m[...,ktrig]*np.cos(mod[1]*t0) + m[...,k]*np.sin(mod[1]*t0)
                    m_new[...,k]     = m[...,k]*np.cos(mod[1]*t0)     - m[...,ktrig]*np.sin(mod[1]*t0)
                    k += 1
                
            elif mod[0] in ('SIN','SINE'):
                sin = True
                if cos == False : 
                    ktrig = k 
                    k += 1
                else : 
                    print("assuming same freq for SIN and COS")
                    m_new[...,k]     = m[...,k]*np.cos(mod[1]*t0)     + m[...,ktrig]*np.sin(mod[1]*t0)
                    m_new[...,ktrig] = m[...,ktrig]*np.cos(mod[1]*t0) - m[...,k]*np.sin(mod[1]*t0)
                    k += 1
                
            elif mod[0] in ('STEP','EARTHQUAKE','HEAVISIDE'):
                #keep same amplitude but shift time in model 
                nb_eq  = len(mod)-1
                newmod = list(mod)
                
                for i in range(nb_eq) : 
                    newmod[i+1] += t0
                    m_new[...,k+i] = m[...,k+i]
                
                self.mod[j] = tuple(newmod)
                print("New model",self.mod)
                k += nb_eq
                    
            elif mod[0] in ('HTAN'):
                print("WARNING: haven't been tested")
                nb_ss =int((len(mod)-1)/2)
                for i in range(nb_ss):
                    mod[1+i*2] += t0
                k += nb_ss
                
            else:
                assert False, 'Functional form unknown: {}'.format(mod)
            j += 1
            
        if sin+cos == 1 : #one False, one True
            assert False, 'need COS and SINE together in model to shift t axis'
        
        return m_new

    def cut_model(self,N):
        ''' 
        Function that removes some of the component of the functional model 
        based on the length of parameters (N). This allows to include informations 
        when needed and not before.
        '''
        k  = 0 #count number of parameters
        kk = 0 #count number of model elements
        for mod in self.model:
            
                kkk = 0 #count number of element inside model element
                if mod[0] in ('POLY','POLYNOMIAL'):
                    for i in range(mod[1]+1):
                        kkk = i+2 
                        k   += 1
                        if k==N:
                            break
                        
                elif mod[0] in ('COS','COSINE'):
                    kkk = 2
                    k += 1
                    
                elif mod[0] in ('SIN','SINE'):
                    kkk = 2
                    k += 1
                    
                elif mod[0] in ('STEP','EARTHQUAKE','HEAVISIDE'):
                    for i in range(1,len(mod)):
                        kkk = i+1 
                        k   += 1 
                        if k==N:
                            break
                        
                elif mod[0] in ('HTAN'):
                    for i in range(1,len(mod),2):
                        kkk = i+2 
                        k  += 1
                        if k==N:
                            break
                        
                elif mod[0] in ('EXP', 'EXPONENTIAL'):
                    kkk = 3
                    k += 1
    
                elif mod[0] in ('LOG', 'LOGARITHM'):
                    kkk = 3
                    k += 1
            
                elif mod[0] in ('Bsp', 'BSPLINE'):
                    for i in range(2,len(mod),2):
                        kkk = i+2 
                        k  += 1
                        if k==N:
                            break
        
                elif mod[0] in ('Isp', 'ISPLINE'):
                    for i in range(2,len(mod),2):
                        kkk = i+2
                        k  += 1
                        if k==N:
                            break 
                
                else:
                    assert False, 'Functional form unknown: {}'.format(mod)
                
                kk += 1
                if k==N:
                    break
        
        newmodel = self.model[:kk]
        newmodel[-1] = newmodel[-1][:kkk]
        self.L = N
        self.mod = newmodel
        
        return self.mod
        
    def expend_model(self,k,dt, verbose=True):
        ''' Add events in reference model not in self.mod, 
        if we are getting temporarily close to the occurence of the event 
            * *dt* : anticipation time
        WORKS ONLY IF CHRONOLOGICAL ORDER IN MODEL'''
        
        t = self.t[k+1]
        mod = self.mod
        
        n = 0 #count number of param to add
        kk = 0
        for ref in self.model:
            Lmod = 0 
            if kk < len(mod):
                Lmod = len(mod[kk])
                if len(ref)==len(mod[kk]) and ref[0]==mod[kk][0]:
                    kk += 1
                    continue
            
            if ref[0] in ('STEP','EARTHQUAKE','HEAVISIDE'):
                times = [i for i in ref[1:] if i <= (t+dt)]
                indx = len(times)
                if indx > 0 :
                    if Lmod ==0 : #no earthquake before
                        n +=indx
                        mod.insert(kk,ref[:indx+1])
                    else : 
                        n += indx+1 -len(mod[kk])
                        newmod = [x for x in mod if x!=mod[kk]]
                        newmod.insert(kk,ref[:indx+1])
                        mod = newmod
            
            elif ref[0] in ('HTAN'):
                times = [i for i,j in zip(ref[1::2],ref[2::2]) if i <= (t+dt+j)]
                indx = len(times)
                if indx > 0 :
                    if Lmod ==0 :
                        n +=indx
                        mod.insert(kk,ref[:indx*2+1])
                    else : 
                        n +=indx+1 -len(mod[kk])
                        newmod = [x for x in mod if x!=mod[kk]]
                        newmod.insert(kk,ref[:indx*2+1])
                        mod = newmod
                        
            elif ref[0] in ('Bsp', 'BSPLINE','Isp', 'ISPLINE'):
                times = [i for i,j in zip(ref[2::2],ref[3::2]) if i <= (t+dt+j)]
                indx = len(times)
                if indx > 0 :
                    if Lmod ==0 :
                        n +=indx
                        mod.insert(kk,ref[:indx*2+2])
                    else : 
                        n +=indx+1 -len(mod[kk])
                        newmod = [x for x in mod if x!=mod[kk]]
                        newmod.insert(kk,ref[:indx*2+2])
                        mod = newmod
                        
            elif ref[0] in ('EXP', 'EXPONENTIAL','LOG', 'LOGARITHM'):
                if ref[1] <= (t+dt+ref[2]):
                    mod.insert(kk,ref)
                    n +=1
                
            kk += 1
        
        if n > 0 and verbose :
            print("increased number of param by",n,"at step",k)
        self.L += n
        self.mod = mod
        
        return self.mod,n 
    
    def identify_outdated(self, dtmax):
        '''
        Function optimizing model as a function of starting time of time series, 
        localize modification that will be berformed later on every state vectors/cov
            * *dtmax* : time after event allowed to optimize localized function in time
                        (apply only on 'STEP','EARTHQUAKE','HEAVISIDE' and cst of 'POLY')
        '''
                
        indexdel = []    # indexes of parameters to remove
        modeldel = []    # model element and index inside 
        Cstindex = None  # index of cst term if time to fix it 
          
        if self.t[0] < dtmax : 
            print("Starting time is {}".format(self.t[0]))
            print("Existing model agrees with the maximum delta time set to {}".format(dtmax))
        
        else :
            k  = 0 #count number of parameters
            kk = 0 #count number of model elements
            for mod in self.mod:
                if mod[0] in ('POLY','POLYNOMIAL'):
                    if mod[1]>=0:
                       print("Fix model at origin (i.e. polynomial term of order zero).")
                       print("    Consider it has already converged to a reliable value because {} > starting time ({})".format(dtmax,self.t[0]))
                       Cstindex = k
                       k += mod[1]+1
                
                elif mod[0] in ('STEP','EARTHQUAKE','HEAVISIDE'):
                    kkk = [] #local count
                    for i in range(1, len(mod)) : 
                        if self.t[0] > mod[i] + dtmax :
                            print("Remove event centered on {}".format(mod[1+i]))
                            indexdel.append(k)
                            kkk.append(i)
                        modeldel.append((kk,kkk))
                        k+=1
                
                elif mod[0] in ('COS','COSINE','SIN','SINE','EXP', 'EXPONENTIAL','LOG', 'LOGARITHM'):
                    k += 1 #all functions with a single parameter 
                        
                elif mod[0] in ('HTAN'):
                    for i in range(1,len(mod),2):
                        k  += 1
            
                elif mod[0] in ('Bsp', 'BSPLINE','Isp', 'ISPLINE'):
                    for i in range(2,len(mod),2):
                        k  += 1
                            
                kk += 1
        
        self.Cstindex = Cstindex
        self.indexdel = indexdel
        
        newmod = self.mod.copy()
        for el in modeldel: 
            kk,kkk = el
            #remove specified indexes
            newtiming = [t for i,t in enumerate(self.mod[kk]) if i not in kkk]
            newmod[kk] = tuple(newtiming)
            
        #save new model
        print("Define new model {}".format(newmod))
        self.mod = newmod
        
    
    def remove_oldstuff(self, m, P):
        '''
        Require the outputs of identify_outdated() 
            * *m, P* : specify local state vector and covariance
        '''
        
        if self.Cstindex is not None : 
            dY = 0.0  # shift of origin 
            for k in self.indexdel: 
                dY += m[k] 
                m = np.concatenate((m[:k],m[k+1:]))
                P = np.delete(P,k,0) #row
                P = np.delete(P,k,1) #column
            
             
            # Shift and fix constant term    
            m[self.Cstindex] += dY
            # Set variance and covariance to zero
            P[self.Cstindex,:] = 0 
            P[:,self.Cstindex] = 0 
            
        else :
            print("No outdated parameters identified. Model stay unchanged.")
        
        return m, P
        
        
    def comp_phase_shift(self, m, P=None):
        '''
        Compute oscillation amplitude and phase shift with its variances 
        from amplitudes of cosine and sine terms with their variances. 
            * *m* : state vector containing at least all the model parameters (length > self.L)
            * *P* : if error is known =P the state covariance (default None)'''

        # Get index of sin and cos
        indx_cos = None
        indx_sin = None
        freqs = []
        
        k = 0
        for mod in self.mod : 
            if mod[0] in ('POLY','POLYNOMIAL'):
                k += mod[1]+1
                    
            elif mod[0] in ('COS','COSINE'):
                freqs.append(mod[1])
                indx_cos = k
                k += 1
                
            elif mod[0] in ('SIN','SINE'):
                freqs.append(mod[1])
                indx_sin = k
                k += 1
    
            elif mod[0] in ('STEP','EARTHQUAKE','HEAVISIDE'):
                k += len(mod)-1
        
            elif mod[0] in ('HTAN'):
                k += (len(mod)-1)//2
                    
            elif mod[0] in ('EXP', 'EXPONENTIAL','LOG', 'LOGARITHM'):
                k += 1
        
            elif mod[0] in ('Bsp', 'BSPLINE','Isp', 'ISPLINE'):
                k += (len(mod)-2)//2
                
            else:
                assert False, 'Functional form unknown: {}'.format(mod)
        
        if (indx_cos and indx_sin) is None:
            print("no 'SIN' and 'COS' inside model, phase shift not computed")
            return m,P
        
        if len(freqs)==2 and freqs[0]!=freqs[1]:
            print("'SIN' and 'COS' don't have same frequencies")
            return m,P
        
        # Copy m
        m_out = np.copy(m)
        
        #select indx in last dimension of ndim array using python Ellipsis
        b  = m[...,indx_sin]
        a  = m[...,indx_cos]
        
        sine_amp  = np.sqrt(a**2 + b**2) 
        phase_shift = np.arctan(a/b)
        m_out[...,indx_sin] = sine_amp
        m_out[...,indx_cos] = phase_shift 
        
        #error propagation if known (P !=None)
        if type(P) == type(None) :
            P_out = None    
        else : 
            #careful deal with variance directly not uncertainty
            P_out = np.copy(P)
            sb = P[...,indx_sin,indx_sin]
            sa = P[...,indx_cos,indx_cos]
            phase_shift_err = (a**2*sb +b**2*sa)/((a**2+b**2)**2)
            sine_amp_err    = (a**2*sa +b**2*sb)/(a**2+b**2)
            P_out[...,indx_sin,indx_sin] = sine_amp_err
            P_out[...,indx_cos,indx_cos] = phase_shift_err
         
        return m_out, P_out
        
        
    def get_label(self, L, unit, phase=False):
        ''' 
        Get an array of labels for each model parameters (name and units) 
        which can be used for plotting
            * *L*     : number of parameters
            * *unit*  : string of length unit
            * *phase* : True if sine and cosine amplitudes converted into \
                        sine amplitude and phase shift '''
         
        label = [None]*L
                
        k = 0
        for mod in self.mod : 
            if mod[0] in ('POLY','POLYNOMIAL'):
                label[k] = 'Offset\n (%s)'%unit
                k += 1
                if mod[1] >= 1:
                    label[k] = 'Velocity\n $(%s/day)$'%unit
                    k += 1
                if mod[1] >= 2:
                    for i in range(2,mod[1]+1):
                        label[k] = 'Polynomial coef.\n $(mm/day^%d)$'%i
                        k += 1
                    
            elif mod[0] in ('COS','COSINE'):
                if phase== True:
                    label[k] = 'Phase shift\n '
                    k += 1
                else : 
                    label[k] = 'Amplitude of cosine\n (%s)'%unit
                    k += 1
                
            elif mod[0] in ('SIN','SINE'):
                label[k] = 'Amplitude of sine\n (%s)'%unit
                k += 1
                
            elif mod[0] in ('STEP','EARTHQUAKE','HEAVISIDE'):
                for tmp in range(len(mod)-1):  
                    label[k] = 'Amplitude of quake\n (%s)'%unit
                    k += 1
        
            elif mod[0] in ('HTAN','Isp', 'ISPLINE'):
                for tmp in range((len(mod)-1)//2): 
                    label[k] = 'Amplitude of slow slip\n (%s)'%unit
                    k += 1
                    
            elif mod[0] in ('EXP', 'EXPONENTIAL'):
                label[k] = 'Amplitude of exponential\n (%s)'%unit
                k += 1

            elif mod[0] in ('LOG', 'LOGARITHM'):
                label[k] = 'Amplitude of logarithm\n (%s)'%unit
                k += 1
        
            elif mod[0] in ('Bsp', 'BSPLINE'):
                for tmp in range((len(mod)-2)//2): 
                    label[k] = 'Amplitude of bspline\n (%s)'%unit
                    k += 1
            else:
                assert False, 'Functional form unknown: {}'.format(mod)
            
        return label
        
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
# PRIVATE FUNCTIONS out of class 
# (no common parameters as time may be full array or one value)
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------  
def _step(t,qtime):
    '''
    Generate step function(s) of x centred on qtime(s)
    y= 0 before event(x<quake_time) otherwise y=+1 (y of len t)
        * x     : time array (array/list)
        * qtime : list of times at which earthquakes occur (array/list)''' 
    
    if len(qtime)==0 :
        y = np.zeros(len(t))
        print('no earthquake time, empty list')    
    elif len(qtime)==1 : 
        #heaviside allows for y=1 at t=quake_time
        y = np.heaviside(t-qtime,1.)   
    else :               #output is 2D (len(sst) x len(x))
        y = np.array([np.heaviside(t-qt,1.) for qt in qtime])
    
    return y

def _htan(t,sst,ss_dt):
    '''
    Model slow slip event as hyperbolic tangent (smoothed step function) 
    drawback : non-zero value far from event
        * x     : single time or array of times
        * sst   : centre of the function (float or 1D array)
        * ss_dt : width or extend of the smoothing (float)'''

    if len(sst)==0 :
        s = np.zeros(len(t))
        print('no slowslip time, empty list') 
    elif len(sst)==1 :  
        s = 0.5 +0.5*np.tanh((t-sst)/ss_dt)
    else :
        s = np.array([0.5 +0.5*np.tanh((t-time)/dt) for time,dt in zip(sst,ss_dt)])
        
    return s   

def _bspline(t, order, center, dtk, normalise=True):
    '''
    Uniform b-splines for the time vector
    Can deal with several splines at once (center and dtk of same size)
        * t         : must be an array for normalisation
        * order     : Order (integer)
        * center    : Center of the spline (array/list)
        * dtk       : Half-width (array/list)'''
    
    # Time
    if len(center)==1 : #one event
        x = (t-center)/dtk + order + 1
    else :     #several events
        x = np.array([(t-ctr)/dt +order +1 for ctr,dt in zip(center,dtk)])

    # Check order
    if not np.mod(order,2):  #if order pair remove 0.5 to time
        x -= 0.5        
    
    # Iterate over order
    b = np.zeros(np.shape(x))
    for k in range(order+2):
        m = x-k-(order+1)/2
        up = m**(order)
        b += ((-1)**k)*nCk(order+1,k)*up*(m>=0)
    
    if normalise==False:
        return b
        
    #normalise 
    if len(center)==1:
        b_norm = b/np.nanmax(b)
    else : 
        b_norm = np.array([bb/np.nanmax(bb) for bb in b])
    
    # All done
    return b_norm 


def _ispline(t, order, center, dtk, normalise=True):
    '''
    Uniform integrated b-splines
    Can deal with several splines at once (center and dtk of same size)
        * t         : must be an array for normalisation
        * order         : Order (integer)
        * center        : Center of the i-spline (array/list)
        * dtk           : Half-width  (array/list)'''

    # Time
    if len(center)==1 : #one event
        x = (t-center)/dtk + order + 1
    else :     #severalevents
        x = np.array([(t-ctr)/dt +order +1 for ctr,dt in zip(center,dtk)])

    # Check order
    if not np.mod(order,2):   #if order pair remove 0.5 to time
        x -= 0.5
    
    # Iterate
    b = np.zeros(np.shape(x))
    for k in range(order+2):
        m = x-k-(order+1)/2
        up = m**(order+1)
        b += ((-1)**k)*nCk(order+1,k)*up*(m>=0)
    
    if normalise==False:
        return b
        
    #normalise 
    if len(center)==1 :
        b_norm = b/np.nanmax(b) 
    else : 
        b_norm = np.array([bb/np.nanmax(bb) for bb in b])
        
    # All done
    return b_norm

# ----------------------------------------------------------------------

# Combinatorial
def nCk(n,k):
    '''Combinatorial function.'''
    c = factorial(n)/(factorial(n-k)*factorial(k)*1.0)
    return c


def fault_vel_2D(x, y, xxx_todo_changeme):
    '''compute the sign of velocity on each side of 
    a fault trace with equation ax+by-c=0 '''
    (a, b, c) = xxx_todo_changeme
    if (isinstance(x,float) or isinstance(x,int)) :
        vel = 1.0
        if a*x+b*y <= c:
            vel = -1.0
    else :     #(x,y) arrays from meshgrid
        vel = np.ones(np.shape(x))
        vel[a*x+b*y <= c] = -1.0
    return vel  
    
#EOF
