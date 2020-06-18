# -*- coding: utf-8 -*-
###########################################################################
# Script to convert earthquake properties into informations for running the 
# kalman filter and store into text file containing: 
# * Xeq,Yeq   : index/pixel position of earthquake in interferograms 
# * teq       : time (dec year) of earthquake wrt initial date of time series
# * Rinf      : radius (km) of likely influence of the earthquake
# * inclon,lat : local spacing between pixel in km (aproximate)
#
# Manon Dalaison 
# April 2019
#
###########################################################################

# Import global stuff
import numpy as np
import h5py
import datetime as dt
import matplotlib.pyplot as plt
import operator

#############################  CONFIGURATION  #############################
# LOCATIONS
loc = '/Users/username/Yourfolder/'

infile  = loc + 'Phases.h5' 

eqinfo  = '../../GCMT/CMT_KF.txt'
eqloc   = '../../GCMT/CMT_KF_ISC.txt' #relocalised events when available
lonfile = loc + 'lon.flt'
latfile = loc + 'lat.flt'

RELOC = False   # relocate using second catalogue (eqloc)?
EXTEND = False  # get newer earthquakes
SELECT = False  # select event with greater chance to appear in data

#Outputs
PLOT = True
outfig  = loc+'test/earthquakes-GCMT.png'
outfile = loc + 'test/EQ_list.txt'
outplotinfo = loc + 'test/EQ_info.txt'

# CONSTANTS
mu  = 30 *10**9                     # rigidity Pa : kg m−1 s−2

###########################################################################
# Import date and geometry of reference
print('Load data')

fin    = h5py.File(infile,'r')

if 'PROC-STACK' in infile:
    ny, nx = np.shape(fin['cmask'])
elif 'Phases' in infile:
    ny, nx = np.shape(fin['rawts'])[:-1]
else : 
    assert False, "Do not recognise file name, you have to addapt the code"
    
#Ordinal date to decimal year
init      = dt.datetime.fromordinal(fin['dates'][0].astype(int))
yr_start  = dt.date(init.year,1,1).toordinal()
yr_len    = dt.date(init.year+1,1,1).toordinal() - yr_start
day_inyr  = fin['dates'][0] - yr_start
t0        = init.year + day_inyr/yr_len

last = dt.datetime.fromordinal(fin['dates'][-1].astype(int))

#Real spatial grid 
lontmp = np.fromfile(lonfile,dtype=np.float32)
lattmp = np.fromfile(latfile,dtype=np.float32)
lon    = np.reshape(lontmp,(ny,nx))
lat    = np.reshape(lattmp,(ny,nx))

if PLOT :
    fin = h5py.File(infile,'r')
    if 'PROC-STACK' in infile:
        toplot = fin['figram'][-1,:,:]
    elif 'Phases' in infile:
        toplot = fin['rawts'][:,:,-1]
        
#-------------------------------------------------------------------------    
#Get earthquake properties
elon,elat,Deq,Mw = np.loadtxt(eqinfo, unpack=True, usecols=list(range(4)), dtype =np.float)
Yr,Mo,Dy,Hr,Min  = np.loadtxt(eqinfo, unpack=True, usecols=list(range(4,9)), dtype ='i4')

if isinstance(Yr,np.int32):
    #Special case of one earthquake
    Leq = 1
    Yr,Mo,Dy,Hr,Min  = [Yr],[Mo],[Dy],[Hr],[Min]
    elon,elat,Deq,Mw = [elon], [elat], [Deq], [Mw]
else:
    Leq = len(Yr)

#tuples of dates
edate  = [(yr,mo,dy,hr,mi) for yr,mo,dy,hr,mi in zip(Yr,Mo,Dy,Hr,Min)]

if RELOC:
    # load second set of earthquake properties
    elon2,elat2,Deq2,Mw2 = np.loadtxt(eqloc, unpack=True, usecols=list(range(4)), dtype =np.float)
    Yr2,Mo2,Dy2,Hr2,Min2  = np.loadtxt(eqloc, unpack=True, usecols=list(range(4,9)), dtype ='i4')
    edate2 = [(yr,mo,dy,hr,mi) for yr,mo,dy,hr,mi in zip(Yr2,Mo2,Dy2,Hr2,Min2)]

    
#Get better locations
if RELOC == True: 
    
    iin   = [i for i,t in enumerate(edate2) if t in edate]
    iout  = [i for i,t in enumerate(edate) if t in edate2]
    
    #replace 
    print("WARNING: Relocate",len(iin),"earthquakes")
    print("   Difference along lon :",elon[iout] - elon2[iin])
    print("   Difference along lat :",elat[iout] - elat2[iin])
    print("   Difference in Mw    :",Mw[iout] - Mw2[iin])
    print("   Difference in Depth :",Deq[iout] - Deq2[iin])
    elon[iout] = elon2[iin]
    elat[iout] = elat2[iin]
    Deq[iout]  = Deq2[iin]
    Mw[iout]  = Mw2[iin]

#Add events 
if EXTEND == True:
    #Add big events missing
    #print('Get missing big events from second catalogue')
    #iplus   = [i for i,t in enumerate(edate2) if ((Mw2[i] > 3.7) and (t not in edate)) and (t<(last.year,last.month-1,last.day,0,0))]
    
    #Extend in time to recent period not necessarily in GCMT catalogue
    print('Last date in first catalogue:', edate[-1])
    iplus = [i for i,t in enumerate(edate2) if ((t>edate[-3]) and (t not in edate)) and (t<(last.year,last.month-1,last.day,0,0))]
    #iplus = np.concatenate((ipluss,iplusss))
    
    print('Add',len(iplus),'events to list')
    elon = np.concatenate((elon,elon2[iplus]))
    elat = np.concatenate((elat,elat2[iplus]))
    Deq  = np.concatenate((Deq,Deq2[iplus]))
    Mw   = np.concatenate((Mw,Mw2[iplus]))
    Yr,Mo,Dy,Hr,Min = np.concatenate(([Yr,Mo,Dy,Hr,Min],[Yr2[iplus],Mo2[iplus],Dy2[iplus],Hr2[iplus],Min2[iplus]]),axis=1)
    Leq = len(Yr)

#-------------------------------------------------------------------------        
#Convert time
print('Convert time of earthquake in time series time')
teq = np.zeros(Leq)
for i in range(Leq):
    yr_start = dt.date(Yr[i],1,1).toordinal() 
    yr_len   = dt.date(Yr[i]+1,1,1).toordinal() - yr_start
    day_inyr = dt.datetime(Yr[i],Mo[i],Dy[i],Hr[i],Min[i]).toordinal() - yr_start
    teq[i]   = Yr[i] + day_inyr/yr_len -t0  

#Select corresponding pixel
print('Start looking for nearest neighbour')
Xeq,Yeq = np.zeros((2, Leq), dtype=int)
Dist    = np.zeros(Leq)

for i in range(Leq):
    # distance to station point
    dist = np.sqrt((lon-elon[i])**2 + (lat-elat[i])**2)
    # get index of minimum(s)
    ind = np.unravel_index(dist.argmin(), lon.shape)
    Yeq[i],Xeq[i] = ind
    Dist[i] = np.sqrt(((lon[ind] - elon[i])*111)**2 + ((lat[ind] - elat[i])*111)**2)


#Select relevent events 
maskeq = np.ones(Leq)
if SELECT:
    maskeq[Deq>=20] = 0                 #depth inferior to 30km
    maskeq[teq<0.77] = 0                 #don't occur in first 6 monthes
    maskeq[Dist>0.2] = 0                #nearest pixel is closer than 0.5km
    maskeq[Mw<=3.7] = 0                   #Mw larger than 4.5
    maskeq[(Xeq>20)^(Yeq>20)] = 0       #not at the edge of image
    maskeq[(Xeq<nx-500)^(Yeq<ny-10)] = 0

    teq = teq[maskeq>0]
    Xeq = Xeq[maskeq>0]
    Yeq = Yeq[maskeq>0]

#Radius of influence for Spatial P0
'''Rinf = (1/(mu*Deq[maskeq>0]*10**3)*10**(1.5*Mw[maskeq>0]+9.1))**1/2.*10**(-3) #in km  
Rinf = Rinf*22.   #amplify to capture whole signal 
Rinf = Rinf+10.   #to account for loc uncertainty (measured in defo map)
'''
print("WARNING: fix the influence radius to 9 km")
Rinf = np.full(teq.shape,9.0) # fix to 9 km 

#Distance between pixels (averaged over 100 pixels around center) 
inclon = abs(lon[Yeq,Xeq]-lon[Yeq,Xeq+20])*111./20. #km 
inclat = abs(lat[Yeq,Xeq]-lat[Yeq+20,Xeq])*111./20.

#Sort according to date
L = sorted(zip(Xeq,Yeq,teq,Rinf,inclon,inclat), key=operator.itemgetter(2))
Xeq,Yeq,teq,Rinf,inclon,inclat = zip(*L)
Rinf = np.array(Rinf)

#Build Output
DataOut = np.column_stack((Xeq,Yeq,teq,Rinf,inclon,inclat))

#Verify earthquakes are distinguishable
dte  = np.diff(teq)
dloc = np.sqrt((np.diff(Yeq)*np.mean(inclat))**2 +(np.diff(Xeq)*np.mean(inclon))**2)
distinc = (dte > 0.08) + (dloc > (Rinf[:-1]+Rinf[1:])/2.)

for i in range(len(dte)):
    if distinc[i] == False:
        print('WARNING: Identify closeby earthquakes')
        print('Occured',round(teq[i],3),round(teq[i+1],3),'at locations',(Xeq[i],Yeq[i]),(Xeq[i+1],Yeq[i+1]))
        #smallone = np.argmin([Rinf[i],Rinf[i+1]])
        #DataOut = np.vstack([ DataOut[:i+smallone,:],DataOut[i+smallone+1:,:] ])
        
DataOut = DataOut[np.concatenate(([True],distinc)),:]        

#-------------------------------------------------------------------------   
#Save in File
np.savetxt(outfile,DataOut,fmt='%i %i %1.5f %1.3f %1.5f %1.5f')
print('Saved file :',outfile)

#Save corresponding information for GMT plot and record    
DataPlot = np.column_stack((elon[maskeq>0], elat[maskeq>0], Yr[maskeq>0],\
                            Mo[maskeq>0],Dy[maskeq>0],Hr[maskeq>0],Min[maskeq>0],\
                            Deq[maskeq>0], Mw[maskeq>0]  ))

np.savetxt(outplotinfo,DataPlot,fmt='%1.5f %1.5f %i %i %i %i %i %1.2f %1.1f')


#-------------------------------------------------------------------------   
#Plot to visualise earthquake
if PLOT:
    Xeq,Yeq,teq,Rinf,inclon,inclat = DataOut.T
    plt.figure(figsize=(5,11))
    plt.pcolormesh(toplot,cmap='jet',vmin=-90,vmax=60) #fin['figram'][-10,:,:])
    plt.scatter(Xeq,Yeq,2**Mw[maskeq>0],'r')
    plt.ylim(np.shape(toplot)[0],0)
    for i in range(len(teq)):
        plt.text(Xeq[i],Yeq[i],str(i+1)+' t= '+str(round(teq[i],2)),color='r')
    
    plt.savefig(outfig,dpi=230)
    print('Saved figure :', outfig)
    
    
    
