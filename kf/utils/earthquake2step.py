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
from matplotlib import path
from scipy.spatial import distance
import operator

#############################  CONFIGURATION  #############################
# LOCATIONS
loc = 'testdata/synthfault/'
infile  = loc + 'Synth_interf.h5'  

eqinfo  = loc + 'eq_properties.txt'
eqloc   = ''                  #relocalised events when available
lonfile = loc + 'lon.flt'
latfile = loc + 'lat.flt'

RELOC = False   # relocate using second catalogue (eqloc)?
EXTEND = False  # get newer earthquakes and big (Mw>4) missing
SELECT = True  # select event with greater chance to appear in data

#Outputs
PLOT = True
outfig  = loc+'earthquakes-GCMT.png'
outfile = loc +'EQ_list.txt'
outplotinfo = loc +'EQ_info.txt'

# CONSTANTS
mu  = 30 *10**9                     # rigidity Pa : kg m−1 s−2

###########################################################################
# Import date and geometry of reference
print('Load data')

fin    = h5py.File(infile,'r')

if ('STACK' in infile) or ('interf' in infile):
    try : 
        ny, nx = np.shape(fin['igram'])[1:]
    except : 
        ny, nx = np.shape(fin['figram'])[1:]
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
print("Last date in InSAR:",last)

#Real spatial grid 
lontmp = np.fromfile(lonfile,dtype=np.float32)
lattmp = np.fromfile(latfile,dtype=np.float32)
lon    = np.reshape(lontmp,(ny,nx))
lat    = np.reshape(lattmp,(ny,nx))

if PLOT :
    fin = h5py.File(infile,'r')
    if ('STACK' in infile) or ('interf' in infile):
        toplot = fin['igram'][-1,:,:]
    elif 'Phases' in infile:
        toplot = fin['rawts'][:,:,-1]

print("For info print geographical corners:",\
        lon[0,0],lat[0,0],"\n",
        lon[-1,0],lat[-1,0],"\n",
        lon[-1,-1],lat[-1,-1],"\n",
        lon[0,-1],lat[0,-1])

#-------------------------------------------------------------------------    
#Get earthquake properties
elon,elat,Deq,Mw = np.loadtxt(eqinfo, unpack=True, usecols=list(range(4)), dtype =np.float)
Yr,Mo,Dy,Hr,Min  = np.loadtxt(eqinfo, unpack=True, usecols=list(range(4,9)), dtype ='i4')

if isinstance(Yr,np.int32):
    #Special case of one earthquake
    Leq = 1
    Yr,Mo,Dy,Hr,Min  = np.array([Yr]),np.array([Mo]),np.array([Dy]),np.array([Hr]),np.array([Min])
    elon,elat,Deq,Mw = np.array([elon]), np.array([elat]), np.array([Deq]), np.array([Mw])
else:
    Leq = len(Yr)

#tuples of dates
edate  = [(yr,mo,dy,hr,mi) for yr,mo,dy,hr,mi in zip(Yr,Mo,Dy,Hr,Min)]

if RELOC or EXTEND:
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
    print('Get missing big events from second catalogue')
    iplus   = [i for i,t in enumerate(edate2) if ((Mw2[i] > 4) and (t not in edate)) and (t<(last.year,last.month-1,last.day,0,0))]
    
    #Extend in time to recent period not necessarily in GCMT catalogue
    #print('Last date in first catalogue:', edate[-1])
    #iplus = [i for i,t in enumerate(edate2) if ((t>edate[-3]) and (t not in edate)) and (t<(last.year,last.month-1,last.day,0,0))]
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

#limits of track 
spatialprint = path.Path([(lon[0,0],lat[0,0]),
        (lon[-1,0],lat[-1,0]),(lon[-1,-1],lat[-1,-1]),
        (lon[0,-1],lat[0,-1]),(lon[0,0],lat[0,0])])
inzone = spatialprint.contains_points(np.array([elon,elat]).T)

#Pre-Select relevent events to reduce distance computation 
maskeq = np.ones(Leq)
maskeq[~inzone] = 0             #inside track footprint
maskeq[teq<0] = 0               #occur after start of time series

if SELECT:
    maskeq[Deq>=30] = 0               #depth inferior to 30km (basly constrained)
    maskeq[teq<0.25] = 0              #don't occur in first 3 monthes
    #maskeq[teq>4.5] = 0
    maskeq[Mw<=4] = 0               #Mw larger than 3.7

Yr, Mo, Dy = Yr[maskeq>0], Mo[maskeq>0], Dy[maskeq>0]
Hr, Min = Hr[maskeq>0],Min[maskeq>0]
Deq  = Deq[maskeq>0]
Mw   = Mw[maskeq>0]    
elon = elon[maskeq>0]
elat = elat[maskeq>0]
teq  = teq[maskeq>0]
Leq  = len(teq)

if np.sum(Mw>6.2)>0:
   print('''WARNING: this method is only appropriate for point-source like earthquakes
                   excluding large magnitude earthquakes (Mw>6-6.3?)''')

#Prepare table of Meta-data for plot and easy mask
InfoOut = np.column_stack((elon, elat, Yr, Mo, 
                Dy, Hr, Min, Deq, Mw))

#Select corresponding pixel
print('Start looking for closest pixel')
Xeq,Yeq = np.zeros((2, Leq), dtype=int)

for i in range(Leq):
    # distance to station point
    dist = np.sqrt((lon-elon[i])**2 + (lat-elat[i])**2)
    # get index of minimum(s)
    ind = np.unravel_index(dist.argmin(), lon.shape)
    Yeq[i],Xeq[i] = ind

# Condition on where is the earthquake in the image
maskeq = np.ones(Leq)
if SELECT:
    maskeq[(Xeq>1)^(Yeq>1)] = 0       #not at the edge of image
    maskeq[(Xeq<nx-1)^(Yeq<ny-1)] = 0

    teq = teq[maskeq>0]
    Xeq = Xeq[maskeq>0]
    Yeq = Yeq[maskeq>0]
    InfoOut = InfoOut[maskeq>0]

#Radius of influence for Spatial P0
'''Rinf = (1/(mu*Deq[maskeq>0]*10**3)*10**(1.5*Mw[maskeq>0]+9.1))**1/2.*10**(-3) #in km  
Rinf = Rinf*22.   #amplify to capture whole signal 
Rinf = Rinf+10.   #to account for loc uncertainty (measured in defo map)
'''
print("WARNING: fix the influence radius to 9 km for Mw<5.5")
Rinf = np.full(teq.shape,9.0) # fix to 9 km 
Rinf[InfoOut[:,8]>=5.5] = 12 

#Rough amplitude of expected slip (a priori standard deviation)
print("WARNING: fix the standard deviation of earthquake slip to 30 mm for Mw<5.5")
Aeq = np.full(teq.shape,30.)
Aeq[InfoOut[:,8]>=5.5] = 50.

#Distance between pixels (averaged over 100 pixels around center) 
inclon = abs(lon[Yeq,Xeq]-lon[Yeq,Xeq+10])*111./10. #km 
inclat = abs(lat[Yeq,Xeq]-lat[Yeq+10,Xeq])*111./10.

#Build Output
DataOut = np.column_stack((Xeq,Yeq,teq,Rinf,Aeq,inclon,inclat))

#Sort according to date
order = DataOut[:,2].argsort()
DataOut = DataOut[order]
InfoOut = InfoOut[order]

#Verify earthquakes are distinguishable
#Compute all pairs distances (in time and space)
Xeq,Yeq,teq = np.array(DataOut[:,0]),np.array(DataOut[:,1]),np.array(DataOut[:,2])
dtime = np.abs(teq[:,None] - teq) #Use broadcasting
locarr = np.array([np.array(Yeq)*np.mean(inclat),np.array(Xeq)*np.mean(inclon)]).T
dloc = distance.cdist(locarr,locarr)

idA,idB = np.where((dtime <0.08) & (dloc < 30))   #1 month, 20 km
idsame = [(a,b) for a,b in zip(idA,idB) if a < b] #extract uniq pair from matrice
maskeq = np.ones(len(teq))

if len(idsame)>0:
    print('WARNING: Identify closeby earthquakes')
    print("Indexes of undistinguishable earthquakes",idsame)

for pair in idsame:
    a,b = pair
    print('Occured',round(teq[a],3),round(teq[b],3),'at locations',(Xeq[a],Yeq[a]),(Xeq[b],Yeq[b]))
    print('Dates {}/{}/{}'.format(InfoOut[a,2],InfoOut[a,3],InfoOut[a,4]),
                '{}/{}/{}'.format(InfoOut[b,2],InfoOut[b,3],InfoOut[b,4]))
    print('Magnitudes',InfoOut[a,-1],InfoOut[b,-1])
    
    # Keep earthquake with largest estimated magnitude
    if InfoOut[a,-1] >= InfoOut[b,-1]:
        maskeq[b] = 0
    else:
        maskeq[a] = 0

        
DataOut = DataOut[maskeq>0,:]        
InfoOut = InfoOut[maskeq>0,:]

#-------------------------------------------------------------------------   
#Save in File
np.savetxt(outfile,DataOut,fmt='%i %i %1.5f %1.2f %1.2f %1.5f %1.5f')
print('Saved file :',outfile)

#Save corresponding information for GMT plot and record    
np.savetxt(outplotinfo,InfoOut,fmt='%1.5f %1.5f %i %i %i %i %i %1.2f %1.1f')

#-------------------------------------------------------------------------   
#Plot to visualise earthquake
if PLOT:
    plt.figure(figsize=(5,11))
    plt.pcolormesh(lon,lat,toplot,cmap='jet',vmin=-90,vmax=60) #fin['figram'][-10,:,:])
    plt.scatter(InfoOut[:,0],InfoOut[:,1],2**InfoOut[:,-1],c=InfoOut[:,-2],cmap="plasma") #Mw,Depth
    for i in range(DataOut.shape[0]):
        plt.text(InfoOut[i,0],InfoOut[i,1],"{}".format(i),color='r',fontsize=7)
    
    plt.savefig(outfig,dpi=230)
    print('Saved figure :', outfig)
    
    
    
