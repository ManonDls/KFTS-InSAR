# -*- coding: utf-8 -*-
###################################################################
# Plot outputs of kalman filter
# 
# Date : April 2018 - September 2020
# Author : M. Dalaison
###################################################################
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import datetime as dt
import h5py
import os

import configparser
import argparse
from ast import literal_eval

# Local stuff
import kf.makeplot as mplt
import kf.readinput as infmt
from kf.timefunction import TimeFct

plt.close('all')               # close all figures
#plt.style.use('myplotstyle')

#--------------------------------------------------------------------
## SETUP

# Initialize shwitches
PROJ  = False         # Project data from LOS to regular axis
RMS   = False
EQS   = False
FAULT = False
TOPO  = False
VOLC  = False


# Info from inline argument
parser = argparse.ArgumentParser( description='Simple automatic plot of KFTS output')
parser.add_argument('-c', type=str, dest='config', default=None,
          help='Specify INI config file used for KFTS')
parser.add_argument('-geom', type=str, dest='geom', default='Stack/',
          help='Where are geometric files with respect to workdir: lon.flt, inc.flt...')
parser.add_argument('-rmsTh', type=float, dest='rmsTh', default=None,
          help='Threshold of max RMS (typically 5 mm)')
parser.add_argument('-topo', type=str, dest='topo', default=None,
          help='Name and loc of the DEM file with respect to workdir (dem.wgs84.grd)')
parser.add_argument('-fault', type=str, dest='fault', default=None,
          help='Name and loc of the fault file')
parser.add_argument('-earthquake', type=str, dest='eq', default=None,
          help='Name and loc of the list of earthquakes')
parser.add_argument('-volc', type=str, dest='volc', default=None,
          help='Name and loc of the file containing volcanoe location')
parser.add_argument('-box', type=float, nargs=4, dest='box', default=None,
          help='Minimum_longitude maximum_longitude minimum_latitude maximum_latitude (no comma)')

args = parser.parse_args()

# Read config file
config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
config.read(args.config)

loc     = config['INPUT'].get('workdir', fallback='./')
outdir  = os.path.join(loc, config['OUTPUT'].get('outdir', fallback=''))
locfig  = os.path.join(loc, config['OUTPUT'].get('figdir', fallback=''))
datfile = os.path.join(loc, config['INPUT'].get('infile'))
fmtfile = config['INPUT'].get('fmtfile', fallback='ISCE')

model   = literal_eval(config['MODEL SETUP'].get('model'))
EQ      = config['MODEL SETUP'].getboolean('EQ', fallback=False)
eqinfo  = config['INPUT'].get('eqinfo', fallback=None)
if eqinfo is not None:
    eqinfo = os.path.join(loc, eqinfo)

# Adjust model in the case of earthquakes
if EQ == True:
    #Get earthquake properties in actual reference frame
    #file obtained by running "earthquake2step.py"
    Xeq,Yeq,teq,Rinf,Aeq,sx,sy = np.loadtxt(eqinfo, unpack=True)
    Xeq = Xeq.astype('int32')
    Yeq = Yeq.astype('int32')

    if isinstance(teq,float):
        #specific case of one earthquake only
        Leq = 1
        teq = [teq]
        Xeq,Yeq = [Xeq],[Yeq]
        Rinf,Aeq,sx,sy = [Rinf],[Aeq],np.array(sx),np.array(sy)
    elif isinstance(teq,np.ndarray):
        teq = teq.tolist()
        Leq = len(teq)
    else:
        assert False, "Verify content of {}".format(eqinfo)

    teq.insert(0,'STEP')
    model.insert(len(model),tuple(teq))
    print('New functional model with earthquakes :',model)

# KFTS files
inparm  = outdir +'States.h5'
inrawts = outdir +'Phases.h5'

# Coordinates
if args.geom == 'None':
    args.geom = None
if args.geom is not None:
    lon = loc + args.geom + 'lon.flt'
    lat = loc + args.geom + 'lat.flt'

if args.rmsTh is not None :
    RMS = True
    inrms = outdir +'RMS_map.h5'

if PROJ :
    incfile = loc + args.geom +'inc.flt'
    headfile = loc + args.geom +'head.flt'

if args.topo is not None :
    TOPO = True             # !! May be very heavy to plot !!
    topo = loc + args.topo
        
if args.fault is not None :
    FAULT = True
    flt_file = loc + args.fault  #lon,lat

if args.eq is not None :
    EQS = True
    eq_file = loc + args.eq  #CSV name, lat, lon, ?, mag

if args.volc is not None :
    VOLC = True
    volc_file = loc + args.volc  #lon, lat, name


#--------------------------------------------------------------------
## Import Data

if args.geom is not None:
    # Get longitude and latitude
    lon = np.fromfile(lon, dtype=np.float32)
    lat = np.fromfile(lat, dtype=np.float32)

#print('longitude',np.shape(lon),lon)
#print('latitude',np.shape(lat),lat)
   
print("Import and read Kalman outputs")
fin = h5py.File(inparm,'r')         
L    = np.shape(fin['state'])[-1] -np.shape(fin['indx'])[-1]  #number of parameters
parms  = fin['state'][:,:,:L]          #3D (y,x,m)
prm_std = fin['state_cov'][:,:,:L,:L]
ny, nx  = np.shape(parms)[:2]

fin = h5py.File(inrawts,'r') 
phases  = fin['rawts']        
ph_std  = fin['rawts_std'] 
dates   = fin['tims']


#Lon, lat ready for plotting
print('2D map shape x',nx,'and y',ny)
print('Number of parameters:',L)
if args.geom is not None:
   yv    = np.reshape(lat,(ny,nx))
   xv    = np.reshape(lon,(ny,nx))
else :
   yv,xv = np.meshgrid( list(range(nx)),list(range(ny)) )

#deal with zero lon and lat
if args.box is None:
   minlon,maxlon = np.min(xv),np.max(xv)
   minlat,maxlat = np.min(yv),np.max(yv)
else :
   minlon,maxlon,minlat,maxlat = args.box

print("Spatial box: lon {} {}, lat {} {}".format(minlon,maxlon,minlat,maxlat)) 

if RMS :
    print("Import RMS")
    fin = h5py.File(inrms,'r')
    rms = fin['rms'][:]

if TOPO : 
    print("Import topography")
    tlon,tlat,topodat = mplt.load_topo_grd(topo, xv, yv)
    print("shape topo",np.shape(topodat))

if PROJ :
    los_vect = mplt.inchd2los(incfile, headfile)
    los_map = np.reshape(los_vect,(3,ny,nx))
    print("LOS vect",np.shape(los_vect),np.shape(los_map))  

if EQS :
    print("Get earthquakes properties")
    lateq,loneq,mag = np.loadtxt(eq_file,delimiter=',',usecols=[1,2,4],
                                                        unpack=True,skiprows=1)
    nameeq = np.genfromtxt(eq_file,delimiter=',',usecols=[0],dtype=str,skip_header=1)
    mask= mag>3
    lateq,loneq,mag,nameeq = lateq[mask],loneq[mask],mag[mask],nameeq[mask] 
    print("Earthquake",lateq,loneq,mag)

if FAULT :
    print("Get faults location")
    lonflt,latflt = np.loadtxt(flt_file,unpack=True)
    lonflt2,latflt2 = np.loadtxt(flt_file2,unpack=True)

if VOLC :
    print("Get volcanoes location and name")
    lonvolc,latvolc = np.loadtxt(volc_file,usecols=[0,1],unpack=True)
    namevolc = np.genfromtxt(volc_file,usecols=[2],dtype=str)
    print('Volcanoes',namevolc)

#Initiate model class
mod = TimeFct(dates,model)
mod.check_model()
parm_names = mod.get_label(L,'mm')

#--------------------------------------------------------------------
# Clean and define new arrays

# Select phase of interest 
disp   = phases[:,:,-2]
derr   = ph_std[:,:,-2]
errors = np.sqrt(np.diagonal(prm_std,axis1=2,axis2=3))

# Mask by RMS
if RMS:
    mask = (rms > args.rmsTh)
    parms[mask,:] = np.nan
    disp[mask]    = np.nan
    errors[mask,:] = np.nan
    derr[mask]     = np.nan

# Convert 0 to NaN
parms[parms ==0.0] = np.nan 
errors[errors==0.0] = np.nan

## Mask point with error > mean +/- 2*std
#mask2 = derr > 2*np.nanstd(derr)
#derr[mask2] = np.nan
#disp[mask2] = np.nan

#for i in range(L):
    #mask3 = abs(errors[:,:,i]-np.nanmean(errors[:,:,i])) > 2*np.nanstd(errors[:,:,i])
    #parms[:,:,i][mask3] = np.nan
    #errors[:,:,i][mask3] = np.nan

#---------------------------------------------------------------
## Draw and plot profiles 
'''
vel = mplt.TSAout(m_found[:,:,1],xv,yv) #m_found[:,:,1]
dalong, dacross, prof = vel.getprofile(66.7872, 31.4800, 200., 120., 0.3)

plt.figure(figsize=(10,3))
plt.plot(dalong,prof,'.')

dalong, dacross, prof = vel.getprofile(66.9018, 31.6761, 200., 120., 0.3)
plt.plot(dalong,prof,'.')

plt.savefig(locfig+'profile.png',dpi=200)
'''
#---------------------------------------------------------------
## Plot and Save
print('Start plots')

########## Parameter maps with uncertainties
cm = plt.get_cmap('RdBu_r').copy()
cm.set_bad(color='0.8')

mplt.plot_param_2D(os.path.join(locfig, 'outputparamsKF.png'), parms, L, xv, yv, cm=cm, 
                        names=parm_names, axlim=[minlon,maxlon,minlat,maxlat])

cm = plt.get_cmap('viridis').copy()
cm.set_bad(color='0.8')
mplt.plot_param_2D(os.path.join(locfig, 'outputstd.png'), errors, L, xv, yv, cm=cm, norm='log', 
                        names=parm_names, axlim=[minlon,maxlon,minlat,maxlat])

########### Plot Time series of selected pixels
Npix = 4 
Ypxl= np.random.randint(0,ny-1,size=Npix)   
Xpxl= np.random.randint(0,nx-1,size=Npix)
pixels = [(i,j)for i,j in zip(Ypxl,Xpxl)]
letter = ['A','B','C','D','E','F']

dates = dates[:]
mplt.plot_TS(os.path.join(locfig, 'timeseries_randpxls_one.png'), dates, phases, ph_std,
            pixel=pixels, model=model, params=parms[:], label=letter)

fig0,ax0 = plt.subplots(1,len(pixels),figsize=(11,2.9),sharex=True,sharey=True)
ax0 = ax0.ravel()

# Change reference time of descriptive model 
params = mod.shift_t0(dates[0],parms[:])

k=0
for indx in pixels:
    i,j = indx
    
    curve = mod.draw_model(params[i,j,:])
    ax0[k].errorbar(dates,phases[i,j,:],yerr=ph_std[i,j,:],
                        fmt='.',lw=0.8,color='C0',markersize=4,label='phases')
    ax0[k].plot(dates,curve,'-',c='red',linewidth=0.8,label='model',zorder=4)
    ax0[k].set_xlim(min(dates),max(dates)+0.05)
    ax0[k].set_xlabel('time (years)')
    if 'LISEG' in mod.model[0]:
        ylim = ax0[k].get_ylim()
        for i in range(1,len(mod.model[0])):
            ax0[k].plot([mod.model[0][i],mod.model[0][i]],ylim,'k--',lw=1)
        ax0[k].set_ylim(ylim)
    k +=1

ax0[0].legend(loc='best',fontsize=9)
ax0[0].set_ylabel('displacement (mm)')
for i in range(len(ax0)):
    ax0[i].text(0.88,0.85,letter[i],transform=ax0[i].transAxes,
                                fontsize=16,weight = 'semibold')

fig0.savefig(os.path.join(locfig, 'timeseries_randpxls.png'),dpi=250,bbox_inches='tight')

########### Plot Cummulated displacement 
fig,ax = plt.subplots(1,1,figsize=(9.5,8),sharex=True,sharey=True)

# MAKE CODE CRASH
if TOPO : 
    # Background topo
    cm = plt.get_cmap('gray')(np.linspace(0,1,600))
    #blue = np.array([135/256., 206/256., 250/256., 1]) #for sea
    #cm[400:403] = blue
    newcm = colors.ListedColormap(cm[300:])
    for i in range(len(ax)):
        ax[i].pcolormesh(tlon,tlat,topodat,cmap=newcm,vmax=3000)
# MAKE CODE CRASH

cm = plt.get_cmap('jet').copy()
cm.set_bad(color='0.8')

Vmin, Vmax = np.nanpercentile(disp,1), np.nanpercentile(disp,99)
disp[disp==0]=np.nan
img0 = ax.pcolormesh(xv,yv,disp,vmin=Vmin,vmax=Vmax,cmap=cm) 

plt.plot(xv[Ypxl,Xpxl],yv[Ypxl,Xpxl],'sk',markersize=2)

for yi,xi,i in zip(Ypxl,Xpxl,range(len(Xpxl))):
     plt.text(xv[yi,xi],yv[yi,xi],letter[i],fontsize=8,color='k')

if EQS :
    ax.scatter(loneq,lateq,s=mag,zorder=5)

if FAULT :
    ax.plot(lonflt,latflt,'r-')
    ax.plot(lonflt2,latflt2,'r-')

if VOLC :
    for i in range(len(namevolc)):
        if (lonvolc[i]>np.min(xv) and lonvolc[i]<np.max(xv)) \
                and (latvolc[i]>np.min(yv) and latvolc[i]<np.max(yv)):
            ax.plot(lonvolc[i],latvolc[i],'r^',markersize=4)
            ax.text(lonvolc[i],latvolc[i]+0.01,namevolc[i],
                        fontsize=9,horizontalalignment='center')

plt.colorbar(img0,ax=ax)
fig.savefig(os.path.join(locfig, 'displacement.png'),dpi=250,bbox_inches='tight')

########## Project on regular axes 
if PROJ :
    fig2,ax2 = plt.subplots(1,3,figsize=(8*2,6),sharex=True,sharey=True)
    ax2 =  ax2.ravel()
    
    titles = ["East","North","Up"]
    for i in range(len(ax2)):
        pltdisp = los_map[i,:,:]*disp
        Vmin = np.nanmean(pltdisp) -3*np.nanstd(pltdisp)
        Vmax = np.nanmean(pltdisp) +3*np.nanstd(pltdisp)
        img = ax2[i].pcolormesh(xv,yv,pltdisp,cmap=cm,vmin=Vmin,vmax=Vmax)
        ax2[i].set_title(titles[i])
        plt.colorbar(img,ax=ax2[i])

    fig2.savefig(os.path.join(locfig, 'displacement_proj.png'),dpi=250,bbox_inches='tight')
    
    
#---------------------------------------------------------------
plt.close("all")

print('Saved Figures in ',locfig)
