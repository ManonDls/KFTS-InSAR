####
# KFTS quality check
# Script to display innovation and check for patterns
#
# Manon Dalaison 2020
####

#Global import
import numpy as np
import matplotlib
matplotlib.use('Agg') #so that don't ask for display!
import matplotlib.pyplot as plt
import h5py
import os
import datetime as dt

import configparser
import argparse
from ast import literal_eval

#Local import 
import kf.readinput as infmt
from kf.timefunction import TimeFct


#----------------------------------------------------------------

#First get file name from inline argument
parser = argparse.ArgumentParser( description='Plot KFTS info in Updates.h5')
parser.add_argument('-c', type=str, dest='config', default=None,
          help='Specify INI config file of KFTS')
args = parser.parse_args()


#Read config file
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
    eqinfo =  os.path.join(loc,eqinfo)

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
        Rinf,sx,sy = [Rinf],np.array(sx),np.array(sy)
    elif isinstance(teq,np.ndarray):
        teq = teq.tolist()
        Leq = len(teq)
    else:
        assert False, "Verify content of {}".format(eqinfo)

    teq.insert(0,'STEP')
    model.insert(len(model),tuple(teq))
    print('New functional model with earthquakes :',model)

infile = os.path.join(outdir,'Updates.h5')

#---------------------------------------------------------------------
#Import 
data = infmt.SetupKF(datfile, fmt=fmtfile)
tfct = TimeFct(data.time,model)
dates = [dt.datetime.fromordinal(dd) for dd in data.orddates]

#Import and read innovation
fin = h5py.File(infile,'r') #dictionary
innov = fin['mean_innov']
gain = fin['param_gain']
Ny, Nx, Nt, L = np.shape(gain)

labels = tfct.get_label(L,'mm')

#--------------------------------------------------------------------
#Compute arrays

meanI = np.nanmean(innov,axis=2)
medI  = np.nanmedian(innov,axis=2)
stdI  = np.nanstd(innov,axis=2)
lastI = innov[:,:,-1]

cin1,cax1 = np.median(meanI[meanI!=0.0]) +np.quantile(meanI[meanI!=0.0],0.999), \
                np.median(meanI[meanI!=0.0]) -np.quantile(meanI[meanI!=0.0],0.999)
cin2,cax2 = np.median(medI[medI!=0.0]) +np.quantile(medI[medI!=0.0],0.999), \
                np.median(medI[medI!=0.0]) -np.quantile(medI[medI!=0.0],0.999)
cin3,cax3 = 0., 3*np.quantile(stdI[stdI!=0.0],0.9)
cin4,cax4 = np.median(lastI[~np.isnan(lastI)&(lastI!=0.0)]) \
                +np.quantile(lastI[~np.isnan(lastI)&(lastI!=0.0)],0.999), \
                np.median(lastI[~np.isnan(lastI)&(lastI!=0.0)]) \
                -np.quantile(lastI[~np.isnan(lastI)&(lastI!=0.0)],0.999)

if np.nanmean(innov,axis=2).max()/cax2 > 10**2:
    print("WARNING: some pixels have extremely high innovation")


#--------------------------------------------------------------------
print("** Start plot **")

print("Mean", np.nanmean(innov,axis=2).min(),np.nanmean(innov,axis=2).max())
print("STD", np.nanstd(innov,axis=2).min(),np.nanstd(innov,axis=2).max())
print("last innov", innov[:,:,-1].min(),innov[:,:,-1].max())

#Plot mean median and std innov maps
fig,ax = plt.subplots(1,4,figsize=(13,4),sharex=True,sharey=True)
img0 = ax[0].imshow(meanI, vmin=cin1, vmax=cax1) 
img1 = ax[1].imshow(medI, vmin=cin2, vmax=cax2)
img2 = ax[2].imshow(stdI, vmin=cin3, vmax=cax3)
img3 = ax[3].imshow(lastI, vmin=cin4, vmax=cax4)
plt.colorbar(img0,ax=ax[0],orientation='horizontal',shrink=0.7,aspect=12.)
plt.colorbar(img1,ax=ax[1],orientation='horizontal',shrink=0.7,aspect=12.)
plt.colorbar(img2,ax=ax[2],orientation='horizontal',shrink=0.7,aspect=12.)
plt.colorbar(img3,ax=ax[3],orientation='horizontal',shrink=0.7,aspect=12.)

ax[0].set_title("Mean innovation for all timesteps")
ax[1].set_title("Median innovation")
ax[2].set_title("Standard deviation of innovation")
ax[3].set_title("Innovation at last time step")


#Plot evolution of gain for 4 random pixels
Npix = 5
fig1,ax1 = plt.subplots(1,L,figsize=(13,3.5))
sx,sy = np.random.randint(0,Nx-1,size=Npix),np.random.randint(0,Ny-1,size=Npix)

for k in range(L):
    for j in range(Npix):
        toplot = gain[sy[j],sx[j],:,k]
        toplot[toplot==0]=np.nan
        ax1[k].plot(dates[1:],toplot,'.')
    ax1[k].set_title(labels[k])
    ax1[k].set_xlabel('Time')
    #print("labels are", ax1[k].xaxis.get_majorticklabels())
    plt.setp( ax1[k].xaxis.get_majorticklabels(), rotation=70 )

ax1[0].set_ylabel('Gain')
fig1.tight_layout()

print("** Save plot **")
fig.savefig(locfig+'innovation_stats.png',dpi=150)
fig1.savefig(locfig+'gain_randompixels.png',dpi=150)

