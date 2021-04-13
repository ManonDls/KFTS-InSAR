# Python 3
#######################################################################
# Compute RMS of the KF solution (Phases.h5) and store it in H5 file 
# Use same config file as KFTS to locate evrything easily
#
# For plotting with GMT need to cenvert *.h5 in *.grd
# Manon Dalaison 2020
#######################################################################

import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
#from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from matplotlib.colors import LogNorm
import time as TIME
import h5py
import sys,os

import configparser
import argparse
from ast import literal_eval

# Local 
import kf.readinput as infmt


# record running time 
start_time = TIME.time()

######################## Read config ##################################
#First get inline arguments (arg.dest)
parser = argparse.ArgumentParser(description='RMS of the KF solution (Phases.h5)')
parser.add_argument('-c', type=str, dest='config', default=None,
          help='Specify INI config file of KFTS')
parser.add_argument('-read', type=bool, dest='read', default=False,
          help="Specify if compute or read RMS from file with a boolean")
parser.add_argument('-lon', type=str, dest='lonfile', default=None,
          help="Longitude file if truncation of the file is needed")
parser.add_argument('-lat', type=str, dest='latfile', default=None,
          help="Latitude file if truncation of the file is needed")
parser.add_argument('-los', type=str, dest='losfile', default=None,
          help="LOS file if truncation of the file is needed")
args = parser.parse_args()


#Read config file
config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
config.read(args.config)

loc      = config['INPUT'].get('workdir', fallback='./')
procfile = loc + config['INPUT'].get('infile')
fmtfile  = config['INPUT'].get('fmtfile', fallback='ISCE')

outdir = loc + config['OUTPUT'].get('outdir', fallback='')
locfig = loc + config['OUTPUT'].get('figdir', fallback='')

subregion=None
TRUNCDATA=False
secFT = config['FOR TESTING']
SUBREGION   = secFT.getboolean('SUBREGION', fallback=False)
if SUBREGION:
    x1,x2,y1,y2 = literal_eval(secFT.get('limval',fallback='0,0,0,0'))
    subregion   = infmt.Subregion(x1,x2,y1,y2)
    print("WARNING: select subregion",x1,x2,y1,y2)
TRUNCDATA   = secFT.getboolean('TRUNCDATA', fallback=False)

infile = os.path.join(outdir,'Phases.h5')

# Output
outfile = os.path.join(outdir,'RMS_map.h5')


######################## Import and read data ##################################
## Interferograms
print("** Read Data **")
data = infmt.SetupKF(procfile, fmt=fmtfile, subregion=subregion)
data.get_interf_pairs()
Nint = data.igram.shape[0]

## Phase evolution from KFTS
print('Opening {}'.format(infile))
fin      = h5py.File(infile,'r') #dictionary
phases   = fin['rawts']
phas_std = fin['rawts_std']
mnpstd   = np.mean(phas_std,axis=2)

if args.read : 
    # Read existing file
    print("** Read RMS file **")
    ofil  = h5py.File(outfile,'r')   
    igram_comp = ofil['igram_comp']
    rms        = ofil['rms']

    print("\n** Start plots **")

else:
    # Open new file 
    ofil       = h5py.File(outfile,'w')    

    # Create new datasets
    igram_comp = ofil.create_dataset('igram_comp',data.igram.shape, dtype='float32')
    rms        = ofil.create_dataset('rms',data.igram.shape[1:], dtype='float32')
    #res        = ofil.create_dataset('res',igram.shape, dtype='float32')


    ############## Reconstitute interferograms and compute RMS #####################

    # Reconstitute interf separately to loop to avoid problem with NaNs 
    # appearing when Links*Phases (np.nan*0 !=0)

    print("** Reconstitute interferograms **")
    ## Have to loop, otherwise too heavy for large Sentinel interfero
    i = 0
    for ip,im in zip(data.iplus,data.imoins):
        sys.stdout.write('\r {}/{}'.format(i,Nint))
        sys.stdout.flush()
        ofil['igram_comp'][i,:,:] = phases[:,:,np.array(ip)]-phases[:,:,np.array(im)] 
        i +=1

    print("\n Time for reconstitution {}".format(TIME.time() - start_time))

    numbint = np.zeros((data.Ny,data.Nx))
    print("** Compute RMS **")
    for x in range(0,data.Ny):
        sys.stdout.write('\r {}/{}'.format(x,data.Ny))
        sys.stdout.flush()
    
        #count number of interferograms for each pixel in the row
        N = np.sum(np.isfinite(data.igram[:,x,:]), axis=0)
        N = N.astype('float')
        N[N==0] = np.nan
        numbint[x,:] = N 

        #compute RMS
        ofil['rms'][x,:] = np.sqrt(np.nansum((data.igram[:,x,:] - igram_comp[:,x,:])**2,axis=0)/N)
        #ofil['res'][:,x,:] = np.abs(igram.value - igram_comp.value)
    
    #-------------------------------------------
    print("\n** Start plots **")

    fig2 = plt.figure()
    ax2 = plt.gca()
    img = ax2.imshow(numbint)
    plt.colorbar(img,ax=ax2)
    fig2.savefig(locfig+"Numb_interf.png",dpi=150)


#plot RMS and estimated error
fig1,ax1 = plt.subplots(1,2,figsize=(11,7.5),sharex=True,sharey=True)

img0 = ax1[0].imshow(rms.value,vmin=np.nanpercentile(rms.value,2),vmax=np.nanpercentile(rms.value,98))#,norm=LogNorm(vmin=0.001,vmax=50))
img1 = ax1[1].imshow(mnpstd,norm=LogNorm(vmin=0.001,vmax=50))

plt.colorbar(img0,ax=ax1[0],shrink=0.6,aspect=15,orientation='horizontal')
plt.colorbar(img1,ax=ax1[1],shrink=0.6,aspect=15,orientation='horizontal')

ax1[0].set_title("RMS in interferogram\n reconstruction max= {}".format(
                                                round(np.nanmax(rms.value))))
ax1[1].set_title("Mean standard deviation\n of phases max= {}".format(
                                                round(np.nanmax(mnpstd)))) 

#fig1.tight_layout()


#plot reconstituted and real igram 
fig,ax = plt.subplots(2,3,figsize=(10,7))
ax = ax.ravel()
plt.suptitle('reconstructed-real (top) and real (bottom) interferograms')

tomap = np.random.randint(0,Nint,size=3)
for i in range(3):
    img  = ax[i].imshow(igram_comp[tomap[i],:,:]-data.igram[tomap[i],:,:],
                        vmin=-10,vmax=10)
    img2 = ax[i+3].imshow(data.igram[tomap[i],:,:])
    plt.colorbar(img,ax=ax[i],shrink=0.6,aspect=15)
    plt.colorbar(img2,ax=ax[i+3],shrink=0.6,aspect=15)

fig.tight_layout()

ofil.close()

#print runing time 
print("--- {} seconds ---".format(TIME.time() - start_time))

# Cut lon and lat products
if args.lonfile is not None:
        width = data.fin['figram'].shape[2]
        length = data.fin['figram'].shape[1]
        lon = np.fromfile(args.lonfile, 'f').reshape((length, width))[y1:y2,x1:x2]
        lon.astype('f').tofile(os.path.join(outdir, 'lon.flt'))
if args.latfile is not None:
        width = data.fin['figram'].shape[2]
        length = data.fin['figram'].shape[1]
        lat = np.fromfile(args.latfile, 'f').reshape((length, width))[y1:y2,x1:x2]
        lat.astype('f').tofile(os.path.join(outdir, 'lat.flt'))
if args.losfile is not None:
        assert False, 'Need to finalize this los cutting thing in PrepIgramStack first'
        width = data.fin['figram'].shape[2]
        length = data.fin['figram'].shape[1]
        los = np.fromfile(args.losfile, 'f').reshape((length, width))[y1:y2,x1:x2]
        los.astype('f').tofile(os.path.join(outdir, 'los.flt'))

######################## Save Figures ##########################################
resol = 200

fig.savefig(locfig+'Data_interfero_sample.png',dpi=resol)
fig1.savefig(locfig+'RMS_kf.png',dpi=resol)
plt.close('all')

