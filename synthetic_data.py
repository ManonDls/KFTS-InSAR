#---------------------------------------------------------
# Create synthetic dataset (interferometric network)
# for testing KFTS behaviour like in 
# Dalaison & Jolivet (2020)
#
#----------------------------------------------------------

import numpy as np

import kf.utils.createinput as cinpt
import kf.utils.createdeformation as draw

#----------------------------------------------------------
### PARAMETERS
loc = 'testdata/synthfault/'

## Acquisition properties
#temporal frame
final_t = 3*365.           # Final time
delta_t = 12.              # Time increment
start_t = 2013.0           # Initial time in decimal year

#spatial domain
nx,ny = (60,55)            # Number of pixels in X and Y
lonmin,lonmax = 29,31      # Longitude bounds 
latmin,latmax = 29,31      # Latitudes bounds

#noise level 
sig_y   = 10.              # Std of unmodeled phase delay
sig_i   = 0.1              # Std of interferometric network misclosure

## Signal properties 
# (designed to look like tectonic deformation)
freq    = (2.0*np.pi)/365 # Frequency of oscillating signal 
model  = [('POLY',1),('SIN',freq),('COS',freq),('ISPLINE',2,210,100),('STEP',500)]
m_r    = [0.0,40./365.,6.,4.,150.,100.]  #in mm & mm/jour
#P_par  = np.square([10.0,0.05,5.,5.,70.,70.,0.])  #a priori error estimate (last=0 for 1st phase)

#fault geometry
strike = 150. #degree to north
lon0 = 29.2
lat0 = 30.5


#----------------------------------------------------------
#----------------------------------------------------------
### Build synthetic data
np.random.seed(46)         # fix random sequence generated

#setup epochs
time   = np.arange(0,final_t,delta_t)      #in days 
date   = start_t + time/365.               #in absolute decimal years
L      = len(m_r)
rr     = np.square(sig_i)    #incertitude sur données (!=0)

#construct 2-D mesh
x = np.linspace(lonmin, lonmax, nx)      #lon
y = np.linspace(latmin, latmax, ny)      #lat
yv, xv = np.meshgrid(y, x, indexing='ij') 

#fault geometry y = ax+b
a = -np.tan(np.pi/2.-(360.-strike)*np.pi/180.) #slope of fault
b = lat0-lon0*a

#fault deformation from Okada dislocations using CSI 
#slow slip
print("Slow slip at lon {} lat {}".format(lon0,lat0))
print("On day 210, equivalent to 2013/7/29")
#draw.createsyndefo(m_r[-2],nx,ny,lonmin,lonmax,latmin,latmax,
#                lon0=lon0, lat0=lat0, strike=strike, length=150.,
#                filename = loc+'fault_slip1.enu')
ve_ss,vn_ss,vu_ss = draw.readfile(nx,ny,loc+'fault_slip1.enu')

#earthquake on the same fault not on same location
X0 = 29.7
Y0 = a*X0 + b
print("Earthquake at lon {} lat {}".format(X0,Y0))
print("On day 500, equivalent to 2014/5/15")
#draw.createsyndefo(m_r[-1],nx,ny,lonmin,lonmax,latmin,latmax,
#               lon0=X0, lat0=Y0, strike=strike, length=90., width=90.,
#                filename = loc+'fault_slip2.enu')
ve_eq,vn_eq,vu_eq = draw.readfile(nx,ny,loc+'fault_slip2.enu')

#velocity field for locked fault
vel = draw.interseismic_velo(-m_r[1], xv,yv,fault_geom=(a,b))

#----------------------------------------------------------
## Initiate parameters of the deformation model for all pixels 
#(True values of the model-related state vector)
m_real = np.zeros((len(y),len(x),L))
m_real[:] = m_r

#space-dependent parameters
m_real[:,:,-1] = ve_eq
m_real[:,:,-2] = ve_ss 
m_real[:,:,1] = vel

#add noise
m_real   = m_real +np.array(m_real)*sig_y/100.\
                *np.random.normal(size=(len(y),len(x),L))


#----------------------------------------------------------
## Construct InSAR "data"

syn = cinpt.SynteticKF(time,startdate=start_t)
Ref, Phas, m_ph  = syn.create_timeseries(model, m_real, sig_y, atmo=True)
interf, R, links = syn.create_interfero(rr, sig_i, t_sep=4, perp_dist=700)

# Save output
syn.time *= 1/365.
cinpt.writeh5input(syn, loc+"Synth_interf.h5")

# Save True State
cinpt.writeh5timeseries(syn, loc+"True_phases.h5")
cinpt.writeh5model(m_real, loc+"True_params.h5")

# Save lon/lat
lat = yv.flatten().astype(np.float32)
lon = xv.flatten().astype(np.float32)

lon.tofile(loc+'lon.flt')
lat.tofile(loc+'lat.flt')
