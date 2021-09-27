#####################################################################
# Script to create deformation field typical to simple fault slip
# used for synthetic test of Kalman Filter
#
# date : 13/11/2018
# author : Manon 
# python 3.6
#####################################################################

#import global stuff
import numpy as np

def createsyndefo(dislo,nx,ny,lonmin,lonmax,latmin,latmax,lon0=None,lat0=None,\
                    strike=0.0,length=100.,width=70.,filename='fault_defo.enu'):
    ''' 
    To create synthetic deformation field due to fault dislocation 
    from simple okada computation 
    Args :  * dislo : amount of along strike dislocation 
            * ny, ny : number of pixels along longitude and latitude
            * lonmin, lonmax, latmin, latmax : bounds of spatial domain 
            * lon0, lat0 : fault location (optional, default is center of domain)
            * strike : fault strike
    '''
    
    # import CSI (Classic Slip Inversion) by R. Jolivet Z. Duputel and M. Simons
    # http://www.geologie.ens.fr/~jolivet/csi/
    import csi.planarfault as fault
    import csi.gps as data
    
    if lon0==None or lat0==None:
        #center the fault
        lon0 = lonmin+ (lonmax-lonmin)/2.
        lat0 = latmin+ (latmax-latmin)/2.
    
    #create a Fault
    flt = fault('my example fault', lon0=lon0, lat0=lat0)
    
    #fault properties: lon, lat, depth, strike, dip, fault length, width, number of patches in strike, in dip
    flt.buildPatches(lon0, lat0, 1., strike, 40., length, width, 1, 1)
    
    #define fault slip : along strike, dip and tensile (=0)
    flt.slip[0] = dislo
    
    #create grid of GPS stations
    gps = data('my test', lon0=lon0, lat0=lat0)
    
    names = ['ST'+str(i) for i in range(nx*ny)]
    lons = np.linspace(lonmin,lonmax,nx)
    lats = np.linspace(latmin,latmax,ny) 
    grd  = np.meshgrid(lats, lons, indexing='ij') #lat lon or y x 
    
    gps = data('my test', lon0=lon0, lat0=lat0)
    gps.setStat(names, grd[1].flatten(), grd[0].flatten(), loc_format='LL', initVel=True)
    
    #compute greenfunctions
    flt.buildGFs(gps, slipdir='sd')
    
    #get deformation
    gps.buildsynth(flt, vertical=True)
    gps.write2file(filename, data='synth')

def readfile(nx,ny, filename='fault_defo.enu'):
    
    v_east,v_north,v_up = np.loadtxt(filename,usecols=(3,4,5),skiprows=1,unpack=True)
    [v_east,v_north,v_up] = [np.reshape(el,(ny,nx)) for el in [v_east,v_north,v_up]]
    #lon,lat same as xv,yv --> not exported
    
    return v_east,v_north,v_up
    
def interseismic_velo(dislo, xv,yv,dip=40.,fault_geom=(1,0)):
    '''
    interseimic 2D velocity field 
        * dislo : amount of long term slip along fault 
        * xv,yv : 3D mesh
        * dip   : fault dip 
        * fault_geom : geometry of fault (ax+b)
    '''
    
    (a,b) = fault_geom #y=ax+b
    B,A,C = 1,-a,-b    #Ax+By+C=0
    
    #distance to fault for every point
    dist = (A*xv+B*yv+C)/np.sqrt(A**2+B**2)*(111.*10**3) #in m 
    
    #parameters
    S     = dislo             #slip (mm/yr)
    D     = 20.*10**3         #locking depth (m) 
    Xd    = D/(np.tan(dip)) 
    gama  = (dist -Xd)/D
    
    #Coseismic
    U1= S/np.pi *(np.cos(dip)*(np.arctan2((dist -Xd),D)-np.pi/2.*np.sign(dist))
	   +(np.sin(dip)-gama*np.cos(dip))/(1+gama**2))
    
    #Interseismic (horizontal)
    Uh =  -S/2.*np.cos(dip)*np.sign(dist) - U1
    
    return Uh

#EOF
