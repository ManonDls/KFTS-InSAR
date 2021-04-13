from __future__ import print_function
###########################################################################@
# Plot outputs of time series analysis for InSAR
#
#Date : July 2018
#Author : Manon Dalaison 
###########################################################################@

from builtins import range
import numpy as np
import netCDF4
import operator
import scipy.linalg as la
import scipy.interpolate as sciint

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as mcm


###########################################################################
# WORK FROM DATA
    
class MidpointNormalize(colors.Normalize):
    ''' To center divergent colorbar on zero
    (or other midpoint value)
    '''
    def __init__(self, vmin, vmax, midpoint=0, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        from scipy import ma, interp
        
        normalized_min = max(0., 1. / 2 * (1 - abs((self.midpoint - self.vmin) / (self.midpoint - self.vmax))))
        normalized_max = min(1., 1. / 2 * (1 + abs((self.vmax - self.midpoint) / (self.midpoint - self.vmin))))
        normalized_mid = 0.5
        x, y = [self.vmin, self.midpoint, self.vmax], [normalized_min, normalized_mid, normalized_max]
        return ma.masked_array(interp(value, x, y))

def plot_covariance(fig,ax,m,P,L):
    ''' 
    Plot matrix P in 2 subplots : 
    one for the parameter covariance
    one for the phase covariance
    '''
    if P.ndim == 2 :
        img0 = ax[0].pcolormesh(P[:L,:L])
        img1 = ax[1].pcolormesh(P[L:,L:])
        LL = np.shape(P)[0]- L
        ax[0].axis([0,L, L, 0])
        ax[1].axis([0, LL, LL, 0])
        fig.colorbar(img0, ax=ax[0])
        fig.colorbar(img1, ax=ax[1])
        ax[0].set_title('Estimated covariance of parameters')
        ax[1].set_title('Estimated covariance of phases')
        fig.tight_layout()
    else :
        print('State covariance (P) not 2D use plot_param_2D fct')

def plot_param_2D(figname, m, L, xv, yv, bounds=None, names=None, axlim=None, cm=plt.get_cmap('viridis'), norm='linear', resol=250):
    ''' 
    Plot final parameters for each pixel on 2D maps
    * figname : string containing location and name of the output figure
    * m       : 3D parameter array to plot
    * L       : number of parameters (will select the L first elements of m)
    * xv, yv  : spatial meshgrid for x and y tick label
    * bounds  : color map bounding values (list of pairs with length L)
    * names   : specify name of parameters if known
    * norm    : colorbar normalization 'linear' 'log' or 'center'.
                'center' is for a linear colormap with the midcolor of cm attributed to zero '''
    
    if axlim is None:
        aratio = (np.max(xv)-np.min(xv))/(np.max(yv)-np.min(yv))
    else :
        aratio = (axlim[1]-axlim[0])/(axlim[3]-axlim[2])
    scale = 10
    
    if L <6:
        Taillefigure = (aratio*scale*L+1, 1/aratio*scale)
        fig,ax = plt.subplots(1,L,figsize=Taillefigure,sharex=True,sharey=True) #(L*4,3.5)
    else :
        lcol = L//2+L%2
        Taillefigure = (aratio*scale*lcol+1, 1/aratio*scale*1.8)
        fig,ax = plt.subplots(2,lcol,figsize=Taillefigure,sharex=True,sharey=True)
        ax = ax.ravel()

    if bounds == None:
        bounds = [[]]*L
        for i in range(L):
            if norm in ['linear','center']:
                bounds[i] = [np.nanpercentile(m[:,:,i],1,interpolation='higher'),\
                                np.nanpercentile(m[:,:,i],99,interpolation='higher')]
            elif norm =='log':
                bounds[i] = [np.nanmin(m[:,:,i]),np.nanmax(m[:,:,i])]
        print("bounds",bounds)
        
    #Plot retrieved parameters map
    if names is not None :
        param_names = names
    else:
        print("WARNING: use default parameter names but may be wrong")
        param_names = ['offset','velocity','sine amplitude',
                        'cos amplitude']
        if L > 4 : 
            for k in range(4,L):
                param_names.append('step'+str(k))

    for i in range(L):
        if norm =='linear':
            img0 = ax[i].pcolormesh(xv,yv,m[:,:,i],vmin=bounds[i][0],vmax=bounds[i][1],cmap=cm)
        elif norm =='center':
            normcol = MidpointNormalize(vmin=bounds[i][0], vmax=bounds[i][1], midpoint=0)
            img0 = ax[i].pcolormesh(xv,yv,m[:,:,i], cmap=cm, norm=normcol)
        elif norm == 'log':
            img0 = ax[i].pcolormesh(xv,yv,m[:,:,i],norm=colors.LogNorm(vmin=bounds[i][0],vmax=bounds[i][1]),cmap=cm)
            
        cb = fig.colorbar(img0,ax=ax[i],shrink=0.4,aspect=22)
        ax[i].set_title(param_names[i])
        if i == 1:
            cb.set_label('mm/yr')
        else:
            cb.set_label('mm')
    
        # Bound axes
        if axlim is not None:
            if len(axlim)==2:
                ax[i].set_xlim(left=axlim[0])
                ax[i].set_ylim(bottom=axlim[1])
            elif len(axlim)==4:
                ax[i].set_xlim(left=axlim[0],right=axlim[1])
                ax[i].set_ylim(bottom=axlim[2],top=axlim[3])
            else : 
                assert False, "Format of axlim not understood, should be a 1D array of length 2 or 4"
            
    fig.tight_layout()
    fig.savefig(figname,bbox_inches='tight',dpi=resol)
    plt.close()
    
def plot_TS(figname, time, rawts, err, pixel=None, model=[], params=[], label=[], resol=250, hold=False):
    ''' 
    Plot time series for specific pixel(s) with associated errorbars 
        * figname : string containing location and name of the output figure
        * pixel   : if rawts is not 1D a list of tuples of pxl index 
                (index in same order as in rawts)
        * model : tuple of model elements (optional)
        * params : list of parameter values (optional) see timefunction for details
        * hold : Boolean, hold figure (if True do not save and close figure) '''

    fig,ax = plt.subplots(1,1,figsize=(6,4))

    #load class for model if arguments filled in
    if len(model)>0 and len(params)>0 :
        
        from kf.timefunction import TimeFct
        print("initiate timefunction")
        mod = TimeFct(time,model)
        mod.check_model(verbose=False)
        if time[0] >0:
            params = mod.shift_t0(time[0],params)
            
    #pixel argument not specified means that rawts contain 1 time series
    if pixel==None :
        assert (rawts.ndim==1),"Time series is not 1D, should select pixel(s)"
        plt.errorbar(time,rawts,yerr=err,fmt='.')

    else :
        #identify time index in rawts 
        tidx = 999
        i = 0
        for s in np.shape(rawts):
            if s == len(time):
                tidx = i
            i += 1
        if tidx ==999 or tidx==1:
            assert False, "no axis in time series has the same length as time"
        
        #check if one pixel tuple
        if isinstance(pixel, tuple):
            i,j = pixel
            if tidx ==0:
                plt.errorbar(time,rawts[:,i,j],yerr=err[:,i,j],fmt='.')
            elif tidx==2:
                plt.errorbar(time,rawts[i,j,:],yerr=err[i,j,:],fmt='.')
        
        #for a list of pixels, iterate
        elif isinstance(pixel,list):
            cm = plt.get_cmap('viridis')
            cmcycle = [cm(1.*i/len(pixel)) for i in range(len(pixel))] 
            k=0
            for indx in pixel:
                i,j = indx
                lab = "pxl %i, %i"%(i,j) 
                if len(label) == len(pixel):
                    lab = label[k]
                    
                if tidx ==0:
                    plt.errorbar(time,rawts[:,i,j],yerr=err[:,i,j],fmt='.',color=cmcycle[k],label=lab)
                    if len(model)>0 and len(params)>0 :
                        curve = mod.draw_model(params[:,i,j])
                        ax.plot(time,curve,'-',c=cmcycle[k],linewidth=0.8)
                elif tidx==2:
                    plt.errorbar(time,rawts[i,j,:],yerr=err[i,j,:],fmt='.',color=cmcycle[k],label=lab)
                    if len(model)>0 and len(params)>0 :
                        curve = mod.draw_model(params[i,j,:])
                        ax.plot(time,curve,'-',c=cmcycle[k],linewidth=0.8)
                
                else:
                    print('PROBLEM')

                
                k+=1

            if len(pixel)< 8:
                plt.legend() 
        else : 
            assert False, "Format of pixel not understood"
    
    if np.nanmax(time) > 100.0:
        ax.set_xlabel('Time (days)')
    else :
        ax.set_xlabel('Time (yrs)')
    
    plt.ylabel('Relative displacement (m)')
    plt.title('time series')
    
    if hold == False:
        fig.tight_layout()
        fig.savefig(figname,bbox_inches='tight',dpi=resol)
        plt.close()
    else: 
        return fig,ax

def plot_gpsTS(figname, time, gpsdisp, err, resol=250, hold=False):
    ''' 
    Plot time series for specific pixel(s) with associated errorbars 
        * figname : string containing location and name of the output figure
        * time    : time array (may be 2D if gpsdisp is too)
        * gpsdisp : array of  displacement over time. May be 2D if several time 
                    series with second axis of size len(time).
        * err     : std of gpsdisp with same shape '''
    
    fig,ax = plt.subplots(1,1,figsize=(6,4))
    
    if gpsdisp.ndim ==1 :
        plt.errorbar(time,gpsdisp,yerr=err,fmt='.')
    
    elif gpsdisp.ndim ==2 and err.ndim ==2 : #several time series 
        t = time 
        for i in range(gpsdisp.shape[0]):
            if time.ndim == 2 :
                t = time[i]
            plt.errorbar(t,gpsdisp[i],yerr=err[i],fmt='.')

    else :
        assert False, "Number of dimentions of GPS displacement array seem wrong "
    
    if hold == False:
        fig.tight_layout()
        fig.savefig(figname,bbox_inches='tight',dpi=resol)
        plt.close()
    else: 
        return fig,ax
        
        
def view_mask(mask, locfig):
    '''
    Plot mask and save figure in locfig '''
    
    fig,ax = plt.subplots(1,1)
    ax.pcolormesh(mask)
    fig.savefig(locfig+'pxlmask.png',bbox_inches='tight')
    plt.close()
    
def view_data(igram, interf_list, locfig, dim=(0,0)):
    '''
    Plot interferograms and save figures 
    * igram       : interferograms (3D array)
    * interf_list : selection of interferograms to plot
    * locfig      : directory containing a directory called "interfs" for storage 
    * dim         : optional parameter used if igram is not 3D'''
    
    fig,ax = plt.subplots(5,4,figsize=(10,10.5),sharex=True, sharey=True)
    ax = ax.ravel()
    k = 0 #count elements in figure 
    
    for i in interf_list:
        if igram.ndim==3:
            img = ax[k].pcolormesh(igram[i,:,:])
        else:
            img = ax[k].pcolormesh(np.reshape(igram[:,i],dim))
        fig.colorbar(img,ax=ax[k])
        k += 1
        
        if k == 20 : 
            print('save figure',i)
            plt.tight_layout()
            fig.savefig(locfig+'interfs/interfero_'+str(i+1)+'.png',bbox_inches='tight')
            plt.close('all')
            fig,ax = plt.subplots(5,4,figsize=(10,10.5),sharex=True, sharey=True)
            ax = ax.ravel()
            k = 0 

    fig.tight_layout()
    fig.savefig(locfig+'interfs/interfero_'+str(i+2)+'.png',bbox_inches='tight')
    plt.close()
        

def plot_baselines(t, bperp, imoins, iplus, locfig, cm=plt.get_cmap('jet'), resol=250):
    '''
    Plot baselines and interferograms
    * t             : time array (N)
    * bperp         : perpendicular baselines (M or N) (or None)
    * imoins, iplus : given by get_interf_pairs() (M)
    * cm            : reference color map discretised later
    * resol         : dpi image resolution (default is 250)'''
    
    if not type(t).__module__==np.__name__:  #check if object and convert in array
        t = t.value
    
    if max(iplus-imoins) <=0 :           #if substract higher phase to lower ones 
        iinf, isup = iplus, imoins
    else :                               #if substract lower phase to higher ones 
        iinf, isup = imoins, iplus
    
    if bperp is None :
        #If spatial baseline not given, display temporal baseline only
        print("WARNING: spatial baseline is None")
        
        #sort interferograms by date of first date 
        L = sorted(zip(iinf,isup), key=operator.itemgetter(0))
        iinf,isup = zip(*L)
        iinf,isup = np.array(iinf),np.array(isup)

    elif (len(bperp)==len(imoins)):      #bperp the same lengt as links(axis=0)
        bperpt = np.zeros(len(t))        #create pberp with size of time 
        for i in range(1,len(t)):        #t=0 as reference
            if i in isup:
                k = np.array([indx for indx,val in enumerate(isup) if val==i])
                k = k[0]
                bperpt[i] = bperpt[iinf[k]] +bperp[k]
    
    elif (len(bperp)==len(t)):      #bperp the same length as time
        bperpt = bperp
    
    fig,ax = plt.subplots(1,1,figsize=(8.7,6))
    cmap = [cm(1.*i/len(t)) for i in range(len(t))]
    
    # Plot Network
    if bperp is None:
        ax.plot([t[iinf],t[isup]],[np.array(range(len(iinf))),np.array(range(len(iinf)))],'-',c='0.5',linewidth=0.8)
        ax.set_ylabel('Interferogram number')
    else:
        ax.plot([t[iinf],t[isup]],[bperpt[iinf],bperpt[isup]],'-',c='0.4',linewidth=0.8)
        ax.plot(t, bperpt, '.', markersize=10, c='deeppink')
        ax.set_ylabel('Perpendicular baseline (m)')

    if np.nanmax(t)-np.nanmin(t) > 100.0:
        ax.set_xlabel('Time (days)')
    else :
        ax.tick_params(axis='x',which='minor',direction='in',length=4)
        ax.set_xlim(np.nanmin(t)-0.05,np.nanmax(t)+0.05)
        ax.set_xticks(t, minor=True)
        ax.set_xlabel('Time (yrs)') 

    fig.tight_layout()
    fig.savefig(locfig+'baselines.png',bbox_inches='tight',dpi=resol)
    plt.close()
    
    
def plot_errors(ax0, ax1, time, true, model_kf, model_ls, sig_y,\
            interf, interf_kf, interf_ls):
    '''
    Plot unmodelled phase delay
    '''
    ax0.set_title('unmodelled phase delay')
    ax0.set_xlabel('time (days)')
    ax0.set_ylabel('Error in model (mm)')
    ax0.plot([0,max(time)],[sig_y,sig_y],'k--',label='std of noise on data')
    ax0.plot(time, abs(true - model_kf),c='C0')
    ax0.plot(time, abs(true - model_ls),c='C1')

    ax1.set_title('unwrapping error')
    ax1.set_ylabel('error in interferograms (mm)')
    ax1.set_xlabel('interferogram number')
    ax1.plot(abs(interf-interf_kf),c='C0',label='data - kalman model')
    ax1.plot(abs(interf-interf_ls),c='C1',label='data - least quare model')
    
    moy_kf = np.nanmean(abs(interf-interf_kf))
    moy_ls = np.nanmean(abs(interf-interf_ls))
    ax1.plot([0,len(interf)],[moy_kf,moy_kf],c='blue',linestyle='--')
    ax1.plot([0,len(interf)],[moy_ls,moy_ls],c='orange',linestyle='--')

def plot_interfs(igrams,lonfile,latfile,interf_list,figfile='./',minlat=1e-3,minlon=1e-3,time_ax=0,labels=None,\
            bounds=None, faults=None, zoom=None, cm=plt.get_cmap('jet')):
    '''
    Plot interferograms individually in a loop 
    igrams           : 3D with space in Y,X (lat,lon)
    lonfile, latfile : *.flt with coordinates
    interf_list      : list of interfero number to plot 
    time_ax          : axis with time array over which to itterate (0 ou 2,-1)
    labels           : array of strings to write on corner of plots
    faults           : list of lon,lat array in pairs
    zoom             : list of [minx,maxx,miny,maxy] '''

    if time_ax==0:
        ny,nx = np.shape(igrams)[1:]
    elif time_ax==-1 or time_ax==2:
        ny,nx = np.shape(igrams)[:-1]
    print('ny,nx',ny,nx)
    print('shape igram',np.shape(igrams))

    lon   = np.reshape(np.fromfile(lonfile, dtype=np.float32),(ny,nx))
    lat   = np.reshape(np.fromfile(latfile, dtype=np.float32),(ny,nx))
     
    if np.nanmean(lon) > 180:
        lon -= 360

    # Filter out absurd lon lat by cutting top and bottom 
    # because pcolormesh cannot deal with NaN in xv or yv 
    '''mask = (lat<minlat) & (lon<minlon)
    lon[mask] = np.nan

    # Eliminate rows containing NaNs 
    line_av = np.mean(lon,axis=1)
    idx_fin = np.where(np.isfinite(line_av))[0] # indices of rows filled with finite values
    top,bot = idx_fin[0], idx_fin[-1]-1000
    
    lon   = lon[top:bot,:]
    lat   = lat[top:bot,:]
    
    if time_ax==0:
        igrams  = igrams[:,top:bot,:]
    elif time_ax==-1 or time_ax==2:
        igrams  = igrams[top:bot,:,...]
   ''' 
    if zoom is not None:
        print("Cut image to zoom")
        x1,x2,y1,y2 = zoom
        lon = lon[y1:y2,x1:x2]
        lat = lat[y1:y2,x1:x2]
        if time_ax==0:
            igrams = igrams[:,y1:y2,x1:x2]
        elif time_ax==-1 or time_ax==2:
            igrams = igrams[y1:y2,x1:x2,:]

    if bounds == None:
        print("Warning: Computing bounds of color scale, may be very long for big array")
        bounds = [np.nanmean(igrams)-2*np.nanstd(igrams),np.nanmean(igrams)+2*np.nanstd(igrams)]
        print("bounds",bounds)
    
    #Get appropriate size for Figure
    aratio = (np.max(lon)-np.min(lon))/(np.max(lat)-np.min(lat))
    scale = 10
    
    for i in interf_list:
        print("Interferogram",i)
        fig = plt.figure(figsize=(aratio*scale,1/aratio*scale))   
        
        if time_ax==0:
            plt.pcolormesh(lon, lat, igrams[i,:,:],vmin=bounds[0],vmax=bounds[1],cmap=cm)
        elif time_ax==-1 or time_ax==2:
            plt.pcolormesh(lon, lat, igrams[:,:,i],vmin=bounds[0],vmax=bounds[1],cmap=cm)
        
        if labels is None:
            plt.text(0.1,0.05,str(i),color='r',transform=plt.gca().transAxes)
        else :
            plt.text(0.1,0.05,labels[i],color='r',transform=plt.gca().transAxes)
        
        if faults is not None:
            for fault in faults:
                lonflt,latflt = fault
                plt.plot(lonflt,latflt,'k-',linewidth=0.8)
        
        plt.xlim(np.nanmin(lon),np.nanmax(lon))
        plt.ylim(np.nanmin(lat),np.nanmax(lat))
        plt.colorbar()
        plt.tight_layout()
        
        if labels is None:
            fig.savefig(figfile +'interfero_' +str(i) +'.png', dpi=100)
        else:
            fig.savefig(figfile +'interfero_'+labels[i] +'.png', dpi=100)
        
        plt.close('all')

            
def colorbar(fig, value, legend, cm='jet', bound=None):
    ''' 
    Draw generic vertical colorbar on left of plot
    fig    : figure of interest
    value  : values we want to scale the colors with (e.g. time)
    legend : descriptive text appearing next to the colorbar
    cm     : colormap 
    '''
    if bound !=None and len(bound)==2 :
        minb,maxb = bound[0],bound[1]
    else : 
        minb,maxb = np.nanmin(value),np.nanmax(value)
        
    norm = colors.Normalize(minb,maxb)
    mappable = mcm.ScalarMappable(norm,cm)
    mappable.set_array(value)
    fig.subplots_adjust(right=0.84)
    cb_ax = fig.add_axes([0.92, 0.2, 0.02, 0.6])
    cb = fig.colorbar(mappable,cax=cb_ax)
    cb.set_label(legend,fontsize='12')
    
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n))) 
    return new_cmap

def load_topo_grd(topofile,lon,lat):
    '''to load DEM converted into .grd by gdal_translate
    and cut into shape defined in reference lon, lat 
    to save memory and time when plotting'''
    
    # Get topography
    topo = netCDF4.Dataset(topofile, 'r')
    print(topo)
    tlon,tlat = np.meshgrid(topo.variables['lon'][:],topo.variables['lat'][:])
    topodat = topo.variables['Band1'][:]
    topo.close()
    
    # Cut topo 
    mina0 = np.argmax(tlat > np.nanmin(lat),axis=0)[100]
    maxa0 = np.argmax(tlat > np.nanmax(lat),axis=0)[100]
    mina1 = np.argmax(tlon > np.nanmin(lon),axis=1)[100]
    maxa1 = np.argmax(tlon > np.nanmax(lon),axis=1)[100]
    tlon,tlat = tlon[mina0:maxa0,mina1:maxa1],tlat[mina0:maxa0,mina1:maxa1]
    topodat   = topodat[mina0:maxa0,mina1:maxa1]
    
    return tlon,tlat,topodat
    
    
def inchd2los(incfile, headfile):
    ''' taken and modified from CSI insar.py
    Takes incidence and heading as binary files (e.g. inc.flt)
    '''
    incidence = np.fromfile(incfile, dtype=np.float32)
    heading = np.fromfile(headfile, dtype=np.float32)
    
    # Convert angles
    alpha = (heading+90.)*np.pi/180.
    phi = incidence *np.pi/180.

    # Compute LOS
    Se = -1.0 * np.sin(alpha) * np.sin(phi)
    Sn = -1.0 * np.cos(alpha) * np.sin(phi)
    Su = np.cos(phi)

    return np.array([Se,Sn,Su])

#-----------------------------------------------------------------------------
# Tools to analyse time series analysis (TSA) outputs
#
# Inspired by CSI tools by R. Jolivet
# M. Dalaison - April 2019
#-----------------------------------------------------------------------------
class TSAout(object):
    '''
    Class for plotting and visualizing output of TSA
    '''

    def __init__(self, data, lon, lat, err = None):
        '''
        work on a 2D field'''
        
        #store
        self.dat = data.flatten()
        self.lon = lon.flatten()
        self.lat = lat.flatten()
        self.err = err
        
    def getprofile(self,loncenter, latcenter, length, azimuth, width):
        '''
        Project the SAR velocities onto a profile.
        Works on the lat/lon coordinates system.
        Args:
            * loncenter         : Profile origin along longitude.
            * latcenter         : Profile origin along latitude.
            * length            : Length of profile.
            * azimuth           : Azimuth in degrees.
            * width             : Width of the profile.
        '''
        # GEOMETRY
        xc, yc = loncenter, latcenter
        alpha  = azimuth*np.pi/180.    #azimuth into radians
        length = length/111.           #rough km to degree
        width  = width/111. 
        
        # Compute the endpoints of the profile
        xe1 = xc + (length/2.)*np.sin(alpha)
        ye1 = yc + (length/2.)*np.cos(alpha)
        xe2 = xc - (length/2.)*np.sin(alpha)
        ye2 = yc - (length/2.)*np.cos(alpha)
    
        # Design a box
        x1 = xe1 - (width/2.)*np.cos(alpha)
        y1 = ye1 + (width/2.)*np.sin(alpha)
        x2 = xe1 + (width/2.)*np.cos(alpha)
        y2 = ye1 - (width/2.)*np.sin(alpha)
        x3 = xe2 + (width/2.)*np.cos(alpha)
        y3 = ye2 - (width/2.)*np.sin(alpha)
        x4 = xe2 - (width/2.)*np.cos(alpha)
        y4 = ye2 + (width/2.)*np.sin(alpha)
    
        # make the box
        self.box = [[x1, y1],[x2, y2],[x3, y3],[x4, y4]]
    
        # Get the points in this box.
        # 1. import shapely and nxutils
        import matplotlib.path as path
    
        # 2. Create an array with the positions
        XY = np.vstack((self.lon,self.lat)).T
        
        # 3. Create a box
        rect = path.Path(self.box, closed=False)
        
        # 4. Find those who are inside
        Bol = rect.contains_points(XY)
        
        # 4. Get these values
        xg = self.lon[Bol]
        yg = self.lat[Bol]
        prof = self.dat[Bol]
    
        # Check if lengths are ok
        assert len(xg)>100, 'Not enough points to make a worthy profile: {}'.format(len(xg))
    
        # Get distance along and accross profile (in km)   
        theta = np.arctan2(abs(ye1-ye2),abs(xe1-xe2))                          #angle with respect to vertical 
        Rot = [[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]] #rotation matrix
        Dalong, Dacross = np.dot(Rot,np.array(((xg - xe2)*111.,(yg - ye1)*111.)))
        
        #Dalong = (((xg - xe1)*111)**2+((yg - ye1)*111)**2)**1/2. #euclidian distance to end
        
        # All done
        return Dalong, Dacross, prof
    
    def remove_spat_ramp(self,mask, poly=3):
        '''
        Estimate the best-fitting ramp parameters for a matrix.
        Args:
            * phs     -  Input unwrapped phase matrix.
            * mask    -  Mask of reliable values.
            * poly    -  Integer flag for type of correction.
                            1 - Constant
                            3 - Plane
                            4 - Plane + cross term'''
        
        
        nn = self.dat.shape[0]
        mm = self.dat.shape[1]
        
        # space to fit ramp
        [ii,jj] = np.where((mask != 0) & np.isfinite(self.dat))
        y = ii/(1.0*nn)
        x = jj/(1.0*mm)
        numg = len(ii)
         
        # whole space from which we substract the ramp
        X,Y = np.meshgrid(np.arange(mm)*1.0,np.arange(nn)*1.0)
        X = X.flatten()/(mm*1.0)
        Y = Y.flatten()/(nn*1.0)
        
        # Adjust matrix according to poly degree 
        if poly==1:
            Asub = np.ones(numg)
            A    = np.ones(nn*mm)
        elif poly==3:
            Asub = np.column_stack((np.ones(numg),x,y))
            A    = np.column_stack((np.ones(nn*mm),X,Y))
        elif poly==4:
            Asub = np.column_stack((np.ones(numg),x,y,x*y))
            A    = np.column_stack((np.ones(nn*mm),X,Y,X*Y))
        
        # Get best fitting polynomial
        ramp = la.lstsq(Asub, self.dat[ii,jj], cond=1.0e-8)
        del Asub
        
        # Polynomial corresponding to the rank
        self.ramp = ramp[0] 
        
        #Remove from data 
        dphs = np.dot(-A,self.ramp)
        del A
        dphs = np.reshape(dphs,(nn,mm))
        newdat = self.dat + dphs
        #refph = np.nanmean(newdat[self.ref[0]:self.ref[1], self.ref[2]:self.ref[3]])
                
        return newdat#-refph
