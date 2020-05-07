import scipy.fftpack as fftw
import numpy  as np
#import matplotlib.pyplot as plt
import scipy.spatial.distance as scidis

np.random.seed(46)         # fix random sequence generated

def generateAtmo(shape,sigma,lamb,nt=1):
    '''
    Generate synthetic atmospheric noise as the convolution 
    of a white noise and a decreasing exponential.
        :shape: shape of 2D spatial grid (integer or tuple of integer)
        :sigma: std of noise on one snapchot (float e.g. 1.)
        :lamb:  spatial wavelength (float e.g. 30.)
        :nt:    number of time snapchots (integer, default one)
        
    Returns noise map(s) with shape (shape,nt) if nt>1 '''
    
    if (isinstance(shape,float) or isinstance(shape,int)):
        #square grid 
        shape = int(shape)
        Nx,Ny = shape,shape
    elif len(shape) == 2: 
        #rectangular grid
        Nx,Ny = shape[0],shape[1]
    else :
        assert False,"format of shape not understood"

    # Generate white noise
    if nt == 1:                         #one image of noise to produce
        white = np.random.rand(Ny,Nx) 
    elif nt > 1:
        white = np.random.rand(Ny,Nx,nt)
    else :
        assert False,  "cannot determine the number of temporal snapchots"

    # Generate correl  
    yv,xv = np.meshgrid(list(range(1,Ny+1)), list(range(1,Nx+1)), indexing='ij')
    distances = np.min(scidis.cdist(np.stack((yv.flatten(),xv.flatten())).T, \
                    np.array([[1,1], [1,Nx], [Ny,Nx], [Ny,1]])).reshape((Ny,Nx,4)), axis=2)
    correl = np.exp(-distances/lamb)
    
    if nt > 1: 
        correl = np.repeat(correl[:, :, np.newaxis], nt, axis=2)

    # FFT it
    fwhite = fftw.fft2(white,axes=[0,1])
    fcorrel = fftw.fft2(correl,axes=[0,1])
    noise = np.real(fftw.ifft2(fwhite*fcorrel,axes=[0,1]))
    
    # Rescale
    if nt == 1:
        noise = sigma*(noise-np.mean(noise))/np.std(noise)
    if nt > 1 :
        noise = np.array([sigma*(noise[...,i]-np.mean(noise[...,i]))/np.std(noise[...,i]) for i in range(nt)])
        noise = np.transpose(noise,(1,2,0))
    return noise 
 
#  TEST
# noise = generateAtmo((100,50),2.,5.,nt=10) 
# fig,ax = plt.subplots(1,3)
# img0 = ax[0].pcolormesh(noise[...,0])
# img1 = ax[1].pcolormesh(noise[...,1])
# img2 = ax[2].pcolormesh(noise[...,8])
# fig.colorbar(img0, ax=ax[0])
# fig.colorbar(img1, ax=ax[1])
# fig.colorbar(img2, ax=ax[2])
# plt.show()


