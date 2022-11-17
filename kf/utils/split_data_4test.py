import numpy as np
import h5py 
import readinput_mpi as infmt


## Config 
main    = 'TestData/Etna/'
infile  = 'PROC-STACK.h5'     #path 
fmtfile = 'Etna'   			  #string 
VERBOSE = True     			  #boolean
NumRmvd = 2 			      #integer

## Read full data 
data = infmt.SetupKF(main+infile, fmt=fmtfile, verbose=VERBOSE)
data.get_interf_pairs()

if fmtfile == 'ISCE':
    intrfname = 'figram'
elif fmtfile == 'RAW':
    intrfname = 'igram'
elif fmtfile == 'Etna':
    intrfname = 'igram_aps'

#Find line with recent interferograms
intindx  = [i for i in range(data.links.shape[0]) if -1. in data.links[i,-NumRmvd:]]
notintindx = [i for i in range(data.links.shape[0]) if i not in intindx]

#Create files
fold = h5py.File(main+'PROC-STACK-OLD.h5', 'w')
fnew = h5py.File(main+'PROC-STACK-NEW.h5', 'w')

## Divide in two 
#  OLD
fold.create_dataset('tims',data=data.time[:-NumRmvd])
fold.create_dataset('dates',data=data.orddates[:-NumRmvd])

fold.create_dataset(intrfname,data=data.igram[notintindx]) 
fold.create_dataset('Jmat',data=data.links[notintindx,:-NumRmvd]) 
fold.create_dataset('bperp',data=data.bperp[notintindx]) 

#  NEW
#overlap ? 
lent = NumRmvd +data.max_tsep

fnew.create_dataset('tims',data=data.time[-lent:])
fnew.create_dataset('dates',data=data.orddates[-lent:])
 
fnew.create_dataset(intrfname,data=data.igram[intindx]) 
fnew.create_dataset('Jmat',data=data.links[intindx,-lent:]) 
fnew.create_dataset('bperp',data=data.bperp[intindx]) 

print("Write 2 files in: {}".format(main))

