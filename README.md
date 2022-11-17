# Kalman Filter for time series analysis of InSAR data

[![Language](https://img.shields.io/badge/python-3.6%2B-blue.svg)](https://www.python.org/)
[![License: GPL v3](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](https://github.com/ManonDls/KFTS-InSAR/blob/master/LICENCE.txt)
[![Citation](https://img.shields.io/badge/doi-10.1029%2019JB019150-blue)](https://doi.org/10.1029/2019JB019150)

A library to iterativelly recover the phase evolution over time from interferograms with uncertainties 
(stored in HDF5 file)


### Documentation

https://manondls.github.io/KFTS-InSAR/

[Dalaison & Jolivet (2020)](https://doi.org/10.1029/2019JB019150)


### Assumptions

+ Assuming python 3 is installed on your system.
+ H5py installed with openmpi variant for multiprocessing (py36-h5py @2.10.0_1+openmpi)
+ Numpy installed with openblas variant for multiprocessing (py36-numpy @1.18.1_0+openblas)

Install `kf` on your system using (optional if KFTS-InSAR in PYTHONPATH): 

```
python -m pip install dist/KFTS_InSAR-0.1-py3-none-any.whl
```

### Easy test

A sample dataset (set of unwrapped interferograms) is provided in  `testdata/`
To run KFTS on this dataset : 
```
python kfts.py -c configs/config_Etna.ini 
```

To quickly visualize outputs and plot metrics :
```
python kf2rms.py -c configs/config_Etna.ini 
python checkinnov.py -c configs/config_Etna.ini 
python plotoutput.py -c configs/config_Etna.ini  -geom ./ -rmsTh 2
```
### Synthetics 

To reproduce the synthetic data set in Dalaison & Jolivet (2020)
```
python synthetic_data.py
```

### Prepare input H5file

#### Directly from ISCE files 
KFTS can be selfsuficient with a working routine starting directly from ISCE outputs (`.unw`, `.unw.vrt`, `.unw.xml`...). If other software are used, adjustment of formats and file structure probably has to be done. Parameters in config file must be adjusted to your setting. 
```
python prepare_input.py -c configs/config_prepareinput.ini 
```
Many options are not required by KFTS but help clean the interferogram stack. Examples include deramping, filtering out low coherence region, producing longitude and latitude files with the same geometry as interferograms, cutting interferogram edges. Referencing to a common region defined in all interferograms (not NaN) is compulsory. Zeros in interferograms are assumed to be NaNs. 

#### Using MintPy functionalities

If you want, KFTS is compatible with [MintPy](https://github.com/insarlab/MintPy) functionalities. KFTS provides an independant way to estimate time series from unwrapped interferograms in an iterative way with associated uncertainty propagation. The iterative nature of KFTS allows for a flexible model description of deformation adapting over time to new data acquisition and transiant event occurences (e.g. earthquakes). Therefore, it offers a potential towards long-duration (several years) and continuous monitoring. You can prepare your interferogram stack with MintPy runing the first steps in `smallbaselineApp.py` routine (e.g. `'load_data', 'modify_network', 'reference_point', 'quick_overview', 'correct_unwrap_error'`). KFTS will load `inputs/ifgramStack.h5` and substitute to the `'invert_network'` step.  For instance, you can use : 
```
python smallbaselineApp.py inputs/smallbaselineApp.cfg --end correct_unwrap_error

```
Please refer to MintPy documentations for details about this command. Then KFTS substitutes to the `'invert_network'` step of MintPy. It will run as usual specifying `infile  = inputs/ifgramStack.h5` and `fmtfile = MintPy` in KFTS config file.  
By essence, KFTS assumes interferograms are already cleaned from most known sources of error (stratified atmosphere, topography, tides...) so that the output is directly interpretable with its uncertainty. Additional reworking of the time series output requires aditional work on uncertainty propagation. 
