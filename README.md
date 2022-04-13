# Kalman Filter for time series analysis of InSAR data

A library to iterativelly recover the phase evolution over time from interferograms 
(stored in HDF5 file)


v0.1.0 :[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3816783.svg)](https://doi.org/10.5281/zenodo.3816783)

### Documentation

https://manondls.github.io/KFTS-InSAR/

[Dalaison & Jolivet (2020)](https://doi.org/10.1029/2019JB019150)


### Assumptions

+ Assuming python 3 is installed on your system.
+ H5py installed with gcc8 and openmpi variants for multiprocessing (py36-h5py @2.10.0_1+gcc8+openmpi)
+ Numpy installed with gcc8 and openblas variants for multiprocessing (py36-numpy @1.18.1_0+gcc8+openblas)

Install `kf` on your system using : 

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
