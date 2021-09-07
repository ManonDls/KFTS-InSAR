# Kalman Filter for time series analysis of InSAR data

A library to iterativelly recover the phase evolution over time from interferograms 
(stored in HDF5 file)


v0.1.0 :[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3816783.svg)](https://doi.org/10.5281/zenodo.3816783)

### Documentation

https://manondls.github.io/KFTS-InSAR/

### Assumptions

+ Assuming python 3 is installed on your system.

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
