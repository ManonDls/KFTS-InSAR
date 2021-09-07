#---------------------------------------------------------------------
This folder contains subsample of real data to test KFTS 

Data are 20 by 20 pixels interferograms built using Envisat SAR acquisitions 
over Mount Etna. The data is thought to be representative of a SBAS 
interferometric network. Interferograms are unwraped and contain NaNs where 
the coherence is not high enough for unwrapping. Interferograms are referenced 
with respect to the mean value on 4 pixels (x: 13-15, y: 17-19). 
(see Doin et al 2011 or Dalaison and Jolivet 2020 for details about the dataset)

* Etna_sample.h5 
            HDF5 file containing datasets for KFTS 
            ('h5ls -v Etna_sample.h5' for details)

* Etna_sample_new.h5 
            same as previously but containing new interferograms, 
            designed to test the update of pre-existing KFTS

* Figs/data_NaN.png, Figs/sample_interf.png
            illustrative figures for rapid preview of data

