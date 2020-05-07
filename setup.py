import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
     name='KFTS-InSAR',  
     version='0.1',
     scripts=['kfts.py'],
     author="Manon Dalaison & Romain Jolivet",
     author_email="dalaison@geologie.ens.fr",
     description="Kalman Filter for time series analysis of InSAR data",
     long_description=long_description,
     long_description_content_type="text/markdown",
     url="https://github.com/ManonDls/KFTS-InSAR",
     packages=setuptools.find_packages(),
     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent", 
     ],
     python_requires='>=3.6',
 )
