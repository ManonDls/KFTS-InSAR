########################################
###### Config file for KFTS-InSAR ###### 
########################################

[INPUT]

# Reference directory from which other paths are defined
workdir = testdata/synthfault/

# File containing interferograms
infile  = Synth_interf.h5

# Format of infile (ISCE,RAW or Etna)
# Only used to adjust the name of the interferograms key in "infile"
fmtfile = RAW

# File containing information about earthquake (used only if EQ is True)
# X Y (in pxl number) time (in decimal year since start) radius_of_influence (in km) dx dy 
eqinfo  = EQ_list.txt

#####################################
[OUTPUT]

# Directory for output h5 file (States.h5 and Phases.h5)
# The absolute path will be: workdir+outdsir
outdir  = ./

# Directory for saving figures (used only if PLOT is True)
# The absolute path will be: workdir+figdir
figdir  = ./     

#####################################
[MODEL SETUP]

# Is there earthquake to model? (if True, eqinfo required)
EQ      = True

# Frequency of oscillating signal (rad/year)
freq    = 6.283185

# Std of unmodeled phase delay 
# (same unit as unwrapped interferograms (often mm))
sig_y   = 10.0

# Std of interferometric network misclosure 
# (same unit as unwrapped interferograms)
sig_i   = 0.1

# Functional element of descriptiv model
# see https://manondls.github.io/KFTS-InSAR/func.html
model  = [('POLY',1),('SIN',${freq}),('COS',${freq})]

# A priori std estimate of model parameters (mm)
sig_a  = 10,20.,5.,5.

#####################################
[KALMAN FILTER SETUP]

# Print details?
VERBOSE = True 

# Create and save plots?
PLOT    = False 

# Start from previously computed state (in instate)?
UPDT    = False

# Minimum number of valid acquisition on a pixel
pxlTh   = 10


#####################################
