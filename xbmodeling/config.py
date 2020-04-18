# This is a configuration file where we should put
# default and 'hard-coded' values. It's a one-stop
# shop so we don't have to make a bunch of changes
# in the code if we want to change these values

#import io

# Values for the generative model
modelconf = {
    "cmbFile" : "inputmaps/COM_CMB_IQU-143-fgsub-sevem-field-Pol_1024_R2.01_full.fits", # Default CMB map
    "groundTemperature" : 300, # Default ground temperature
    "beamParams" : [1, 0, 30, 30, 30, 1],
    "extendedBeam" : [0.25, 120, 120, 60, 60, 0]

}

#"BeamParams" : io.load_beam_params()