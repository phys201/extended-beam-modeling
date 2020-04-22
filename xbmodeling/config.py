# This is a configuration file where we should put
# default and 'hard-coded' values. It's a one-stop
# shop so we don't have to make a bunch of changes
# in the code if we want to change these values

# import io

# Values for the generative model
modelconf = {
    "cmbFile": "xbmodeling/inputmaps/commanderR300_full_IQU_NSIDE_512.fits",  # Default CMB map
    "groundTemperature": 300,  # Default ground temperature
    "beamParams": [1, 0, 30, 30, 30, 1],
    "extendedBeam": [0.25, 120, 120, 60, 60, 0],
    "defaultResolution": 125,
}

# "BeamParams" : io.load_beam_params()


mountconf = {}
i2m = 2.54/100 #inches to meters
d2r = 0.0175 #degrees to radians
# 23 Sep 2019
# Mount dimensions taken from Colin's pointing model defaults. Can't find mount in repo.
# FB dimensions from Michael Gordon's Forebaffle construction write up
mountconf["keck"] = {
    "nbaffles" : 5, # number of forebaffles
    #"gsrad" : 600/2*i2m, # in meters
    #"gsheight" : 200*i2m, # in meters
    "gsrad" : 0, # in meters
    #"gsrad" : [2.79,5.05,7.34], # in meters
    "gsheight" : [0,0.94,3.97], # in meters
    "gssides" : 8,
    "fbrad": 25/2*i2m, # in meters
    "fbheight": 29*i2m, # in meters
    "winrad" : 16.1/2*i2m, # in meters. #from membrane ring Dia.
    "aptoffr" : 0.5458, # in meters
    "drumangle" : 0, #211*d2r, # Degrees
    "aptoffz" : 1.5964, # in meters
    "dkoffx" : -1.0196, # in meters
    "dkoffy" : 0.0, # in meters
    "eloffx" : 0.0, # in meters
    "eloffz" : 1.1750, # in meters
}


# Update parameters!
mountconf["B3"] = {
    "nbaffles" : 1, # number of forebaffles
    "gsrad" : 600/2*i2m, # in meters
    "gsheight" : 200*i2m, # in meters
    "gssides" : 8,
    "fbrad": 48/2*i2m, # in meters
    "fbheight": 40*i2m+0.07, # in meters
    "winrad" : 0.69/2, # in meters
    "aptoffr" : 0., # in meters
    "drumangle" : 0, # Degrees
    "aptoffz" : 36.5*i2m, # in meters
    "dkoffx": 0.0,  # in meters
    "dkoffy": 0.0,  # in meters
    "eloffx": 0.0,  # in meters
    "eloffz" : 0.0, # in meters
}
