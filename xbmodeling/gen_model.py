# External dependencies
import healpy as hp
import numpy as np

# Local dependencies
from xbmodeling.config import *
from xbmodeling.pointing_model import *
from xbmodeling.azel2radec import *


def coord2thetaphi(coords):
    theta = ((coords[0] + np.pi/2) % (np.pi))
    phi = coords[1] % 2 * np.pi
    return (theta, phi)


def make_cmb_map(filename, nside_out=modelconf["defaultResolution"]):
    # Load CMB Map
    cmbmap = hp.read_map(filename, field=None, nest=False)

    if nside_out != hp.npix2nside(np.shape(cmbmap)[1]):
        print("Current map NSIDE={0}. Converting to NSIDE={1}".format(hp.npix2nside(np.shape(cmbmap)[1]), nside_out))
        cmbmap = hp.pixelfunc.ud_grade(cmbmap, nside_out)

    return cmbmap


def make_ground_template(T=modelconf["groundTemperature"], nside=modelconf["defaultResolution"]):
    groundmap = np.zeros(hp.nside2npix(nside))
    if type(T) == type(None):
        return groundmap

    # Healpix coordinates
    # The standard coordinates are the colatitude $\theta$, $0$ at the North Pole, $\pi/2$ at the equator and $\pi$ at
    # the South Pole and longitude $\phi$ between $0$ and $2\pi$ eastward, in a Mollview projection, $\phi=0$ is at the
    # center and increases eastward toward the left of the map.

    # Planck Maps:
    # Aligned in Galactic coordinates.
    # healpix theta = 0 at b = pi and theta = pi at b = -pi

    # Determine where ground should be and set to a constant temperature.
    #rotate = hp.rotator.Rotator(coord=['C', 'G'],deg=False)
    #gal_theta, gal_phi = rotate(coord2thetaphi([np.pi, 0]))
    theta, phi = hp.pixelfunc.lonlat2thetaphi(0,90)

    ipix_disc = hp.query_disc(nside=hp.npix2nside(np.shape(groundmap)[0]), vec=hp.ang2vec(theta, phi),
                              radius=np.radians(90))
    groundmap[ipix_disc] = T

    return groundmap


def make_beam_map(params=modelconf["beamParams"], nside=modelconf["defaultResolution"]):
    # Allow a None value for params to return a delta function
    if type(params) == type(None):
        # params = [1,0,0,1e-5,1e-5,1]
        beammap = np.zeros(hp.nside2npix(nside))
        beammap[0] = 1
        return beammap

    pixels = np.arange(hp.nside2npix(nside))
    theta, phi = hp.pixelfunc.pix2ang(nside, pixels)
    x = 2 * np.sin(theta / 2) * np.cos(phi) * 180.0 / np.pi
    y = 2 * np.sin(theta / 2) * np.sin(phi) * 180.0 / np.pi

    ang = params[5] * np.pi
    a = np.cos(ang) ** 2 / 2 / params[3] + np.sin(ang) ** 2 / 2 / params[4]
    b = np.sin(2 * ang) / 4 * (1 / params[4] - 1 / params[3])
    c = np.sin(ang) ** 2 / 2 / params[3] + np.cos(ang) ** 2 / 2 / params[4]

    # Make a Gaussian centered on zero and then rotate the whole map?
    beammap = params[0] * np.exp(
        -1 * (a * (x - params[1]) ** 2 + c * (y - params[2]) ** 2 + 2 * b * (x - params[1]) * (y - params[2])))
    return beammap / np.max(beammap)


def make_composite_map(mainA, mainB, extended, nside=modelconf["defaultResolution"]):
    return make_beam_map(mainA, nside=nside) + \
           make_beam_map(mainB, nside=nside) + \
           make_beam_map(extended, nside=nside)


def convolve_maps(cmbmap, beammap):
    # Make the beam window function from the beam map
    # This takes a REALLY long time if the map is large.
    Bl = hp.anafast(beammap)

    # Filter the CMB map using a custom window function
    conv_map = hp.smoothing(cmbmap, beam_window=Bl / np.max(Bl))
    return conv_map


g2c = hp.Rotator(coord=['G','C']).rotate_map_pixel


class GenModelMap:
    def __init__(
            self,
            cmb_file=modelconf["cmbFile"],
            T=modelconf["groundTemperature"],
            nside=modelconf["defaultResolution"],
            main_beam_params_A=None,
            main_beam_params_B=None,
            ext_beam_params=None,
    ):

        self.main_beam_params_A = main_beam_params_A
        self.main_beam_params_B = main_beam_params_B
        self.ext_beam_params = ext_beam_params
        self.nside = nside
        self.T = T
        self.maps = ['cmbmap', 'groundmap', 'beammap', 'convmap']

        self.beammap = make_composite_map(
            self.main_beam_params_A,
            self.main_beam_params_B,
            self.ext_beam_params,
            nside=self.nside,
        )

        # Add the ground template
        self.groundmap = (make_ground_template(T=self.T, nside=self.nside))

        # Initialize the CMB map
        self.cmbmap = make_cmb_map(cmb_file, nside_out=self.nside)
        self.cmbmap = g2c(self.cmbmap[0,:])

        # Convolve the beam map and CMB map
        self.convmap = convolve_maps(self.cmbmap + self.groundmap, self.beammap)

    def regen_model(self, mainA=None, mainB=None, extended=None, cmb_file=None, T=None):
        '''
        It's not super efficient to keep reloading CMB maps, so we'll want
        to just load the map once and only redo the convolution every time
        we update the beam map.
        '''

        # If any parameter set is changed. Change the object attribute.
        if type(mainA) != type(None):
            self.main_beam_params_A = mainA

        if type(mainB) != type(None):
            self.main_beam_params_B = mainB

        if type(extended) != type(None):
            self.ext_beam_params = extended

        self.beammap = make_composite_map(
            self.main_beam_params_A,
            self.main_beam_params_B,
            self.ext_beam_params,
            nside=self.nside
        )

        # Initialize the CMB map
        if type(cmb_file) != type(None):
            self.cmbmap = make_cmb_map(cmb_file, nside_out=self.nside)
            self.cmbmap = g2c(self.cmbmap[0,:])

        # Add the ground template
        if type(T) != type(None):
            self.T = T
            self.groundmap = (make_ground_template(T=self.T, nside=self.nside))

        # Convolve the beam map and CMB map
        self.convmap = convolve_maps(self.cmbmap + self.groundmap, self.beammap)

    def observe(self, az, el, dk, mjd, r=0, theta=0, pol=0, lat=-89.932, long=350.9, mntstr="B3",showplot=False):

        # MAPO Site location: 314°12'36.7''E, 89°59'35.4''S (Source: JPL Horizons)
        # Extract detector timestream by interpolating map with az/el timesteams

        # From data, output the on-sky pointiing of the detector
        az_app, el_app, parall_angle = beam_pointing_model(az, el, dk, r, theta, pol, mntstr)
        ra, dec = azel2radec(mjd + 2400000.5, np.radians(az_app), np.radians(el_app), np.radians(lat), long)

        Z = np.zeros(hp.nside2npix(self.nside))
        simdata = np.zeros(np.shape(az))+np.nan
        # Slow loop
        for dataind in np.arange(np.size(az)):
            rai, deci = ra[dataind], dec[dataind]
            vec = hp.ang2vec(rai*180/np.pi, deci*180/np.pi, lonlat=True)
            ipix_disc = hp.query_disc(nside=self.nside, vec=vec, radius=np.radians(1e-3),inclusive=True)
            simdata[dataind] = np.mean(self.cmbmap[ipix_disc])
            Z[ipix_disc] = 300

        if showplot:
            hp.mollview(self.convmap+Z)

        return simdata

    def plot(self, mapstr='all', **kwargs):
        if mapstr == 'all':
            [hp.mollview(self.__dict__[ms], title=ms, **kwargs) for ms in self.maps]
        else:
            try:
                hp.mollview(self.__dict__[mapstr], title=mapstr, **kwargs, )
            except KeyError:
                print("The map ''" + mapstr + "'' does not exist!")
