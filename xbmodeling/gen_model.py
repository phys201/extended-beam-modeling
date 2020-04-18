from .config import modelconf
import healpy as hp
import numpy as np


def eq2gal(ra,dec):
    ag = np.radians(192.85948)
    dg = np.radians(27.12828)
    lncp = np.radians(122.93192 - 90.)

    glat = np.arcsin(np.sin(dg)*np.sin(dec) + np.cos(dg)*np.cos(dec)*np.cos(ra-ag))
    glong = np.arcsin(np.cos(dec)*np.sin(ra-ag) / np.cos(glat)) - lncp

    return glat, glong

def make_cmb_ground_map(filename,T = modelconf["groundTemperature"]):
    # Load CMB Map
    cmbmap = hp.read_map(filename, field=None, nest=False)

    if type(T)==type(None):
        return cmbmap

    # The standard coordinates are the colatitude $\theta$, $0$ at the North Pole, $\pi/2$ at the equator and $\pi$ at
    # the South Pole and longitude $\phi$ between $0$ and $2\pi$ eastward, in a Mollview projection, $\phi=0$ is at the
    # center and increases eastward toward the left of the map.
        
    # Determine where ground should be and set to a constant temperature.
    b, l = eq2gal(0,0)
    theta = (b+np.pi)%(2*np.pi)/2
    phi = l%(2*np.pi)
    ipix_disc = hp.query_disc(nside=hp.npix2nside(np.shape(cmbmap)[1]),vec=hp.ang2vec(theta, phi), radius=np.radians(90))
    cmbmap[:,ipix_disc] = T

    return cmbmap

def make_beam_map(params=modelconf["beamParams"], nsides=1024):
    pixels = np.arange(hp.nside2npix(nsides))
    theta, phi = hp.pixelfunc.pix2ang(nsides,pixels)
    x = 2*np.sin(theta/2)*np.cos(phi)*180.0/np.pi
    y = 2*np.sin(theta/2)*np.sin(phi)*180.0/np.pi

    ang = params[5]*np.pi
    a = np.cos(ang) ** 2 / 2 / params[3] + np.sin(ang) ** 2 / 2 / params[4]
    b = np.sin(2*ang)/4*(1/params[4]-1/params[3])
    c = np.sin(ang) ** 2 / 2 / params[3] + np.cos(ang) ** 2 / 2 / params[4]
    beammap = params[0]*np.exp(-1*(a*(x-params[1])**2+c*(y-params[2])**2+2*b*(x-params[1])*(y-params[2])))

    return beammap


# def convolve_maps(cmbmap,beammap):
#
#     return
#
# class GenModelMap:
#     def __init__(self, beamparams=modelconf["BeamParams"]):
#         self.params = beamparams
#
#         self.main_beam_map = make_beam_map(*beamparams)
#         self.cmbmap = make_cmb_ground_map(modelconf["cmbFile"])
#         return
#
#
#     def regen_model(self):
#         '''
#         It's not super efficient to keep reloading CMB maps, so we'll want
#         to just load the map once and only redo the convolution every time
#         we update the beam map.
#         '''
#
#         return
#
#     def observe(self,data):
#         '''
#         Use the telescope pointing information to 'observe' the model map and return a timestream.
#         :param data:
#         :return:
#         '''
#
#         # Extract detector timestream by interpolating map with az/el timesteams
#
#
#         return simdata