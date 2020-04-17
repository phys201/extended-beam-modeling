from .config import modelconf
import healpy as hp
import scipy

def make_cmb_ground_map(filename):
    # Load CMB Map
    cmbmap = hp.read_map(filename, field=None, nest=False)

    # Determine where ground should be and set to a constant temperature.
    # T = modelconf["T"]

    return cmbmap

def make_beam_map(self, params):


    return# beammap



def convolve_maps(cmbmap,beammap):

    return

class GenModelMap:
    def __init__(self, beamparams=modelconf["BeamParams"]):
        self.params = beamparams

        self.main_beam_map = make_beam_map(*beamparams)
        self.cmbmap = make_cmb_ground_map(modelconf["cmbFile"])
        return


    def regen_model(self):
        '''
        It's not super efficient to keep reloading CMB maps, so we'll want
        to just load the map once and only redo the convolution every time
        we update the beam map.
        '''

        return

    def observe(self,data):
        '''
        Use the telescope pointing information to 'observe' the model map and return a timestream.
        :param data:
        :return:
        '''

        # Extract detector timestream by interpolating map with az/el timesteams


        return simdata