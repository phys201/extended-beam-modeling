from .config import modelconf

class GenModelMap:
    def __init__(self, params):
        self.params = params

        self.beam_map = make_beam_map(params)

        return

    def make_beam_map(self,params):

        return

    def make_cmb_ground_map(self, filename=modelconf["cmbFile"]):

        # Load CMB Map
        # cmbmap = load_cmb_map
        # Determine where ground should be and set to a constant temperature.
        #T = modelconf["T"]

        #self.cmbmap = cmbmap
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
        return simdata