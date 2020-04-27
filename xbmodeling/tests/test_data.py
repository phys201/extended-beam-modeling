from unittest import TestCase
import xbmodeling as xb
import xbmodeling.gen_model as gm
from xbmodeling.config import modelconf
import healpy as hp
import pandas as pd

class TestSomethingGeneric(TestCase):
    def test_upper(self):
        self.assertEqual('foo'.upper(), 'FOO')
        
class TestGenModel(TestCase):
    def test_input_cmb_map_type(self):
        filename = modelconf['cmbFile']
        Nside = 64
        cmb_map_to_test = gm.make_cmb_map(filename, nside_out=Nside)
        # Should be 3 maps -- T,Q,U
        self.assertEqual(hp.maptype(cmb_map_to_test),3)
        
    def test_input_cmb_map_nside(self):
        filename = modelconf['cmbFile']
        Nside_test = 64
        cmb_map_to_test = gm.make_cmb_map(filename, nside_out=Nside_test)
        Nside_output = hp.get_nside(cmb_map_to_test)
        self.assertEqual(Nside_test, Nside_output)

    def test_ground_temperature(self):
        Nside = 64
        ground_T_test = 150
        groundmap_test = gm.make_ground_template(T=ground_T_test, nside=Nside)
        # Check temp at some point on ground (lat > 0, any long)
        output_ground_T_ground = hp.get_interp_val(groundmap_test,3,32,lonlat=True)
        # Check temp at some point on sky (lat < 0, any long)
        output_ground_T_sky = hp.get_interp_val(groundmap_test,14,-32,lonlat=True)
        self.assertEqual(round(output_ground_T_ground,5), round(ground_T_test,5))
        self.assertEqual(round(output_ground_T_sky,5), round(0,5))
        
    def test_map_sizes(self):
        Nside = 64
        filename = modelconf['cmbFile']
        cmb_map_to_test = gm.make_cmb_map(filename, nside_out=Nside)
        groundmap_test = gm.make_ground_template(nside=Nside)
        beammap_test = gm.make_beam_map(nside=Nside)
        # All maps should be same size
        self.assertEqual(len(cmb_map_to_test[0,:]), len(groundmap_test))
        self.assertEqual(len(cmb_map_to_test[0,:]), len(beammap_test))
        self.assertEqual(len(beammap_test), len(groundmap_test))
        
    def test_beam_relative_amplitudes(self):
        Nside = 64
        # Keep beams at 0,0 otherwise have to worry about x,y <-> theta, phi conversion
        params_1 = [1, 0, 0, 0.1, 0.1, 1]
        params_2 = [2, 0, 0, 0.1, 0.1, 1]
        beammap_1 = gm.make_beam_map(params=params_1,nside=Nside)
        beammap_2 = gm.make_beam_map(params=params_2,nside=Nside)
        # Compare peaks near center
        peak_1 = hp.get_interp_val(beammap_1,0,0)
        peak_2 = hp.get_interp_val(beammap_2,0,0)
        self.assertTrue(peak_2 > peak_1)
        
    def test_convolution(self):
        # CMB map
        filename = modelconf['cmbFile']
        Nside = 64
        cmb_map_to_test = gm.make_cmb_map(filename, nside_out=Nside)
        # Ground 
        groundmap_test = gm.make_ground_template(T=301, nside=Nside)
        # Beam map
        params_main = [1, 0, 0, 0.1, 0.1, 1]
        params_ext = [0.05, 3, 3, 1, 1, 0]
        beammap_A = gm.make_composite_map(params_main, params_ext, nside=Nside)
        beammap_B = gm.make_composite_map(params_main, params_ext, nside=Nside)
        # Put them in ddataframe that convolve_maps likes
        maps = pd.DataFrame({
            "beammapA": beammap_A,
            "beammapB": beammap_B,
            "groundmap": groundmap_test,
            "cmbmapT": cmb_map_to_test[0, :],
            "cmbmapQ": cmb_map_to_test[1, :],
            "cmbmapU": cmb_map_to_test[2, :],
        })
        # Convolve
        maps = gm.convolve_maps(maps)
        self.assertEqual(len(maps["convmapT"]), len(maps["cmbmapT"]))
        
        