# External dependencies
import os
import sys
from contextlib import contextmanager

import healpy as hp
import numpy as np
import pandas as pd
import emcee

# Local dependencies
from xbmodeling.config import modelconf
from xbmodeling.pointing_model import beam_pointing_model


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

    theta, phi = hp.pixelfunc.lonlat2thetaphi(0, 90)

    ipix_disc = hp.query_disc(nside=hp.npix2nside(np.shape(groundmap)[0]), vec=hp.ang2vec(theta, phi),
                              radius=np.radians(90))
    groundmap[ipix_disc] = T

    return groundmap


def make_beam_map(params=modelconf["beamParams"], nside=modelconf["defaultResolution"]):
    """
    Makes a 2-D Gaussian beam map in healpix coordinates.

    :param params: list or ndarray shape (6,)
        Parameters which describe a two-dimensional Gaussian
    :param nside: int
        Resolution of the map. Should be a power of two and should match the resolution of the CMB maps
    :return beammap: ndarray shape(N,)
        2-D Gaussian in healpix coordinates
    """

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

    # make a two-dimensional gaussian
    beammap = params[0] * np.exp(
        -1 * (a * (x - params[1]) ** 2 + c * (y - params[2]) ** 2 + 2 * b * (x - params[1]) * (y - params[2])))
    return beammap


def make_composite_map(main, extended, nside=modelconf["defaultResolution"]):
    bm = make_beam_map(main, nside=nside) + make_beam_map(extended, nside=nside)
    return bm / np.sum(bm)


def convolve_maps(maps, doconv=True):
    # Convolve sky with beams using healpy's smoothing function.
    # Convert beammaps into window functions using anafast

    # Suppress healpy output
    with suppress_stdout():
        if doconv:
            # Temperature map has both CMB and ground
            maps["convmapT"] = hp.smoothing(
                (maps["cmbmapT"] + maps["groundmap"]).values,
                beam_window=hp.anafast((maps["beammapA"] - maps["beammapB"]).values)
            )

            # Polarization maps only have CMB
            maps["convmapQ"] = hp.smoothing(
                maps["cmbmapQ"].values,
                beam_window=hp.anafast((maps["beammapA"] + maps["beammapB"]).values)
            )

            maps["convmapU"] = hp.smoothing(
                maps["cmbmapU"].values,
                beam_window=hp.anafast((maps["beammapA"] + maps["beammapB"]).values)
            )

        else:
            maps["convmapT"] = maps["cmbmapT"]
            maps["convmapQ"] = maps["cmbmapQ"]
            maps["convmapU"] = maps["cmbmapU"]

    return maps


g2c = hp.Rotator(coord=['G', 'C']).rotate_map_pixel


@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


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

        self.cmb_file = cmb_file
        self.main_beam_params_A = main_beam_params_A
        self.main_beam_params_B = main_beam_params_B
        self.ext_beam_params = ext_beam_params
        self.nside = nside
        self.pixel = hp.nside2npix(nside)
        self.T = T

        # Initialize the CMB maps.
        cmbmap = make_cmb_map(cmb_file, nside_out=self.nside)
        cmbmap = np.array(g2c(cmbmap))

        # Keep each maps separated
        self.maps = pd.DataFrame({
            "beammapA": make_composite_map(main_beam_params_A, ext_beam_params, nside=self.nside),
            "beammapB": make_composite_map(main_beam_params_B, ext_beam_params, nside=self.nside),
            "groundmap": make_ground_template(T=self.T, nside=self.nside),
            "cmbmapT": cmbmap[0, :],
            "cmbmapQ": cmbmap[1, :],
            "cmbmapU": cmbmap[2, :],
        })

        # Convolve the beam maps and CMB maps
        # Are we simply initializing the model? If so, don't do the convolutions
        # just make the conv maps the CMB maps
        if any([not isinstance(obj, type(None)) for obj in [self.main_beam_params_A,
                                                            self.main_beam_params_B,
                                                            self.ext_beam_params]]):
            self.maps = convolve_maps(self.maps)
        else:
            self.maps = convolve_maps(self.maps, doconv=False)

        self.maplist = list(self.maps.keys())
        
        # Set default bounds on the uniform priors
        main_peak = (0, 2)
        main_center = (-7, 7)
        main_width = (0, 2)
        main_corr = (-1, 1)
        ext_peak = (0, 2)
        ext_center = (-30, 30)
        ext_width = (0, 50)
        ext_corr = (-1, 1)
        self.param_bounds = (main_peak, main_center, main_center, main_width, main_width, main_corr,
                             main_peak, main_center, main_center, main_width, main_width, main_corr,
                             ext_peak, ext_center, ext_center, ext_width, ext_width, ext_corr)

    def regen_model(self, mainA=None, mainB=None, extended=None, cmb_file=None, T=None):
        '''
        It's not super efficient to keep reloading CMB maps, so we'll want
        to just load the map once and only redo the convolution every time
        we update the beam map.
        '''

        # If any parameter set is changed. Change the object attribute.
        if not isinstance(mainA, type(None)):
            self.main_beam_params_A = mainA

        if not isinstance(mainB, type(None)):
            self.main_beam_params_B = mainB

        if not isinstance(extended, type(None)):
            self.ext_beam_params = extended

        self.maps["beammapA"] = make_composite_map(self.main_beam_params_A, self.ext_beam_params, nside=self.nside)
        self.maps["beammapB"] = make_composite_map(self.main_beam_params_B, self.ext_beam_params, nside=self.nside)

        # Initialize the CMB map
        if type(cmb_file) != type(None):
            self.cmb_file = cmb_file
            cmbmap = make_cmb_map(cmb_file, nside_out=self.nside)
            cmbmap = np.array(g2c(cmbmap))
            self.maps["cmbmapT"] = cmbmap[0, :]
            self.maps["cmbmapQ"] = cmbmap[1, :]
            self.maps["cmbmapU"] = cmbmap[2, :]

        # Add the ground template
        if type(T) != type(None):
            self.T = T
            self.maps["groundmap"] = (make_ground_template(T=self.T, nside=self.nside))

        # Convolve the beam map and CMB map
        # Are we simply initializing the model? If so, don't do the convolutions
        # just make the conv maps the CMB maps
        if any([not isinstance(obj, type(None)) for obj in [self.main_beam_params_A,
                                                            self.main_beam_params_B,
                                                            self.ext_beam_params]]):
            self.maps = convolve_maps(self.maps)
        else:
            self.maps = convolve_maps(self.maps, doconv=False)

        # return self

    def observe(self,
                tod,
                det_info,
                mainA=None,
                mainB=None,
                extended=None,
                cmb_file=None,
                T=None,
                extendedopt=modelconf["extendedOption"],
                mntstr="B3",
                showplot=False):

        if not isinstance(extended, type(None)) and not extendedopt == "custom":
            x = 2 * np.sin(det_info["r"].values[0] / 2) * np.cos(det_info["theta"].values[0]) * 180.0 / np.pi
            y = 2 * np.sin(det_info["r"].values[0] / 2) * np.sin(det_info["theta"].values[0]) * 180.0 / np.pi
            if extendedopt == "main":
                extended[1:3] = [0, 0]
            elif extendedopt == "boresight":
                extended[1:3] = [-x, -y]
            elif extendedopt == "buddy":
                extended[1:3] = [-2 * x, -2 * y]
            else:
                raise NameError("Extended Beam Option: " + extendedopt + " not found.")

        # Check to see if the input parameters are the same as the parameters we already have.
        # If they are, we've already done a convolution, so don't do another one.
        if self.check_change(mainA=mainA, mainB=mainB, extended=extended, cmb_file=cmb_file, T=T):
            self.regen_model(mainA=mainA, mainB=mainB, extended=extended, cmb_file=cmb_file, T=T)

        # Initialize variables
        samples = len(tod.index)
        # From data, output the on-sky pointing of the detector
        tod_pointing = beam_pointing_model(tod, det_info, mntstr)

        ipix = hp.pixelfunc.ang2pix(self.nside,
                                    tod_pointing["app_ra_0"].values,
                                    tod_pointing["app_dec_0"].values,
                                    lonlat=True)

        mapind = self.maps.iloc[ipix]

        tod_pointing["simdata"] = (mapind["convmapT"].values +
                                   mapind["convmapQ"].values * np.cos(2 * np.deg2rad(tod_pointing["pa_0"].values)) +
                                   mapind["convmapU"].values * np.sin(2 * np.deg2rad(tod_pointing["pa_0"].values))) * \
                                  det_info["ukpervolt"].values[0] * 1e6

        if showplot:
            Z = np.zeros(hp.nside2npix(self.nside))
            Z[ipix] = 300
            hp.mollview(self.maps["convmapT"].values + Z, coord="C", norm="hist")
            hp.graticule()

        return tod_pointing

    def plot(self, mapstr, coord="C", norm="hist", **kwargs):
        if mapstr == 'all':
            [hp.mollview(self.maps[ms].values, title=ms, coord=coord, norm=norm ** kwargs) for ms in self.maps.keys()]
        else:
            try:
                hp.mollview(self.maps[mapstr], title=mapstr, coord=coord, norm=norm, **kwargs, )
            except KeyError:
                print("The map ''" + mapstr + "'' does not exist!")
                return 0
        hp.graticule()
        return 1

    def check_change(self, mainA=None, mainB=None, extended=None, cmb_file=None, T=None):
        # If any parameter set is changed. Change the object attribute.
        checkchange = []

        # Check the beammap values
        if not isinstance(mainA, type(None)):
            checkchange.append((self.main_beam_params_A != mainA))

        if not isinstance(mainB, type(None)):
            checkchange.append((self.main_beam_params_B != mainB))

        if not isinstance(extended, type(None)):
            checkchange.append((self.ext_beam_params != extended))
        # Check the CMB map file
        if not isinstance(cmb_file, type(None)):
            checkchange.append((self.cmb_file != cmb_file))

        # check the ground template
        if not isinstance(T, type(None)):
            checkchange.append((self.T != T))

        return any(checkchange)

    def log_prior(self, theta, bounds=None):
        """
        Returns natural log of prior distribution.  All params have uniform priors.

        :theta: tuple of 18 floats 
            First 6 are amplitude, x_center, y_center, x_width, y_width, correlation for main beam A.
            Second 6 are the same parameters for main beam B
            Third 6 are the same parameters for the extended beam
        :return log_prior_value: float

        """
        if bounds is None:
            bounds = self.param_bounds
            
        # If within the param bounds, return 0.  Otherwise -inf
        inside_bounds = all([bounds[0] < param < bounds[1] for param, bounds in zip(theta,self.param_bounds)])

        if inside_bounds:
            return 0.0
        else:
            return -np.inf
        
    def log_likelihood(self, theta, tod, det_info, sigma):
        """
        Returns natural log of likelihood distribution.

        :theta: tuple of 18 floats 
            First 6 are amplitude, x_center, y_center, x_width, y_width, correlation for main beam A.
            Second 6 are the same parameters for main beam B
            Third 6 are the same parameters for the extended beam
        :tod: pandas dataframe
            Contains real observed values of polarization timestreams (field 'realdata')
            Also contains telescope motion and detector info 
        :det_info: pandas dataframe
            Contains detector pointing information.
        :sigma: single float, or array of floats w/ same length as tod['realdata']
            Estimate of uncertainty in real data.
        :return log_likelihood_value: float

        """
        
        observed = tod['realdata'].values
        
        # Now run the model with the given beam params to get simulated data
        mainA = theta[0:6]
        mainB = theta[6:12]
        extended = theta[12:18]
        tod = self.observe(tod, det_info,
                           mainA = mainA,
                           mainB = mainB,
                           extended = extended)
        predicted = tod['simdata'].values
        
        residual = (observed - predicted)**2/sigma**2
        prefactor = 1/np.sqrt(2 * np.pi * sigma**2)
        log_likelihood = np.sum( np.log(prefactor * np.exp(-residual/2)) )
        
        return log_likelihood
    
    def log_posterior(self, theta, tod, det_info, sigma):
        """
        Returns natural log of posterior distribution (prior * likelihood).

        :theta: tuple of 18 floats 
            First 6 are amplitude, x_center, y_center, x_width, y_width, correlation for main beam A.
            Second 6 are the same parameters for main beam B
            Third 6 are the same parameters for the extended beam
        :tod: pandas dataframe
            Contains real observed values of polarization timestreams (field 'realdata')
            Also contains telescope motion and detector info 
        :det_info: pandas dataframe
            Contains detector pointing information.
        :return log_posterior_value: float

        """
        
        log_prior_value = self.log_prior(theta)
        log_likelihood_value = self.log_likelihood(theta, tod, det_info, sigma)
        
        log_posterior_value = log_prior_value + log_likelihood_value
        
        return log_posterior_value
    
    def initialize_walkers(self, initial_guess, gaussian_ball_width, N_walkers, seed=None):
        """
        Returns starting positions of walkers for emcee.
        
        :initial_guess: tuple or array
            initial guesses corresponsing to each parameter in model
            First 6 are amplitude, x_center, y_center, x_width, y_width, correlation for main beam A.
            Second 6 are the same parameters for main beam B
            Third 6 are the same parameters for the extended beam
        :gaussian_ball_width: float
            Width of Gaussian ball determining walker starting position
        :N_walkers: int
            Number of walkers to be used in emcee
        :seed: int between 0 and 2**32-1
            For initializing pseudo-random number generator.  Only use for debugging.
        :return  starting_positions: N_walkers x N_params size array of floats

        """
        # Initialize the RNG.  If None, won't do anything
        np.random.seed(seed)
        # Starting positions are randomly distributed Gaussian ball around initial positions
        N_dim = len(initial_guess)
        gaussian_ball = gaussian_ball_width * np.random.randn(N_walkers, N_dim)
        starting_positions = (1 + gaussian_ball) * initial_guess
        
        return starting_positions
    
    def do_emcee_fit(self, 
                     tod, 
                     det_info, 
                     N_walkers,
                     N_steps,
                     initial_guess, 
                     gaussian_ball_width, 
                     seed = None, 
                     sigma = 1):
        """
        Do the model fit using emcee.
        :tod: pandas dataframe
            Contains real observed values of polarization timestreams (field 'realdata')
            Also contains telescope motion and detector info 
        :det_info: pandas dataframe
            Contains detector pointing information.
        :N_walkers: int
            Number of walkers to be used in emcee
        :N_steps: int
            Number of steps for each walker to take.
        :initial_guess: tuple or array
            initial guesses corresponsing to each parameter in model
            First 6 are amplitude, x_center, y_center, x_width, y_width, correlation for main beam A.
            Second 6 are the same parameters for main beam B
            Third 6 are the same parameters for the extended beam
        :gaussian_ball_width: float
            Width of Gaussian ball determining walker starting position
        :seed: int between 0 and 2**32-1
            For initializing pseudo-random number generator.  Only use for debugging.  Default None.
        :sigma: single float, or array of floats w/ same length as tod['realdata']
            Estimate of uncertainty in real data.  Default 1.
        
        :return fit_df: pandas dataframe
            contains traces for all parameters of model

        """
        # Walker starting positions
        starting_positions = self.initialize_walkers(initial_guess, gaussian_ball_width, N_walkers, seed)
        
        # Setup and run the sampler
        sampler = emcee.EnsembleSampler
        sampler = sampler(N_walkers, len(initial_guess), self.log_posterior, args=(tod, det_info, sigma))
        sampler.run_mcmc(starting_positions, N_steps)
        
        fit_df = pd.DataFrame(np.vstack(sampler.chain))
        fit_df.index = pd.MultiIndex.from_product([range(nwalkers), range(nsteps)], 
                                                  names=['walker', 'step'])
        fit_df.columns = ['mainA_amp','mainA_x','mainA_y','mainA_sigx','mainA_sigy','mainA_corr',
                          'mainB_amp','mainB_x','mainB_y','mainB_sigx','mainB_sigy','mainB_corr',
                          'ext_amp','ext_x','ext_y','ext_sigx','ext_sigy','ext_corr']
        
        return fit_df