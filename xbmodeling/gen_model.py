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


def make_cmb_map(
        filename=modelconf["cmbFile"],
        nside_out=modelconf["defaultResolution"]
):
    """

    Loads CMB fits file using healpy.

    Parameters
    ----------
    filename : str, optional
        Path and filename designating cmb map fits file.
    nside_out : int, optional
        Output resolution of the map. Should be a power of two. Maps
        typically come in nside=2048 or 1024.
    Returns
    -------
    cmbmap : ndarray shape(N, 1-3)
        Returns map in Healpix coordinates. shape(N,1) assumes T map,
        shape(N,2) assumes Q/U map, and shape(N,3) assumes I,Q,U)
    """

    # Load CMB Map
    cmbmap = hp.read_map(filename, field=None, nest=False)

    if nside_out != hp.npix2nside(np.shape(cmbmap)[1]):
        print("Current map NSIDE={0}. Converting to NSIDE={1}".format(
            hp.npix2nside(np.shape(cmbmap)[1]), nside_out))
        cmbmap = hp.pixelfunc.ud_grade(cmbmap, nside_out)

    return cmbmap


def make_ground_template(
        T=modelconf["groundTemperature"],
        nside=modelconf["defaultResolution"]
):
    """

    Makes a 2-D template of a constant temperature ground.

    Parameters
    ----------
    T : int or double, optional
         Ground template temperature.
    nside : int, optional
        Resolution of the map. Should be a power of two and should match
         the resolution of the CMB maps

    Returns
    -------
    groundmap : ndarray shape(N,)
        2-D ground template in Healpix Celestial coordinates.
    """

    groundmap = np.zeros(hp.nside2npix(nside))
    if type(T) == type(None):
        return groundmap

    # Healpix coordinates
    # The standard coordinates are the colatitude $\theta$, $0$ at the North Pole, $\pi/2$ at the equator and $\pi$ at
    # the South Pole and longitude $\phi$ between $0$ and $2\pi$ eastward, in a Mollview projection, $\phi=0$ is at the
    # center and increases eastward toward the left of the map.

    theta, phi = hp.pixelfunc.lonlat2thetaphi(0, 90)

    ipix_disc = hp.query_disc(
        nside=hp.npix2nside(np.shape(groundmap)[0]),
        vec=hp.ang2vec(theta, phi),
        radius=np.radians(90))
    groundmap[ipix_disc] = T

    return groundmap


def make_beam_map(params=None, nside=modelconf["defaultResolution"]):
    """
    Makes a 2-D Gaussian beam map in healpix coordinates.

    Parameters
    ----------
    params : list or ndarray shape (6,), optional
        Parameters which describe a two-dimensional Gaussian. Returns a
        2D pencil beam (delta function) by default.
    nside : int, optional
        Resolution of the map. Should be a power of two and should match
         the resolution of the CMB maps

    Returns
    -------
    beammap : ndarray shape(N,)
        2-D Gaussian in Healpix coordinates
    """

    # Allow a None value for params to return a delta function
    if isnone(params):
        # params = [1,0,0,1e-5,1e-5,1]
        beammap = np.zeros(hp.nside2npix(nside))
        beammap[0] = 1
        return beammap

    pixels = np.arange(hp.nside2npix(nside))
    theta, phi = hp.pixelfunc.pix2ang(nside, pixels)
    x = 2 * np.sin(theta / 2) * np.cos(phi) * 180.0 / np.pi
    y = 2 * np.sin(theta / 2) * np.sin(phi) * 180.0 / np.pi

    ang = params[5] * np.pi
    a = np.cos(ang) ** 2 / 2 / params[3] + np.sin(ang) ** 2 / 2 / \
        params[4]
    b = np.sin(2 * ang) / 4 * (1 / params[4] - 1 / params[3])
    c = np.sin(ang) ** 2 / 2 / params[3] + np.cos(ang) ** 2 / 2 / \
        params[4]

    # make a two-dimensional gaussian
    beammap = params[0] * np.exp(
        -1 * (a * (x - params[1]) ** 2 + c * (
                    y - params[2]) ** 2 + 2 * b * (x - params[1]) * (
                          y - params[2])))
    return beammap


def make_composite_map(
        main=None,
        extended=None, nside=modelconf["defaultResolution"]
):
    """

    Returns normalized beam map with a main and extended beam.

    Parameters
    ----------
    main, extended : list or ndarray shape (6,), optional
        Parameters which describe a two-dimensional Gaussian. Returns a
        2D pencil beam (delta function) by default.

    nside : int, optional
        Resolution of the map. Should be a power of two and should match
         the resolution of the CMB maps

    Returns
    -------
    beammap : ndarray shape(N,)
        2-D Gaussian in Healpix coordinates
    """

    # extended beam should be offset relative to the main beam.
    if not isnone([main, extended]):
        extended[1] += main[1]
        extended[2] += main[2]

    bm = make_beam_map(main, nside=nside) \
         + make_beam_map(extended, nside=nside)
    return bm / np.max(bm)


def convolve_maps(maps, doconv=True):
    """

    Convolve sky with beams using healpy's smoothing function. Beammaps
    are converted into window functions using healpy's anafast function.

    Parameters
    ----------
    maps : Pandas DataFrame
        Collection of CMB and beam maps in healpix coordinates. See
        the GenMapModel class for more information.

    doconv : bool, optional
        Choose whether to do the convolution or just pass the input maps
        as the convolution.

    Returns
    -------
    maps : Pandas DataFrame
        Returns maps data with extra "convmap[T,Q,U]" maps.
    """

    # Get pairsum and pairdiff beams, find and normalize their window functions
    pairsum = (maps["beammapA"] + maps["beammapB"]).values
    pairsum /= np.sum(pairsum)
    pairsum_windowfunc = hp.anafast(pairsum)
    pairsum_windowfunc /= pairsum_windowfunc[0]
    pairdiff = (maps["beammapA"] + maps["beammapB"]).values
    pairdiff /= np.sum(pairdiff)
    pairdiff_windowfunc = hp.anafast(pairdiff)
    pairdiff_windowfunc /= pairdiff_windowfunc[0]

    # Suppress healpy output
    with suppress_stdout():
        if doconv:
            # Temperature map has both CMB and ground
            maps["convmapT"] = hp.smoothing(
                (maps["cmbmapT"] + maps["groundmap"]).values,
                beam_window=pairdiff_windowfunc
            )

            # Polarization maps only have CMB
            maps["convmapQ"] = hp.smoothing(
                maps["cmbmapQ"].values,
                beam_window=pairsum_windowfunc
            )

            maps["convmapU"] = hp.smoothing(
                maps["cmbmapU"].values,
                beam_window=pairsum_windowfunc
            )

        else:
            maps["convmapT"] = maps["cmbmapT"]
            maps["convmapQ"] = maps["cmbmapQ"]
            maps["convmapU"] = maps["cmbmapU"]

    return maps


def isnone(obj):
    """
    Determines if objects are None. Useful for comparing things that
    can be None or iterables.

    Parameters
    ----------
    obj : object or list of objects

    Returns
    -------
    bool
    """
    if not isinstance(obj, list):
        obj = [obj]

    return any([isinstance(x, type(None)) for x in obj])


g2c = hp.Rotator(coord=['G', 'C']).rotate_map_pixel


@contextmanager
def suppress_stdout():
    """

    Temporarily suppress output of running code.

    Example use:

    print("Now you see me.")
    with suppress_stdout():
        print("Now you don't.")

    print("Now you see me again."

    """
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


class GenModelMap:
    """

    Class for instantiating a generative model representing the
    pair-difference response of two co-located orthogonally oriented
    polarimeters observing the Cosmic Microwave Background with some
    ground temperature.

    Parameters
    ----------
    tod : Pandas DataFrame
            Real detector timestreams with telescope pointing
            information
    det_info : Pandas DataFrame
        Detector information
    mntstr : str, optional
            Mount information to be retrieved from the config file
    cmb_file : str, optional
        Path and filename designating cmb map fits file.
    T : int or double, optional
        Ground template temperature.
    nside : int, optional
        Resolution of the map. Should be a power of two.
    main_beam_params_A, main_beam_params_B, ext_beam_params :
        list or ndarray shape (6,), optional
        Parameters which describe a two-dimensional Gaussian. Returns a
        2D pencil beam (delta function) by default.

    Attributes
    ----------
    maps : Pandas DataFrame
        Contains maps of the CMB, ground, beams, and convolutions
        therein in healpix Celestial coordinates.

    Methods
    -------
    observe(tod,det_info,mainA,mainB,extended)
        Create a realized pair-diff timestream.


    """

    def __init__(
            self,
            tod,
            det_info,
            mntstr="keck",
            cmb_file=modelconf["cmbFile"],
            T=modelconf["groundTemperature"],
            nside=modelconf["defaultResolution"],
            main_beam_params_A=None,
            main_beam_params_B=None,
            ext_beam_params=None,
    ):

        self.det_info = det_info
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
            "beammapA": make_composite_map(main_beam_params_A,
                                           ext_beam_params,
                                           nside=self.nside),
            "beammapB": make_composite_map(main_beam_params_B,
                                           ext_beam_params,
                                           nside=self.nside),
            "groundmap": make_ground_template(T=self.T,
                                              nside=self.nside),
            "cmbmapT": cmbmap[0, :],
            "cmbmapQ": cmbmap[1, :],
            "cmbmapU": cmbmap[2, :],
        })

        # Convolve the beam maps and CMB maps
        # Are we simply initializing the model? If so, don't do the convolutions
        # just make the conv maps the CMB maps
        if not isnone([self.main_beam_params_A,
                   self.main_beam_params_B,
                   self.ext_beam_params]):
            self.maps = convolve_maps(self.maps)
        else:
            self.maps = convolve_maps(self.maps, doconv=False)

        self.maplist = list(self.maps.keys())

        # From data, output the on-sky pointing of the detector
        self.tod_pointing = beam_pointing_model(tod, det_info, mntstr)

        self.ipix = hp.pixelfunc.ang2pix(
            self.nside,
            self.tod_pointing["app_ra_0"].values,
            self.tod_pointing["app_dec_0"].values,
            lonlat=True
        )

        # Make a map of the detector pointing
        coveragemap = np.zeros(hp.nside2npix(self.nside))
        coveragemap[self.ipix] = 1
        self.maps["coveragemap"] = coveragemap

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

    def regen_model(
            self,
            mainA=None,
            mainB=None,
            extended=None,
            cmb_file=None,
            T=None
    ):
        """

        Updates class attributes and maps, and convolves beam maps
        with CMB/ground templates

        Parameters
        ----------
        mainA,mainB,extended : list or ndarray shape (6,), optional
            Parameters which describe a two-dimensional Gaussian. Returns a
            2D pencil beam (delta function) by default.
        cmb_file : str, optional
            Path and filename designating cmb map fits file.
        T : int or double, optional
            Ground template temperature.

        """

        # If any parameter set is changed. Change the object attribute.
        if not isnone(mainA):
            self.main_beam_params_A = mainA

        if not isnone(mainB):
            self.main_beam_params_B = mainB

        if not isnone(extended):
            self.ext_beam_params = extended

        self.maps["beammapA"] = make_composite_map(
            self.main_beam_params_A, self.ext_beam_params,
            nside=self.nside)
        self.maps["beammapB"] = make_composite_map(
            self.main_beam_params_B, self.ext_beam_params,
            nside=self.nside)

        # Initialize the CMB map
        if not isnone(cmb_file):
            self.cmb_file = cmb_file
            cmbmap = make_cmb_map(cmb_file, nside_out=self.nside)
            cmbmap = np.array(g2c(cmbmap))
            self.maps["cmbmapT"] = cmbmap[0, :]
            self.maps["cmbmapQ"] = cmbmap[1, :]
            self.maps["cmbmapU"] = cmbmap[2, :]

        # Add the ground template
        if not isnone(T):
            self.T = T
            self.maps["groundmap"] = (
                make_ground_template(T=self.T, nside=self.nside))

        # Convolve the beam map and CMB map
        # Are we simply initializing the model? If so, don't do the convolutions
        # just make the conv maps the CMB maps
        if not isnone([self.main_beam_params_A,
                   self.main_beam_params_B,
                   self.ext_beam_params]):
            self.maps = convolve_maps(self.maps)
        else:
            self.maps = convolve_maps(self.maps, doconv=False)

        return self

    def observe(self,
                mainA=None,
                mainB=None,
                extended=None,
                cmb_file=None,
                T=None,
                extendedopt=modelconf["extendedOption"]):

        """

        Create a noise realized pair-diff timestream and adds it to
        the input DataFrame

        Parameters
        ----------
        mainA,mainB,extended : list or ndarray shape (6,), optional
            Parameters which describe a two-dimensional Gaussian. Returns a
            2D pencil beam (delta function) by default.
        cmb_file : str, optional
            Path and filename designating cmb map fits file.
        T : int or double, optional
            Ground template temperature.
        extendedopt : str, {"main", "buddy", "boresight", "custom"}
            Optional. Determines where our extended be would be
            located.
        showplot bool, optional
            Show where one the convmapT the telescope was pointing.
            Useful for troubleshooting the conversion from telescope
            pointing to healpix coordinates.

        Returns
        -------
        tod : Pandas DataFrame
            input tod DataFrame, but with an additional "simdata"
            column.
        """

        if not isnone(extended) \
                and not extendedopt == "custom":
            x = 2 * np.sin(self.det_info["r"].values[0] / 2) \
                * np.cos(self.det_info["theta"].values[0]) \
                * 180.0 / np.pi
            y = 2 * np.sin(self.det_info["r"].values[0] / 2) \
                * np.sin(self.det_info["theta"].values[0]) \
                * 180.0 / np.pi
            if extendedopt == "main":
                extended[1:3] = [0, 0]
            elif extendedopt == "boresight":
                extended[1:3] = [-x, -y]
            elif extendedopt == "buddy":
                extended[1:3] = [-2 * x, -2 * y]
            else:
                raise NameError("Extended Beam Option: "
                                + extendedopt
                                + " not found.")

        # Check to see if the input parameters are the same as the
        # parameters we already have. If they are, we've already done
        # a convolution, so don't do another one.
        if self.check_change(mainA=mainA, mainB=mainB,
                             extended=extended, cmb_file=cmb_file, T=T):
            self.regen_model(mainA=mainA, mainB=mainB,
                             extended=extended, cmb_file=cmb_file, T=T)

        mapind = self.maps.iloc[self.ipix]

        self.tod_pointing["simdata"] = (
            mapind["convmapT"].values
            + mapind["convmapQ"].values
            * np.cos(2 * np.deg2rad(self.tod_pointing["pa_0"].values))
            + mapind["convmapU"].values
            * np.sin(2 * np.deg2rad(self.tod_pointing["pa_0"].values)))\
            / self.det_info["ukpervolt"].values[0] * 1e6

        return self.tod_pointing

    def mapplot(self, mapstr=None, coord="C", norm="hist", **kwargs):
        """

        Plots a molliweide projection of a map using healpy.mollview

        Parameters
        ----------
        mapstr : str, {"mapname", "all"} or None
            Name of the map that should be plotted. Running with no
            arguments will list the available maps for plotting.

        kwargs
            Optional arguments passed to healpy.mollview
            Default coord="C", norm="hist"

        """

        if isnone(mapstr):
            print("Here's the available keys:")
            print(list(self.maps.keys()))
        elif mapstr == 'all':
            [hp.mollview(self.maps[ms].values, title=ms, coord=coord,
                         norm=norm, **kwargs) for ms in
             self.maps.keys()]
        else:
            try:
                hp.mollview(self.maps[mapstr], title=mapstr,
                            coord=coord, norm=norm, **kwargs)
                hp.graticule()
            except KeyError:
                print("The map ''" + mapstr + "'' does not exist!")
                print("Here's the available keys:")
                print(list(self.maps.keys()))
                return 0

    def check_change(self, mainA=None, mainB=None, extended=None,
                     cmb_file=None, T=None):
        """

        Checks to see if any parameters have changed from the last
        time convolution was run.

        Parameters
        ----------
        main_beam_params_A, main_beam_params_B, ext_beam_params :
            list or ndarray shape (6,), optional
            Parameters which describe a two-dimensional Gaussian. Returns a
            2D pencil beam (delta function) by default.
        cmb_file : str, optional
            Path and filename designating cmb map fits file.
        T : int or double, optional
            Ground template temperature.

        Returns
        -------
        bool
            Any of the input parameters have changed since last.
        """

        # If any parameter set is changed. Change the object attribute.
        checkchange = []

        # Check the beammap values
        if not isnone(mainA):
            checkchange.append((self.main_beam_params_A != mainA))

        if not isnone(mainB):
            checkchange.append((self.main_beam_params_B != mainB))

        if not isnone(extended):
            checkchange.append((self.ext_beam_params != extended))
        # Check the CMB map file
        if not isnone(cmb_file):
            checkchange.append((self.cmb_file != cmb_file))

        # check the ground template
        if not isnone(T):
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
        
    def log_likelihood(self, theta, sigma):
        """
        Returns natural log of likelihood distribution.

        :theta: tuple of 18 floats 
            First 6 are amplitude, x_center, y_center, x_width, y_width, correlation for main beam A.
            Second 6 are the same parameters for main beam B
            Third 6 are the same parameters for the extended beam
        :sigma: single float, or array of floats w/ same length as tod['realdata']
            Estimate of uncertainty in real data.
            
        :return log_likelihood_value: float

        """
        
        observed = self.tod_pointing['realdata'].values
        
        # Now run the model with the given beam params to get simulated data
        mainA = theta[0:6]
        mainB = theta[6:12]
        extended = theta[12:18]
        
        tod = self.observe(mainA = mainA,
                           mainB = mainB,
                           extended = extended)
        predicted = self.tod_pointing['simdata'].values
        
        residual = (observed - predicted)**2/sigma**2
        prefactor = 1/np.sqrt(2 * np.pi * sigma**2)
        log_likelihood = np.sum( np.log(prefactor * np.exp(-residual/2)) )
        
        return log_likelihood
    
    def log_posterior(self, theta, sigma):
        """
        Returns natural log of posterior distribution (prior * likelihood).

        :theta: tuple of 18 floats 
            First 6 are amplitude, x_center, y_center, x_width, y_width, correlation for main beam A.
            Second 6 are the same parameters for main beam B
            Third 6 are the same parameters for the extended beam

        :return log_posterior_value: float

        """
        if type(theta) is np.ndarray:
            theta = theta.tolist()
            
        log_prior_value = self.log_prior(theta)
        log_likelihood_value = self.log_likelihood(theta, sigma)
        
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
                     N_walkers,
                     N_steps,
                     initial_guess, 
                     gaussian_ball_width, 
                     seed = None, 
                     sigma = 1.0):
        """
        Do the model fit using emcee.
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
        sampler = sampler(N_walkers, len(initial_guess), self.log_posterior, args=[sigma])
        sampler.run_mcmc(starting_positions, N_steps)
        
        self.fit_df = pd.DataFrame(np.vstack(sampler.chain))
        self.fit_df.index = pd.MultiIndex.from_product([range(N_walkers), range(N_steps)], 
                                                  names=['walker', 'step'])
        self.fit_df.columns = ['mainA_amp','mainA_x','mainA_y','mainA_sigx','mainA_sigy','mainA_corr',
                          'mainB_amp','mainB_x','mainB_y','mainB_sigx','mainB_sigy','mainB_corr',
                          'ext_amp','ext_x','ext_y','ext_sigx','ext_sigy','ext_corr']
        
        return self.fit_df
