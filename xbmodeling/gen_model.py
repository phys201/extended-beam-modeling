# External dependencies
import os
import sys
from contextlib import contextmanager

import healpy as hp
import numpy as np
import pandas as pd

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
        print("Current map NSIDE={0}. Converting to NSIDE={1}".format(hp.npix2nside(np.shape(cmbmap)[1]), nside_out))
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

    ipix_disc = hp.query_disc(nside=hp.npix2nside(np.shape(groundmap)[0]), vec=hp.ang2vec(theta, phi),
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
    if isinstance(params, type(None)):
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
    bm = make_beam_map(main, nside=nside) \
         + make_beam_map(extended, nside=nside)
    return bm / np.sum(bm)


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

    # Suppress healpy output
    with suppress_stdout():
        if doconv:
            # Temperature map has both CMB and ground
            maps["convmapT"] = hp.smoothing(
                (maps["cmbmapT"] + maps["groundmap"]).values,
                beam_window=hp.anafast((maps["beammapA"]
                                        - maps["beammapB"]).values)
            )

            # Polarization maps only have CMB
            maps["convmapQ"] = hp.smoothing(
                maps["cmbmapQ"].values,
                beam_window=hp.anafast((maps["beammapA"]
                                        + maps["beammapB"]).values)
            )

            maps["convmapU"] = hp.smoothing(
                maps["cmbmapU"].values,
                beam_window=hp.anafast((maps["beammapA"]
                                        + maps["beammapB"]).values)
            )

        else:
            maps["convmapT"] = maps["cmbmapT"]
            maps["convmapQ"] = maps["cmbmapQ"]
            maps["convmapU"] = maps["cmbmapU"]

    return maps


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
                mntstr="keck",
                showplot=False):

        """

        Create a noise realized pair-diff timestream and adds it to
        the input DataFrame

        Parameters
        ----------
        tod : Pandas DataFrame
            Real detector timestreams with telescope pointing
            information
        det_info : Pandas DataFrame
            Detector information
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
        mntstr : str, optional
            Mount information to be retrieved from the config file
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

        if not isinstance(extended, type(None)) \
                and not extendedopt == "custom":
            x = 2 * np.sin(det_info["r"].values[0] / 2) \
                * np.cos(det_info["theta"].values[0]) \
                * 180.0 / np.pi
            y = 2 * np.sin(det_info["r"].values[0] / 2) \
                * np.sin(det_info["theta"].values[0]) \
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

        # From data, output the on-sky pointing of the detector
        tod_pointing = beam_pointing_model(tod, det_info, mntstr)

        ipix = hp.pixelfunc.ang2pix(self.nside,
                                    tod_pointing["app_ra_0"].values,
                                    tod_pointing["app_dec_0"].values,
                                    lonlat=True)

        mapind = self.maps.iloc[ipix]

        tod_pointing["simdata"] = (
            mapind["convmapT"].values
            + mapind["convmapQ"].values
            * np.cos(2 * np.deg2rad(tod_pointing["pa_0"].values))
            + mapind["convmapU"].values
            * np.sin(2 * np.deg2rad(tod_pointing["pa_0"].values))) \
            * det_info["ukpervolt"].values[0] * 1e6

        if showplot:
            Z = np.zeros(hp.nside2npix(self.nside))
            Z[ipix] = 300
            hp.mollview(self.maps["convmapT"].values + Z, coord="C", norm="hist")
            hp.graticule()

        return tod_pointing

    def plot(self, mapstr=None, coord="C", norm="hist", **kwargs):
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

        if isinstance(mapstr,type(None)):
            print("Here's the available keys:")
            print(list(self.maps.keys()))
        elif mapstr == 'all':
            [hp.mollview(self.maps[ms].values, title=ms, coord=coord,
                         norm=norm, **kwargs) for ms in self.maps.keys()]
        else:
            try:
                hp.mollview(self.maps[mapstr], title=mapstr,
                            coord=coord, norm=norm, **kwargs)
            except KeyError:
                print("The map ''" + mapstr + "'' does not exist!")
                print("Here's the available keys:")
                print(list(self.maps.keys()))
                return 0
        hp.graticule()

    def check_change(self, mainA=None, mainB=None, extended=None, cmb_file=None, T=None):
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
