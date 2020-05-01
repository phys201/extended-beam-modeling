import numpy as np
import numbers
from functools import reduce
from xbmodeling.config import mountconf
import pandas as pd
from xbmodeling.azel2radec import azel2radec

'''Note: A lot of this code is based-on or directly copied from some 
of my previous work on raytracing. source: 
https://github.com/jacornelison/shielding-raytracer '''


def extract(cond, x):
    if isinstance(x, numbers.Number):
        return x
    else:
        return np.extract(cond, x)


class vec3():
    """
    Class representing 3-vectors for easier manipulation in the
    pointing model code. In addition to normal vector operations,
    this class has the notable ability to perform arbitrary vector
    rotations.

    """
    def __init__(self, x, y, z):
        (self.x, self.y, self.z) = (x, y, z)

    def __mul__(self, other):
        return vec3(self.x * other, self.y * other, self.z * other)

    def __add__(self, other):
        return vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        return vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def dot(self, other):
        return (self.x * other.x) + (self.y * other.y) + (self.z * other.z)

    def cross(self, other):
        x = self.y * other.z - self.z * other.y
        y = self.z * other.x - self.x * other.z
        z = self.x * other.y - self.y * other.x
        return vec3(x, y, z)

    def dist(self):
        return np.sqrt(self.dot(self))

    def __abs__(self):
        return self.dot(self)

    def norm(self):
        mag = np.sqrt(abs(self))
        return self * (1.0 / np.where(mag == 0, 1, mag))

    def components(self):
        return (self.x, self.y, self.z)

    def extract(self, cond):
        return vec3(extract(cond, self.x),
                    extract(cond, self.y),
                    extract(cond, self.z))

    def place(self, cond):
        r = vec3(np.zeros(cond.shape), np.zeros(cond.shape), np.zeros(cond.shape))
        np.place(r.x, cond, self.x)
        np.place(r.y, cond, self.y)
        np.place(r.z, cond, self.z)
        return r

    def rotate(self, k, th, degrees=True):
        # k is the rotation axis vector
        # th is the angle in radians
        # V cos(th) + (KxV) sin(th) + K (K.V)(1-cos(th))
        if degrees:
            th = np.radians(th)

        k.norm()
        return self * np.cos(th) + k.cross(self) * np.sin(th) + k * k.dot(self) * (1 - np.cos(th))


# Define coordinate system
X = vec3(1., 0., 0.)
Y = vec3(0., 1., 0.)
Z = vec3(0., 0., 1.)
ORG = vec3(0., 0., 0.)  # Origin


def beam_pointing_model(tod, det_info, mntstr="keck"):
    """

    Parameters
    ----------
    tod : Pandas DataFrame
            Real detector timestreams with telescope pointing
            information
    det_info : Pandas DataFrame
            Detector information
    mntstr : str, optional
            Mount information to be retrieved from the config file

    Returns
    -------
    tod : Pandas DataFrame
        input tod DataFrame, but with additional apparent az/el and
        parallactic angle columns.
    """
    mnt = mountconf[mntstr]
    app_celest_pointing = pd.DataFrame(index=tod.index)
    for detind in range(len(det_info.index)):
        # Start at the origin of a cartesian coordinate system
        # pos = origin, direction = Zhat, orientation = Xhat
        # and rotate/offset vectors as necessary to reflect
        # telescope boresight pointing and aperture position
        mountpos, mountdir, mountort = __mount_transform(ORG, Z, X,
                                                        tod, det_info.iloc[detind], mnt)

        # Do an additional rotation to reflect the detector pointing
        detdir, detpol, detort, detort2 = __detector_transform(
            mountdir, mountort, det_info.iloc[detind])

        # Get the apparent azimuth, elevation, and parallactic angle:
        app_topocoords = __get_apparent_topocoords(detdir, detpol,
                                                  detind)

        # Get apparent RA/DEC
        app_celest_pointing = pd.concat(
            [app_celest_pointing, __get_apparent_celestcoords(tod,
                                                             app_topocoords, mnt, detind)],
            axis=1)

    return pd.concat([tod, app_celest_pointing], axis=1)


def __get_apparent_celestcoords(tod, app_topocoords, mnt, detind):
    # Convert apparent az/el to ra/dec.
    ra, dec = azel2radec(tod["time_utc"] + 2400000.5,
                         np.deg2rad(app_topocoords["az_app"]),
                         np.deg2rad(app_topocoords["el_app"]),
                         np.deg2rad(mnt["sitelat"]), mnt["sitelon"])

    return pd.DataFrame({
        "app_ra_" + str(detind): np.rad2deg(ra),
        "app_dec_" + str(detind): np.rad2deg(dec),
        "pa_" + str(detind): app_topocoords["pa"].values
    }, index=tod.index)


def __get_apparent_topocoords(detdir, detpol, detind):
    # Calculate the parallactic angle of the the detector polarization
    # with respect to zenith
    ea = Z.cross(detdir).norm()
    eb = detdir.cross(ea)

    return pd.DataFrame({
        "az_app": (np.arctan2(detdir.dot(Y) * -1, detdir.dot(X)) * 180 / np.pi) % 360.,
        "el_app": (np.arcsin(detdir.dot(Z)) * 180 / np.pi) % 360.,
        "pa": (np.arctan2(detpol.dot(eb), detpol.dot(ea)) * 180 / np.pi) % 360.
    })


def __mount_transform(pos, dir, ort, tod, det_info, mnt):
    # Get the physical location, pointing and orientation of an
    # experiment represented by 3 3-vectors.
    outpos = pos + vec3(mnt["aptoffr"], 0., 0.)
    outpos = outpos.rotate(Z, tod['tel_hor_dk'] + det_info["drumangle"] - 90)
    outpos = outpos + vec3(0., 0., mnt["aptoffz"])
    outpos = outpos.rotate(Y, 90 - tod['tel_hor_el'])
    outpos = outpos + vec3(0., 0., mnt["eloffz"])
    outpos = outpos.rotate(Z, -tod['tel_hor_az'])

    outdir = dir.rotate(Z, tod['tel_hor_dk'] + det_info["drumangle"] - 90) \
        .rotate(Y, 90 - tod['tel_hor_el']) \
        .rotate(Z, -tod['tel_hor_az'])
    outort = ort.rotate(Z, tod['tel_hor_dk'] + det_info["drumangle"] - 90) \
        .rotate(Y, 90 - tod['tel_hor_el']) \
        .rotate(Z, -tod['tel_hor_az'])

    return outpos, outdir, outort


def __detector_transform(indir, inort, det_info):
    # Parallel transport the pointing and orientation vectors in
    # the direction of the detector pointing WRT the boresight.
    inort2 = indir.cross(inort)

    outdir, outort, outort2 = [
    __euler_rotate(obj, -det_info["theta"], -det_info["r"],
                   det_info["theta"], inort, inort2, indir)
        for obj in [indir, inort, inort2]]

    # Rotate the detector orientation to the polarization response
    # orientation. This is important if we're dealing with
    # polarization angles outort = outort.rotate(outdir,pol) outpol =
    # outort * np.cos(np.radians(det_info["phi"])) - outort2 *
    # np.sin(np.radians(det_info["phi"]))
    outpol = outort

    return outdir, outpol, outort, outort2


# Euler Rotation Matrix
def __euler_rotate(V, a, b, g, e1=X, e2=Y, e3=Z):
    V = V.rotate(e3, g)
    XP = e1.rotate(e3, g)
    YP = e2.rotate(e3, g)
    ZP = e3.rotate(e3, g)

    V = V.rotate(YP, b)
    XPP = XP.rotate(YP, b)
    YPP = YP.rotate(YP, b)
    ZPP = ZP.rotate(YP, b)

    return V.rotate(ZPP, a)
