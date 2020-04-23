import numpy as np
import numbers
from functools import reduce
from xbmodeling.config import mountconf


'''
Note: A lot of this code is based-on or directly copied from some of my previous
work on raytracing.
source: https://github.com/jacornelison/shielding-raytracer
'''


def extract(cond, x):
    if isinstance(x, numbers.Number):
        return x
    else:
        return np.extract(cond, x)


class vec3():
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


def beam_pointing_model(az, el, dk, r, theta, pol=0, mntstr="keck"):
    mnt = mountconf[mntstr]
    # Start at the origin of a cartesian coordinate system
    # pos = origin, direction = Zhat, orientation = Xhat
    # and rotate/offset vectors as necessary to reflect
    # telescope boresight pointing and aperture position
    mountpos, mountdir, mountort = mount_transform(ORG, Z, X, az, el, dk, mnt)

    # Do an additional rotation to reflect the detector pointing
    detdir, detpol, detort, detort2 = detector_transform(mountdir, mountort, r, theta, pol)

    # Get the apparent azimuth, elevation, and parallactic angle:
    az_app, el_app, parall_angle = get_apparent_topocoords(detdir, detpol)

    return az_app, el_app, parall_angle


def get_apparent_topocoords(detdir, detpol):
    az_app = np.arctan2(detdir.dot(Y) * -1, detdir.dot(X)) * 180 / np.pi
    el_app = np.arcsin(detdir.dot(Z)) * 180 / np.pi

    # Calculate the parallactic angle of the the detector polarization
    # with respect to zenith
    ea = Z.cross(detdir).norm()
    eb = detdir.cross(ea)
    parall_angle = np.arctan2(detpol.dot(eb), detpol.dot(ea)) * 180 / np.pi

    return az_app % (360.), el_app % (360.), parall_angle % (360.)


def mount_transform(pos, dir, ort, az, el, dk, mnt):
    # Position

    # Avoid changing the input values
    outpos, outdir, outort = [pos, dir, ort]

    outpos = outpos + vec3(mnt["aptoffr"], 0., 0.)
    outpos = outpos.rotate(Z, dk + mnt["drumangle"] - 90)
    outpos = outpos + vec3(0., 0., mnt["aptoffz"])
    outpos = outpos.rotate(Y, 90 - el)
    outpos = outpos + vec3(0., 0., mnt["eloffz"])
    outpos = outpos.rotate(Z, -az)

    outdir = outdir.rotate(Z, dk + mnt["drumangle"] - 90).rotate(Y, 90 - el).rotate(Z, -az)
    outort = outort.rotate(Z, dk + mnt["drumangle"] - 90).rotate(Y, 90 - el).rotate(Z, -az)
    return outpos, outdir, outort


def detector_transform(indir, inort, r, theta, pol=0):
    # Parallel transport the pointing and orientation vectors in
    # the direction of the detector pointing WRT the boresight.
    inort2 = indir.cross(inort)
    outdir, outort, outort2 = [euler_rotate(obj, -theta, -r, theta, inort, inort2, indir)
                               for obj in [indir, inort, inort2]]

    # Rotate the detector orientation to the polarization response orientation.
    # This is important if we're dealing with polarization angles
    # outort = outort.rotate(outdir,pol)
    outpol = outort * np.cos(np.radians(pol)) - outort2 * np.sin(np.radians(pol))

    return outdir, outpol, outort, outort2


# Euler Rotation Matrix
def euler_rotate(V, a, b, g, e1=X, e2=Y, e3=Z):
    V = V.rotate(e3, g)
    XP = e1.rotate(e3, g)
    YP = e2.rotate(e3, g)
    ZP = e3.rotate(e3, g)

    V = V.rotate(YP, b)
    XPP = XP.rotate(YP, b)
    YPP = YP.rotate(YP, b)
    ZPP = ZP.rotate(YP, b)

    return V.rotate(ZPP, a)
