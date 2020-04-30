"""azel2radec converts from horizon coordinates (azimuth and
elevation) to celestial coordinates (right ascension and declination).
It provides 3 methods:
jdcnv      : converts a Gregorian time to a Julian Date
ct2lst     : converts Civil Time to Greenwich Sidereal Time
azel2radec : converts from horizon to celestial coordinates
The azel2radec module also provides a useful constant, second_jul, which is the
conversion of a local second to Julian time.
The azel2radec module is based on code available at
http://idlastro.gsfc.nasa.gov/
and
http://cosmology.berkeley.edu/group/
but does NOT replicate all behavior exactly. The numerical results will be the
same, but the input parameters and outputs may differ.
Viktor Roytman 2012
"""

import numexpr
import numpy

# Define a local second in Julian time - useful for duration calculation
second_jul = 0.1 / 8640

# Define pi - useful for evaluation expressions
pi = numpy.pi


def jdcnv(year, month, day, hour):
    """Convert Gregorian time (UTC) to Julian Date

    Input:
    year  = year (scalar int)
    month = month 1-12 (scalar int)
    day   = day 1-31 (scalar int)
    hour  = fractional hour (scalar double)

    Output:
    julian = Julian Date (scalar double)

    Original IDL source available at:
    http://idlastro.gsfc.nasa.gov/ftp/pro/astro/jdcnv.pro

    Revision history of IDL source:
    Converted to IDL from Don Yeomans Comet Ephemeris Generator,
    B. Pfarr, STX, 6/15/88
    Converted to IDL V5.0   W. Landsman   September 1997
       Added checks on valid month, day ranges W. Landsman July 2008
    """

    year = year
    month = month
    day = day
    # account for leap years
    leap = (month - 14) / 12.0

    julian = day - 32075 + ((1461) * (year + 4800 + leap) / 4.0) \
             + ((367) * (month - 2 - leap * 12) / 12.0) \
             - (3 * (((year + 4900 + leap) / 100.0)) / 4.0) \
             + (hour / 24.0) - 0.5

    return julian


def ct2lst(julian_date):
    """Convert Civil Time (as Julian date) to Greenwich Sidereal Time

    Input:
    julian_date = Julian date (scalar or vector double)

    Output:
    gst = Greenwich Sidereal Time (scalar or vector double)

    The constants used in ct2lst come from Astronomical Algorithms by Jean
    Meeus, p. 84 (Eq. 11-4).

    Original IDL source available at:
    http://idlastro.gsfc.nasa.gov/ftp/pro/astro/ct2lst.pro

    Revision history of IDL source:
    Adapted from the FORTRAN program GETSD by Michael R. Greason, STX,
              27 October 1988.
    Use IAU 1984 constants Wayne Landsman, HSTX, April 1995, results
              differ by about 0.1 seconds
    Longitudes measured *east* of Greenwich   W. Landsman    December 1998
    Time zone now measure positive East of Greenwich W. Landsman July 2008
    Remove debugging print statement  W. Landsman April 2009
    """

    c1 = 280.46061837
    c2 = 360.98564736629
    c3 = 0.000387933
    c4 = 38710000.0
    t0 = numexpr.evaluate('julian_date - 2451545.0')
    t = numexpr.evaluate('t0 / 36525')

    # Compute GST in seconds
    theta = numexpr.evaluate('c1 + (c2 * t0) + (t**2)' \
                             ' * (c3 - (t / c4))')

    # Compute LST in hours
    gst = numexpr.evaluate('theta / 15.0')

    # Deal with LST out of bounds
    negative = numexpr.evaluate('gst < 0.0')
    negative_gst = gst[negative]
    negative_gst = numexpr.evaluate('24.0 + negative_gst % 24')
    gst[negative] = negative_gst
    gst = numexpr.evaluate('gst % 24.0')

    return gst


# IMPORTANT: Latitude is in RADIANS
#            Longitude is in DEGREES
# in order to reduce function calls
def azel2radec(julian_date, azimuth, elevation, latitude, longitude_deg):
    """Convert from horizon coordinates (azimuth-elevation) to celestial
    coordinates (right ascension-declination)

    Input:
    julian_date   = Julian date (scalar or vector double)0
    azimuth       = azimuth in radians (scalar or vector double)
    elevation     = elevation in radians (scalar or vector double)
    latitude      = latitude in radians north of the equator (scalar or
                    vector double)
    longitude_deg = longitude in degrees east of the prime meridian
                    (scalar or vector double)

    Output:
    ra  = right ascension in radians (scalar or vector double)
    dec = declination in radians (scalar or vector double)

    Original IDL source available at:
    http://cosmology.berkeley.edu/group/cmbanalysis/forecast/idl/azel2radec.pro

    Revision history of IDL source:
    Created, Amedeo Balbi, August 1998 (based partly on material
    by Pedro Gil Ferreira)
       Modified, Amedeo Balbi, October 1998, to accept vectors as input
    """

    # Rescale azimuth
    azimuth = numexpr.evaluate('azimuth - pi')

    # Get the Greenwich Sidereal Time of the date
    gst = ct2lst(julian_date)

    # Get the Local Sidereal Time
    lst = numexpr.evaluate('gst + longitude_deg / 15.0')

    # Calculate the declination
    dec = numexpr.evaluate('arcsin( ' \
                           'sin(elevation) * sin(latitude) - ' \
                           'cos(elevation) * cos(azimuth) * cos(latitude)' \
                           ' )')

    # Calculate the right ascension from the hour angle
    ha = numexpr.evaluate('arctan( ' \
                          '(cos(elevation)*sin(azimuth)) / ' \
                          '(sin(elevation)*cos(latitude)+' \
                          'cos(elevation)*cos(azimuth)*sin(latitude))' \
                          ' )')
    ha = numexpr.evaluate('24 * ha / 2.0 / pi')
    ra = numexpr.evaluate('lst - ha')

    # Deal with RA out of bounds
    negative = numexpr.evaluate('ra<0.0')
    negative_ra = ra[negative]
    negative_ra = numexpr.evaluate('24.0 + negative_ra % 24')
    ra[negative] = negative_ra
    ra = numexpr.evaluate('ra % 24.0')

    # Convert from hours to radians
    ra = numexpr.evaluate('ra * pi / 12')

    return ra, dec