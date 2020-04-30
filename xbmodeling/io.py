# from .config import modelconf
import numpy as np
import h5py
import pandas as pd


def load_tod_file(filename, ftype='mat7.3'):
    """
    Loads an external data file which contains :
        -- calibrated but unfiltered detector timestreams for a single pair
           (relative gain correction has been applied, i.e. detectors in the same
           pair have their responses normalized)
        -- telescope pointing timestreams in both AZ/EL and RA/Dec
        -- data with start/end/direction of az scans
        
    Parameters
    ----------
    filename : string
        path to file to be loaded
    type : string, optional
        'mat7.3' (default) -- the file is a .mat file saved with -v7.3 flag
        No other options set up yet

    Returns
    -------
    tuple of tod, scans, det_info
        tod_full : pandas DataFrame
            raw signal timestreams from each detector
            telescope pointing timestreams
        scans : pandas DataFrame
            contains indices marking beginning/end of az scan, as well as
            direction and MJD time of each scan
        det_info : pandas DataFrame
            contains auxillary detector data per detector.  For now just
            contains r,theta (telescope-fixed pointings) and 
            drumangle of that receiver in telescope mount

    """

    # Method for loading file completely changes based on type
    if ftype == 'mat7.3':
        # Need to use h5py for this file type, loadmat won't work
        f = h5py.File(filename, 'r')

        # First let's find which index is A pol and which is B pol
        # This messy bit is for converting HDF5 object reference to string
        pol = []
        for column in f['d']['p']['pol']:
            row_data = []
            for row_number in range(len(column)):
                row_data.append(''.join(map(chr, f[column[row_number]][0])))
            pol.append(row_data)
        A_det = pol[0].index('A')
        B_det = pol[0].index('B')

        # Get A det and B det timestreams
        A_sig = np.asarray(f['d']['fb'])[A_det]
        B_sig = np.asarray(f['d']['fb'])[B_det]

        # Telescope pointing
        tel_hor_az = np.asarray(f['d']['pointing']['hor']['az']).flatten()
        tel_hor_el = np.asarray(f['d']['pointing']['hor']['el']).flatten()
        tel_hor_dk = np.asarray(f['d']['pointing']['hor']['dk']).flatten()
        tel_cel_ra = np.asarray(f['d']['pointing']['cel']['ra']).flatten()
        tel_cel_dec = np.asarray(f['d']['pointing']['cel']['dec']).flatten()
        tel_cel_dk = np.asarray(f['d']['pointing']['cel']['dk']).flatten()

        # UTC time
        time_utc = np.asarray(f['d']['utcfast']).flatten()

        # Put all of above into tod dataframe
        tod_full = pd.DataFrame({
            'time_utc': time_utc,
            'A_sig': A_sig,
            'B_sig': B_sig,
            'tel_hor_az': tel_hor_az,
            'tel_hor_el': tel_hor_el,
            'tel_hor_dk': tel_hor_dk,
            'tel_cel_ra': tel_cel_ra,
            'tel_cel_dec': tel_cel_dec,
            'tel_cel_dk': tel_cel_dk})

        # scan_ind will contain useful info about the telescope az scans
        # The -1 is converting from MATLAB to python indexing
        scans = pd.DataFrame({
            'start': np.asarray(f['fs']['sf'], dtype='int').flatten() - 1,
            'end': np.asarray(f['fs']['ef'], dtype='int').flatten() - 1,
            'dir': np.asarray(f['fs']['inc'], dtype='int').flatten(),
            'time_mjd': np.asarray(f['fs']['t']).flatten()})

        # For each det we need some external info later on, for now
        # just keep the detector's pointing in telescope-fixed coords
        det_info = pd.DataFrame({
            'r': np.asarray(f['d']['p']['r']).flatten(),
            'theta': np.asarray(f['d']['p']['theta']).flatten(),
            'ukpervolt': np.asarray(f['d']['p']['ukpv']).flatten(),
            'drumangle': np.asarray(f['d']['p']['drumangle']).flatten()})

    return tod_full, scans, det_info


def keep_scans(tod_full, scans, halfscan_ind=None):
    """
    Take output of load_tod_file and pick out any number of halfscans
    (one sweep in AZ) to keep.  Trim everything away.

    Parameters
    ----------
    tod_full : pandas DataFrame
        raw signal timestreams from each detector
        telescope pointing timestreams
        detector pointing timestreams
    scans : pandas DataFrame
        contains indices marking beginning/end of az scan, as well as
        direction and MJD time of each scan
    halfscan_ind : array of non-negative integers (optional)
        indices identifying which half az scans to keep. 
        e.g. [0] keeps just the first halfscan,
        np.arange(2,6) keeps halfscans 2, 3, 4, and 5, etc
        If not given, use all scans in tod

    Returns
    -------
    tod : pandas dataframe
        same as input tod_full but only keeping timestreams of chosen scans
        
    """
    if halfscan_ind is None:
        halfscan_ind = np.arange(len(scans))

    # Which scans to keep.  +1 to keep the range inclusive.
    ind_keep = np.array([np.arange(scans['start'][halfscan_ind[i]], scans['end'][halfscan_ind[i]] + 1) \
                         for i in range(len(halfscan_ind))], dtype='int')

    tod = tod_full.loc[np.array(ind_keep).flatten()]

    return tod
