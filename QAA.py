# This script is a Python implementation of the QAA v6 algorithm
# Original Python code by Kelly Luis @ NASA JPL
# https://github.com/m11keluis/oceanoptics/blob/master/Python/algorithms/

import numpy as np
def iop2kd(a: np.ndarray, bb: np.ndarray, bbw: np.ndarray, sa: float = 30):
    """
    iop2kd-Computes diffuse attenuation coefficient (Lee et al. 2013)

    Parameters:
    a : numpy array
        Total absorption (1/m) from QAAv6 (Lee et al. 2002)
    bb : numpy array
        Total backscattering (1/m) from QAAv6 (Lee et al. 2002)
    bbw : numpy array
        Pure water backscattering coefficients
    sa : float, optional
        Solar Zenith Angle (default=30 degrees)

    Returns:
    kd : numpy array
        Diffuse attenuation coefficient (1/m) from Lee et al. (2013)
    """
    if sa == None:
        sa = 30.0

    # Contribution from moleculuar backscattering to total backscattering
    nw = np.divide(bbw, bb)

    # Compute Diffuse Attenuation Coefficient
    kd = (1 + 0.005*sa)*a + (1 - 0.265*nw)*4.259*(1 - 0.52*np.exp(-10.8*a))*bb

    return kd


def Rrs2Zsd(Rrs: np.ndarray, wl: np.ndarray, sa: float = 30.0, iop_functions=None):
    """
    Rrs2Zsd-Computes Secchi Disk Depth from Rrs

    Parameters:
    Rrs : numpy array
        Remote sensing reflectance
    wl : numpy array
        Wavelengths (nm)
    sa : float, optional
        Solar Zenith Angle (default=30 degrees)
    iop_functions : dict, optional
        Dictionary of functions for absorption and backscattering coefficients

    Returns:
    zsd : float
        Secchi Disk Depth (m)
    kd : numpy array
        Derived diffuse attenuation coefficient (1/m) from Lee et al. (2013)
    """
    import numpy as np


    # Load absorption and backscattering coefficients
    if iop_functions is None:
        iop_functions = h20_iop_lut()
        aw = iop_functions['aw'](wl)
        bbw = iop_functions['bbw'](wl)

    _, derived_iop = qaav6(Rrs, wl, aw, bbw)
    a = derived_iop['a'](wl)  # Total absorption
    bb = derived_iop['bb'](wl)  # Total backscattering
    # bbp = derived_iop['bbp'](wl) # suspended particulate backscattering

    kd = iop2kd(a, bb, bbw, sa)

    # Kd to Secchi Disk Depths

    # get minimum Kd and index
    kdmin = np.min(kd)
    index = np.argmin(kd)

    zsd = 1. / (2.5 * kdmin) * np.log(abs(0.14 - Rrs[index]) / 0.013)

    return zsd, kd


def qaav6(Rrs, wl, aw, bbw):
    """
    qaav6 - Computes the QAAv6 algorithm (Lee et al. 2002)

    Parameters:
    Rrs : numpy array
        Remote sensing reflectance
    wl : numpy array
        Wavelengths (nm)
    aw : numpy array
        Total absorption (1/m) from QAAv6 (Lee et al. 2002)
    bbw : numpy array
        Pure water backscattering coefficients
    """

    # Import Libraries
    import numpy as np
    import pandas as pd
    from scipy.interpolate import interp1d

    # Find Closest Wavelength to 412, 443, 490, 555, and 670
    id412 = np.where(abs(wl - 412) == min(abs(wl - 412)))[0][0]
    id443 = np.where(abs(wl - 443) == min(abs(wl - 443)))[0][0]
    id490 = np.where(abs(wl - 490) == min(abs(wl - 490)))[0][0]
    id555 = np.where(abs(wl - 550) == min(abs(wl - 550)))[0][0]
    id670 = np.where(abs(wl - 670) == min(abs(wl - 670)))[0][0]

    # Step 1: Compute ratio of backscattering coefficient to the sum of the absorption and
    # backscattering coefficients, bb/a+bb

    # 1 - Compute Subsurface remote sensing reflectance
    rrs = Rrs/(0.52 + 1.7 * Rrs)

    # for values in rrs that are less than 0, set to epsilon 
    rrs[rrs <= 0] = 1e-10

    g0 = 0.089
    g1 = 0.1245

    # Ratio of Backscattering to the Sum of Total Absorption and Total Backscattering
    u = (-g0 + np.sqrt(g0 ** 2 + 4 * g1 * rrs)) / (2 * g1)

    # Step 2: Determine Reference Wavelength by Rrs value: Determined by value of Red Band Rrs

    if Rrs[id670] < 0.0015:

        wl_ref = wl[id555]
        id_ref = id555

        ki = np.log10((rrs[id443] + rrs[id490]) / (rrs[id_ref] +
                      5 * rrs[id670] * rrs[id670] / rrs[id490]))
        a_ref = aw[id555] + 10 ** (-1.146 - 1.366 * ki - 0.469 * ki ** 2)

    else:

        wl_ref = 670
        id_ref = id670
        a_ref = aw[id670] + 0.39 * \
            (rrs[id670] / (rrs[id443] + rrs[id490])) ** 1.14

    # Step 3: Compute Backscattering of Particles at the Reference Wavelength

    bbp_ref = u[id_ref] * a_ref/(1 - u[id_ref]) - bbw[id555]

    # Step 4
    Y = 2.0 * (1 - 1.2 * np.exp(-0.9 * rrs[id443] / rrs[id555]))
    
    # Step 5: Compute Total Absorption and Backscattering by Particles
    a = []
    bbp = []

    # Use Reference Wavelengths to Compute Total Absorption and Backscattering
    for i in range(len(wl)):
        bbp_temp = bbp_ref * (wl_ref / wl[i]) ** Y
        a_temp = (1 - u[i]) * (bbw[i] + bbp_temp) / u[i]

        bbp = np.append(bbp, bbp_temp)
        a = np.append(a, a_temp)

    # Step 6: Compute Total Backscattering Coefficient
    bb = bbw + bbp

    # Step 7: Put IOPs into DataFrame
    iop_df = pd.DataFrame(data={'a': a, 'bbp': bbp, 'bb': bb}, index=wl)

    iop_functions = dict()
    iop_functions['a'] = interp1d(wl, a, kind="cubic")
    iop_functions['bbp'] = interp1d(wl, bbp, kind="cubic")
    iop_functions['bb'] = interp1d(wl, bb, kind="cubic")

    # Step 8: Compute Sigma and Epsilon

    sigma = 0.74 + 0.2 / (0.8 + rrs[id443] / rrs[id555])
    S = 0.015 + 0.002 / (0.6 + rrs[id443] / rrs[id555])
    epsilon = np.exp(S*(442.5 - 415.5))

    # Step 9 & 10: Compute adg and aph

    adg443 = (iop_functions['a'](412) - sigma * iop_functions['a'](443)) / (epsilon - sigma) - (aw[id412] - sigma * aw[id443])/(epsilon - sigma)

    adg = []
    aph = []

    for i in range(len(wl)):
        adgval = adg443 * np.exp(-S*(wl[i] - 443))
        adg.append(adgval)
        aph.append(iop_functions['a'](wl[i]) - adgval - aw[i])

    iop_df['adg'] = adg
    iop_df['aph'] = aph

    iop_functions['adg'] = interp1d(wl, adg, kind="cubic")
    iop_functions['aph'] = interp1d(wl, aph, kind="cubic")

    return iop_df, iop_functions

def h20_iop_lut() -> dict:
    """
    Returns a dictionary containing the water IOPs for the QAA v6 algorithm.

    Returns
    -------
    dict
        Dictionary containing the water IOPs for the QAA v6 algorithm.
            - "aw" : absorption coefficient for pure sea water
            - "bbw" : backscattering coefficient for pure sea water
            - "bbw_half" : half of bbw


    """
    import numpy as np
    from scipy.interpolate import interp1d

    abw = np.array([[200, 3.07E+00, 1.51E-01],
                    [205, 2.48E+00, 1.36E-01],
                    [210, 1.99E+00, 1.23E-01],
                    [215, 1.61E+00, 1.11E-01],
                    [220, 1.31E+00, 1.00E-01],
                    [225, 1.09E+00, 9.10E-02],
                    [230, 9.27E-01, 8.28E-02],
                    [235, 8.12E-01, 7.54E-02],
                    [240, 7.20E-01, 6.89E-02],
                    [245, 6.34E-01, 6.30E-02],
                    [250, 5.59E-01, 5.77E-02],
                    [255, 5.03E-01, 5.30E-02],
                    [260, 4.57E-01, 4.87E-02],
                    [265, 4.15E-01, 4.49E-02],
                    [270, 3.73E-01, 4.14E-02],
                    [275, 3.30E-01, 3.83E-02],
                    [280, 2.88E-01, 3.54E-02],
                    [285, 2.51E-01, 3.28E-02],
                    [290, 2.15E-01, 3.04E-02],
                    [295, 1.76E-01, 2.82E-02],
                    [300, 1.41E-01, 2.63E-02],
                    [305, 1.16E-01, 2.45E-02],
                    [310, 9.90E-02, 2.28E-02],
                    [315, 8.52E-02, 2.13E-02],
                    [320, 7.30E-02, 1.99E-02],
                    [325, 6.11E-02, 1.86E-02],
                    [330, 5.00E-02, 1.74E-02],
                    [335, 4.06E-02, 1.63E-02],
                    [340, 3.25E-02, 1.53E-02],
                    [345, 2.56E-02, 1.44E-02],
                    [350, 2.04E-02, 1.35E-02],
                    [355, 1.76E-02, 1.27E-02],
                    [360, 1.56E-02, 1.19E-02],
                    [365, 1.32E-02, 1.13E-02],
                    [370, 1.14E-02, 1.06E-02],
                    [375, 1.15E-02, 1.00E-02],
                    [380, 1.14E-02, 9.46E-03],
                    [385, 9.41E-03, 8.94E-03],
                    [390, 8.51E-03, 8.46E-03],
                    [395, 8.13E-03, 8.00E-03],
                    [400, 6.63E-03, 7.58E-03],
                    [405, 5.30E-03, 7.18E-03],
                    [410, 4.73E-03, 6.81E-03],
                    [415, 4.44E-03, 6.47E-03],
                    [420, 4.54E-03, 6.14E-03],
                    [425, 4.78E-03, 5.83E-03],
                    [430, 4.95E-03, 5.55E-03],
                    [435, 5.30E-03, 5.28E-03],
                    [440, 6.35E-03, 5.02E-03],
                    [445, 7.51E-03, 4.78E-03],
                    [450, 9.22E-03, 4.56E-03],
                    [455, 9.62E-03, 4.34E-03],
                    [460, 9.79E-03, 4.14E-03],
                    [465, 1.01E-02, 3.96E-03],
                    [470, 1.06E-02, 3.78E-03],
                    [475, 1.14E-02, 3.61E-03],
                    [480, 1.27E-02, 3.45E-03],
                    [485, 1.36E-02, 3.30E-03],
                    [490, 1.50E-02, 3.15E-03],
                    [495, 1.73E-02, 3.02E-03],
                    [500, 2.04E-02, 2.89E-03],
                    [505, 2.56E-02, 2.77E-03],
                    [510, 3.25E-02, 2.65E-03],
                    [515, 3.96E-02, 2.54E-03],
                    [520, 4.09E-02, 2.44E-03],
                    [525, 4.17E-02, 2.34E-03],
                    [530, 4.34E-02, 2.25E-03],
                    [535, 4.52E-02, 2.16E-03],
                    [540, 4.74E-02, 2.07E-03],
                    [545, 5.11E-02, 1.99E-03],
                    [550, 5.65E-02, 1.92E-03],
                    [555, 5.96E-02, 1.84E-03],
                    [560, 6.19E-02, 1.77E-03],
                    [565, 6.42E-02, 1.71E-03],
                    [570, 6.95E-02, 1.64E-03],
                    [575, 7.72E-02, 1.58E-03],
                    [580, 8.96E-02, 1.52E-03],
                    [585, 1.10E-01, 1.47E-03],
                    [590, 1.35E-01, 1.41E-03],
                    [595, 1.67E-01, 1.36E-03],
                    [600, 2.22E-01, 1.32E-03],
                    [605, 2.58E-01, 1.27E-03],
                    [610, 2.64E-01, 1.22E-03],
                    [615, 2.68E-01, 1.18E-03],
                    [620, 2.76E-01, 1.14E-03],
                    [625, 2.83E-01, 1.10E-03],
                    [630, 2.92E-01, 1.07E-03],
                    [635, 3.01E-01, 1.03E-03],
                    [640, 3.18E-01, 9.95E-04],
                    [645, 3.25E-01, 9.62E-04],
                    [650, 3.40E-01, 9.31E-04],
                    [655, 3.71E-01, 9.00E-04],
                    [660, 4.10E-01, 8.71E-04],
                    [665, 4.29E-01, 8.43E-04],
                    [670, 4.39E-01, 8.16E-04],
                    [675, 4.48E-01, 7.91E-04],
                    [680, 4.65E-01, 7.66E-04],
                    [685, 4.86E-01, 7.42E-04],
                    [690, 5.16E-01, 7.19E-04],
                    [695, 5.59E-01, 6.97E-04],
                    [700, 6.24E-01, 6.76E-04],
                    [705, 7.04E-01, 6.55E-04],
                    [710, 8.27E-01, 6.36E-04],
                    [715, 1.01E+00, 6.17E-04],
                    [720, 1.23E+00, 5.98E-04],
                    [725, 1.49E+00, 5.81E-04],
                    [730, 1.80E+00, 5.64E-04],
                    [735, 2.13E+00, 5.47E-04],
                    [740, 2.38E+00, 5.31E-04],
                    [745, 2.46E+00, 5.16E-04],
                    [750, 2.47E+00, 5.02E-04],
                    [755, 2.51E+00, 4.87E-04],
                    [760, 2.55E+00, 4.74E-04],
                    [765, 2.55E+00, 4.60E-04],
                    [770, 2.51E+00, 4.48E-04],
                    [775, 2.45E+00, 4.35E-04],
                    [780, 2.36E+00, 4.23E-04],
                    [785, 2.25E+00, 4.12E-04],
                    [790, 2.16E+00, 4.01E-04],
                    [795, 2.10E+00, 3.90E-04],
                    [800, 2.07E+00, 3.80E-04],
                    [805, 2.05E+00, 3.69E-04],
                    [810, 2.09E+00, 3.60E-04],
                    [815, 2.25E+00, 3.50E-04],
                    [820, 2.47E+00, 3.41E-04],
                    [825, 2.82E+00, 3.32E-04],
                    [830, 3.10E+00, 3.24E-04],
                    [835, 3.34E+00, 3.15E-04],
                    [840, 3.71E+00, 3.07E-04],
                    [845, 3.98E+00, 3.00E-04],
                    [850, 4.38E+00, 2.92E-04],
                    [855, 4.63E+00, 2.85E-04],
                    [860, 4.94E+00, 2.78E-04],
                    [865, 5.15E+00, 2.71E-04],
                    [870, 5.37E+00, 2.64E-04],
                    [875, 5.61E+00, 2.58E-04],
                    [880, 5.83E+00, 2.51E-04],
                    [885, 6.01E+00, 2.45E-04],
                    [890, 6.26E+00, 2.39E-04],
                    [895, 6.47E+00, 2.34E-04],
                    [900, 6.82E+00, 2.28E-04],
                    [905, 7.10E+00, 2.23E-04],
                    [910, 7.90E+00, 2.18E-04],
                    [915, 9.48E+00, 2.12E-04],
                    [920, 1.12E+01, 2.07E-04],
                    [925, 1.47E+01, 2.03E-04],
                    [930, 1.93E+01, 1.98E-04],
                    [935, 2.34E+01, 1.93E-04],
                    [940, 2.93E+01, 1.89E-04],
                    [945, 3.47E+01, 1.85E-04],
                    [950, 3.83E+01, 1.81E-04],
                    [955, 4.20E+01, 1.77E-04],
                    [960, 4.41E+01, 1.73E-04],
                    [965, 4.49E+01, 1.69E-04],
                    [970, 4.53E+01, 1.65E-04],
                    [975, 4.49E+01, 1.61E-04],
                    [980, 4.37E+01, 1.58E-04],
                    [985, 4.24E+01, 1.55E-04],
                    [990, 4.14E+01, 1.51E-04],
                    [995, 3.97E+01, 1.48E-04],
                    [1000, 3.77E+01, 1.45E-04],
                    [1005, 3.54E+01, 1.42E-04],
                    [1010, 3.32E+01, 1.39E-04],
                    [1015, 3.12E+01, 1.36E-04],
                    [1020, 2.93E+01, 1.33E-04],
                    [1025, 2.69E+01, 1.30E-04],
                    [1030, 2.46E+01, 1.27E-04],
                    [1035, 2.24E+01, 1.25E-04],
                    [1040, 2.04E+01, 1.22E-04],
                    [1045, 1.85E+01, 1.20E-04],
                    [1050, 1.69E+01, 1.17E-04],
                    [1055, 1.61E+01, 1.15E-04],
                    [1060, 1.54E+01, 1.13E-04],
                    [1065, 1.50E+01, 1.10E-04],
                    [1070, 1.49E+01, 1.08E-04],
                    [1075, 1.52E+01, 1.06E-04],
                    [1080, 1.57E+01, 1.04E-04],
                    [1085, 1.66E+01, 1.02E-04],
                    [1090, 1.75E+01, 9.97E-05],
                    [1095, 1.86E+01, 9.78E-05],
                    [1100, 1.99E+01, 9.59E-05],
                    [1105, 2.16E+01, 9.40E-05],
                    [1110, 2.35E+01, 9.22E-05],
                    [1115, 2.66E+01, 9.04E-05],
                    [1120, 3.02E+01, 8.87E-05],
                    [1125, 3.62E+01, 8.70E-05],
                    [1130, 4.34E+01, 8.54E-05],
                    [1135, 5.32E+01, 8.38E-05],
                    [1140, 6.51E+01, 8.22E-05],
                    [1145, 8.00E+01, 8.06E-05],
                    [1150, 9.83E+01, 7.91E-05]])

    a = interp1d(abw[:, 0], abw[:, 1], kind="cubic")
    b = interp1d(abw[:, 0], abw[:, 2], kind="cubic")
    bb = interp1d(abw[:, 0], 0.5*abw[:, 2], kind="cubic")

    out = dict()

    # Compute total absorption and backscattering

    out['aw'] = a
    out['bbw'] = b
    out['bbw_half'] = bb

    return out