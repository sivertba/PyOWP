"""This module provides classes and functions for processing on-water radiometry data, specifically for the Pamela Ramses system that collects remote sensing reflectance.

Classes:
    A class to represent samples from an on-water radiometry sensor system.

    A class to represent a set of OnWaterRadiometrySample objects.

Functions:
aph2chl(aph440: float) -> float:
    Convert absorption coefficient at 440 nm to chlorophyll-a concentration (mg/m3).

OC6PACE(Rrs: np.array, wl: np.array) -> float:
    Compute the chlorophyll-a concentration using the OC6 algorithm.

get_sza(timestamp: datetime, latitude: float, longitude: float, timezone: str) -> float:
    Compute the solar zenith angle of the location where and at the time the sample was collected.

write_to_excel(file_path: str) -> None:
    Write the samples to an Excel file.
"""
from datetime import datetime
import os
import sys

from pvlib import solarposition
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pytz

import QAA

try:
    sys.path.append('Rrs_quality_assurance')
    from Rrs_quality_assurance.qwip_calculations import getQWIP
except BaseException:
    pass

try:
    sys.path.append('PyTrios_master')
except BaseException:
    pass


class OnWaterRadiometrySample:
    """
    A class to represent samples from an on water radiometry sensor system such as the Pamela Ramses system that collects remote sensing reflectance.

    Attributes:
    -----------
    sample_id : str
        Unique identifier for the sample.
    timestamp : datetime
        The date and time when the sample was collected.
    location : tuple
        The geographical location (latitude, longitude) where the sample was collected.
    location_name : str
        The name of the location where the sample was collected.
    reflectance_data : dict
        A dictionary containing wavelength as keys and corresponding reflectance values as values.
        Expected keys are "wl" for wavelength, Es for downwelling irradiance, Lw for upwelling water radiance, and Rrs for remote sensing reflectance.
    iop_data : dict
        A dictionary containing wavelength as keys and corresponding inherent optical properties as values.
    derived_data : dict
        A dictionary containing wavelength as keys and corresponding derived data as values.
    sza : float
        Solar zenith angle of the location where and at the tiem the sample was collected.
    """

    def __init__(
            self,
            sample_id,
            timestamp,
            location,
            reflectance_data,
            verbose=False):
        """ Initialize the OnWaterRadiometrySample object.

        Parameters:
        -----------
        sample_id : str
            Unique identifier for the sample.
        timestamp : datetime
            The date and time when the sample was collected.
        location : tuple
            The geographical location (latitude, longitude) where the sample was collected.
        reflectance_data : dict
            A dictionary containing wavelength as keys and corresponding reflectance values as values.
            Expected keys are "wl" for wavelength, Es for downwelling irradiance, Lw for upwelling water radiance, and Rrs for remote sensing reflectance.
        verbose : bool
            If True, print error messages.
        """
        self.sample_id = sample_id
        self.timestamp = timestamp
        self.location = location
        self.location_name = "TBD"
        self.reflectance_data = reflectance_data
        self.verbose = verbose
        self.maxWlPlot = 850
        self.minWlPlot = 400

        if self.timestamp.tzinfo is None:
            # assume UTC0 if no timezone info
            tzinfo = pytz.timezone('UTC')
            self.timestamp = self.timestamp.replace(tzinfo=tzinfo)

        try:
            self.compute_iop_data()
        except BaseException:
            self.iop_data = None
            if self.verbose:
                print("Error computing IOP data. Sample id: ", self.sample_id)

        try:
            self.sza = get_sza(
                self.timestamp,
                self.location[0],
                self.location[1],
                'UTC')
        except BaseException:
            self.sza = None
            pass

        try:
            self.analyze_sample()
        except BaseException:
            if self.verbose:
                print("Error analyzing sample. Sample id: ", self.sample_id)
            self.derived_data = None
            pass

    def get_sample_id(self) -> str:
        """ Get the sample_id attribute of the OnWaterRadiometrySample object.

        Returns:
        --------
        str:
            The sample_id attribute of the OnWaterRadiometrySample object.
        """
        return self.sample_id

    def get_timestamp(self) -> datetime:
        """ Get the timestamp attribute of the OnWaterRadiometrySample object.

        Returns:
        --------
        datetime:
            The timestamp attribute of the OnWaterRadiometrySample object.
        """
        return self.timestamp

    def get_location(self) -> tuple:
        """ Get the location attribute of the OnWaterRadiometrySample object.

        Returns:
        --------
        tuple:
            The location attribute of the OnWaterRadiometrySample object in the form (latitude, longitude).
        """
        return self.location

    def get_location_name(self) -> str:
        """ Get the location_name attribute of the OnWaterRadiometrySample object.

        Returns:
        --------
        str:
            The location_name attribute of the OnWaterRadiometrySample object.
        """
        return self.location_name

    def get_reflectance_data(self) -> dict:
        """ Get the reflectance_data attribute of the OnWaterRadiometrySample object.

        Returns:
        --------
        dict:
            The reflectance_data attribute of the OnWaterRadiometrySample object.
        """
        return self.reflectance_data

    def get_iop_data(self) -> dict:
        """ Get the iop_data attribute of the OnWaterRadiometrySample object.

        Returns:
        --------
        dict:
            The iop_data attribute of the OnWaterRadiometrySample object.
        """
        return self.iop_data

    def get_derived_data(self) -> dict:
        """ Get the derived_data attribute of the OnWaterRadiometrySample object.

        Returns:
        --------
        dict:
            The derived_data attribute of the OnWaterRadiometrySample object.
        """
        return self.derived_data

    def get_sza(self) -> float:
        """ Get the sza attribute of the OnWaterRadiometrySample object.

        Returns:
        --------
        float:
            The sza attribute of the OnWaterRadiometrySample object.
        """
        return self.sza

    def recompute_sza(self, timezone='UTC') -> float:
        """ Recompute the solar zenith angle of the location where and at the time the sample was collected.

        Returns:
        --------
        float:
            The sza attribute of the OnWaterRadiometrySample object.
        """
        import pytz
        import pandas as pd

        ts = self.timestamp

        # remove timezone info
        ts = ts.replace(tzinfo=None)

        # save as pandas timestamp
        ts = pd.Timestamp(ts)

        self.sza = get_sza(
            ts,
            self.location[0],
            self.location[1],
            timezone)
        return self.sza

    def set_verbose(self, verbose: bool) -> bool:
        """ Set the verbose attribute of the OnWaterRadiometrySample object.

        Parameters:
        -----------
        verbose : bool
            If True, print error messages.

        Returns:
        --------
        bool:
            The verbose attribute of the OnWaterRadiometrySample object.
        """
        self.verbose = verbose
        return self.verbose

    def set_plot_wavelength_range(
            self,
            minWl: int = None,
            maxWl: int = None) -> tuple:
        """ Set the wavelength range for plotting the data.

        Parameters:
        -----------
        minWl : int
            The minimum wavelength for plotting the data.
        maxWl : int
            The maximum wavelength for plotting the data.

        Returns:
        --------
        tuple:
            A tuple containing the minimum and maximum wavelength for plotting the data.
        """
        if minWl is not None:
            self.minWlPlot = minWl
        if maxWl is not None:
            self.maxWlPlot = maxWl
        return (self.minWlPlot, self.maxWlPlot)

    def analyze_sample(self):
        """ Analyze the sample to compute the inherent optical properties and derived data.

        Returns:
        --------
        dict:
            A dictionary containing the computed inherent optical properties and derived data.
        """
        self.derived_data = {}

        self.derived_data["a440"] = self.iop_data["a"](440)

        # # Compute the chlorophyll-a concentration using aph(440) and add it to
        # the derived data
        try:
            chl = aph2chl(self.iop_data["aph"](440))
            self.derived_data["Chl_QAA"] = chl
        except BaseException:
            if self.verbose:
                print(
                    "Error computingchlorophyll-a concentration. Please check the IOP data.")

        # Compute the QWIP score and add it to the derived data
        try:
            qwip_score = getQWIP(
                self.reflectance_data["Rrs"],
                self.reflectance_data["wl"])
            self.derived_data["QWIP"] = qwip_score
        except BaseException:
            if self.verbose:
                print("Error computing QWIP score. Please check the Reflectance data.")
            pass

        # Compute the chlorophyll-a concentration using the OC6 algorithm and
        # add it to the derived data
        try:
            chl_oc6 = OC6PACE(
                self.reflectance_data["Rrs"],
                self.reflectance_data["wl"])
            self.derived_data["Chl_OC6"] = chl_oc6
        except BaseException:
            if self.verbose:
                print(
                    "Error computing chlorophyll-a concentration using OC6PACE. Please check the Reflectance data.")
            pass

        # Compute the Secchi disk depth and add it to the derived data
        try:
            Zsd, _ = QAA.Rrs2Zsd(self.reflectance_data["Rrs"],
                                 self.reflectance_data["wl"],
                                 self.sza)
            self.derived_data["Zsd"] = Zsd
        except BaseException:
            if self.verbose:
                print(
                    "Error computing Secchi Depth. Please check the Reflectance, IOP, or SZA data.")
            pass

    def sza(self, timezone='UTC'):
        return get_sza(
            self.timestamp,
            self.location[0],
            self.location[1],
            timezone)

    def set_sample_id(self, sample_id: str) -> str:
        """ Set the sample_id attribute of the OnWaterRadiometrySample object.

        Parameters:
        -----------
        sample_id : str
            Unique identifier for the sample.

        Returns:
        --------
        str:
            The sample_id attribute of the OnWaterRadiometrySample object.
        """
        self.sample_id = sample_id
        return self.sample_id

    def set_timestamp(self, timestamp: datetime) -> datetime:
        """ Set the timestamp attribute of the OnWaterRadiometrySample object.

        Parameters:
        -----------
        timestamp : datetime
            The date and time when the sample was collected.

        Returns:
        --------
        datetime:
            The timestamp attribute of the OnWaterRadiometrySample object.
        """
        self.timestamp = timestamp
        return self.timestamp

    def set_time(
            self,
            year: int,
            month: int,
            day: int,
            hour: int,
            minute: int,
            second: int) -> datetime:
        """ Set the timestamp attribute of the OnWaterRadiometrySample object.

        Parameters:
        -----------
        year : int
            The year when the sample was collected.
        month : int
            The month when the sample was collected.
        day : int
            The day when the sample was collected.
        hour : int
            The hour when the sample was collected.
        minute : int
            The minute when the sample was collected.
        second : int
            The second when the sample was collected.

        Returns:
        --------
        datetime:
            The timestamp attribute of the OnWaterRadiometrySample object.
        """
        try:
            self.timestamp = datetime(year, month, day, hour, minute, second)
        except ValueError:
            print("Invalid date and time")
        return self.timestamp

    def set_location(self, location: tuple) -> tuple:
        """ Set the location attribute of the OnWaterRadiometrySample object.

        Parameters:
        -----------
        location : tuple
            The geographical location (latitude, longitude) where the sample was collected.

        Returns:
        --------
        tuple:
            The location attribute of the OnWaterRadiometrySample object.
        """
        self.location = location
        return self.location

    def set_latitude(self, latitude: float) -> float:
        """ Set the latitude attribute of the OnWaterRadiometrySample object.

        Parameters:
        -----------
        latitude : float
            The latitude of the location where the sample was collected.

        Returns:
        --------
        float:
            The latitude attribute of the OnWaterRadiometrySample object.
        """
        self.location = (latitude, self.location[1])
        return self.location[0]

    def set_longitude(self, longitude: float) -> float:
        """ Set the longitude attribute of the OnWaterRadiometrySample object.

        Parameters:
        -----------
        longitude : float
            The longitude of the location where the sample was collected.

        Returns:
        --------
        float:
            The longitude attribute of the OnWaterRadiometrySample object.
        """
        self.location = (self.location[0], longitude)
        return self.location[1]

    def set_location(self, latitude: float, longitude: float) -> tuple:
        """ Set the location attribute of the OnWaterRadiometrySample object.

        Parameters:
        -----------
        latitude : float
            The latitude of the location where the sample was collected.
        longitude : float
            The longitude of the location where the sample was collected.

        Returns:
        --------
        tuple:
            The location attribute of the OnWaterRadiometrySample object.
        """
        self.location = (latitude, longitude)
        return self.location

    def set_location_name(self, location_name: str) -> str:
        """ Set the location_name attribute of the OnWaterRadiometrySample object.

        Parameters:
        -----------
        location_name : str
            The name of the location where the sample was collected.

        Returns:
        --------
        str:
            The location_name attribute of the OnWaterRadiometrySample object.
        """
        self.location_name = location_name
        return self.location_name

    def set_reflectance_data(self, reflectance_data: dict) -> dict:
        """ Set the reflectance_data attribute of the OnWaterRadiometrySample object.

        Parameters:
        -----------
        reflectance_data : dict
            A dictionary containing wavelength as keys and corresponding reflectance values as values.

        Returns:
        --------
        dict:
            The reflectance_data attribute of the OnWaterRadiometrySample object.
        """
        self.reflectance_data = reflectance_data
        return self.reflectance_data

    def set_io_data(self, iop_data: dict) -> dict:
        """ Set the iop_data attribute of the OnWaterRadiometrySample object.

        Parameters:
        -----------
        iop_data : dict
            A dictionary containing wavelength as keys and corresponding inherent optical properties as values.

        Returns:
        --------
        dict:
            The iop_data attribute of the OnWaterRadiometrySample object.
        """
        self.iop_data = iop_data
        return self.iop_data

    def compute_iop_data(self) -> dict:
        """ Compute the iop_data attribute of the OnWaterRadiometrySample object.

        Returns:
        --------
        dict:
            The iop_data attribute of the OnWaterRadiometrySample object.
        """
        iop_functions = QAA.h20_iop_lut()
        aw = iop_functions['aw'](self.reflectance_data["wl"])
        bbw = iop_functions['bbw'](self.reflectance_data["wl"])
        _, derived_iop = QAA.qaav6(self.reflectance_data["Rrs"],
                                   self.reflectance_data["wl"],
                                   aw,
                                   bbw)
        self.iop_data = derived_iop
        return self.iop_data

    def set_derived_data(self, derived_data: dict) -> dict:
        """ Set the derived_data attribute of the OnWaterRadiometrySample object.

        Parameters:
        -----------
        derived_data : dict
            A dictionary containing wavelength as keys and corresponding derived data as values.

        Returns:
        --------
        dict:
            The derived_data attribute of the OnWaterRadiometrySample object.
        """
        self.derived_data = derived_data
        return self.derived_data

    def __str__(self) -> str:
        """ Return a string representation of the OnWaterRadiometrySample object.

        Returns:
        --------
        str:
            A string representation of the OnWaterRadiometrySample object.
        """
        return f"OnWaterRadiometrySample: sample_id={
            self.sample_id}, timestamp={
            self.timestamp}, location={
            self.location}, location_name={
                self.location_name}"

    def __repr__(self) -> str:
        """ Return a string representation of the OnWaterRadiometrySample object.

        Returns:
        --------
        str:
            A string representation of the OnWaterRadiometrySample object.
        """
        return f"OnWaterRadiometrySample: sample_id={
            self.sample_id}, timestamp={
            self.timestamp}, location={
            self.location}, location_name={
                self.location_name}"

    def makePlotTilte(self):
        title = f"Data from {self.timestamp}"
        if self.location != (-1, -1):
            title += f", at location {self.location}"
        if self.location_name is not None or self.location_name == "TBD":
            title += f" ({self.location_name})"
        return title

    def plot_reflectance(self):
        """ Plot the reflectance data.
        """
        plt.plot(self.reflectance_data["wl"], self.reflectance_data["Rrs"])
        plt.xlabel("Wavelength (nm)")
        plt.title("Reflectance " + self.makePlotTilte())
        plt.ylabel("Reflectance")
        plt.xlim(self.minWlPlot, self.maxWlPlot)
        plt.show()

    def plot_radiance(self):
        """ Plot the radiance data.
        """
        plt.plot(self.reflectance_data["wl"], self.reflectance_data["Lw"])
        plt.xlabel("Wavelength (nm)")
        plt.title("Radiance " + self.makePlotTilte())
        plt.ylabel("Radiance")
        plt.xlim(self.minWlPlot, self.maxWlPlot)
        plt.show()

    def plot_irradiance(self):
        """ Plot the irradiance data.
        """
        plt.plot(self.reflectance_data["wl"], self.reflectance_data["Es"])
        plt.xlabel("Wavelength (nm)")
        plt.title("Irradiance " + self.makePlotTilte())
        plt.ylabel("Irradiance")
        plt.xlim(self.minWlPlot, self.maxWlPlot)

        plt.show()

    def plot_iop_absorption(self):
        """ Plot the inherent optical properties data.
        """
        iop_copy = self.iop_data.copy()
        wl = self.reflectance_data["wl"]

        # Limit to self.minWlPlot and self.maxWlPlot
        mask = (wl >= self.minWlPlot) & (wl <= self.maxWlPlot)
        wl = wl[mask]

        plt.plot(wl, iop_copy["a"](wl), label="total absorption")
        plt.plot(wl, iop_copy["aph"](wl), label="phytoplankton absorption")
        plt.plot(wl, iop_copy["adg"](wl), label="gelbstoff absorption")

        plt.xlabel("Wavelength (nm)")
        plt.ylabel("IOP Absorption")
        plt.title(f"Absorption {self.makePlotTilte()}")
        plt.legend()
        plt.xlim(self.minWlPlot, self.maxWlPlot)
        plt.show()

    def plot_iop_backscattering(self):
        """ Plot the inherent optical properties data.
        """
        iop_copy = self.iop_data.copy()
        wl = self.reflectance_data["wl"]

        plt.plot(wl, iop_copy["bb"](wl), label="total backscattering")
        plt.plot(wl, iop_copy["bbp"](wl), label="phytoplankton backscattering")

        plt.xlabel("Wavelength (nm)")
        plt.ylabel("IOP Backscattering")
        plt.title(f"Backscattering {self.makePlotTilte()}")
        plt.legend()
        plt.xlim(self.minWlPlot, self.maxWlPlot)
        plt.show()


class OnWaterRadiometrySet:
    """ A class to represent a set of OnWaterRadiometrySample objects.

    Attributes:
    -----------
    samples : list
        A list of OnWaterRadiometrySample objects.

    Methods:
    --------
    add(sample: OnWaterRadiometrySample) -> bool:
        Add a sample to the list of samples.

    remove(sample_id: str) -> bool:
        Remove a sample from the list of samples.

    get_sample(sample_id: str) -> OnWaterRadiometrySample:
        Get a sample from the list of samples.

    get_samples() -> list:
        Return the list of samples.

    get_sample_ids() -> OnWaterRadiometrySet:
        Return the list of sample IDs.

    get_sample_count() -> int:
        Return the number of samples in the list.

    get_samples_by_location(minLat: float, maxLat: float, minLon: float, maxLon: float) -> OnWaterRadiometrySet:
        Return a list of samples collected at a specific location.

    get_samples_by_time(from: datetime, to: datetime) -> OnWaterRadiometrySet:
        Return a list of samples collected on a specific date.

    write_to_excel(file_path: str):
        Write the samples to an Excel file.
    """

    def __init__(self):
        self.samples = []

    def __iter__(self):
        return iter(self.samples)

    def __next__(self):
        return next(self.samples)

    def __str__(self):
        return f"OnWaterRadiometrySet: sample_count={self.get_sample_count()}"

    # if passed an object of this class, it will make a deep copy of the object
    def __deepcopy__(self, memo):
        new_set = OnWaterRadiometrySet()
        for sample in self.samples:
            new_set.add(sample)
        return new_set

    def add(self, sample: OnWaterRadiometrySample) -> bool:
        """ Add a sample to the list of samples.

        Parameters:
        -----------
        sample : OnWaterRadiometrySample
            The sample to be added to the list of samples.

        Returns:
        --------
        bool:
            True if the sample was added successfully, False otherwise.
        """
        self.samples.append(sample)
        return True

    def remove(self, sample_id: str) -> bool:
        """ Remove a sample from the list of samples.

        Parameters:
        -----------
        sample_id : str
            The ID of the sample to be removed.

        Returns:
        --------
        bool:
            True if the sample was removed successfully, False otherwise.
        """
        for sample in self.samples:
            if sample.sample_id == sample_id:
                self.samples.remove(sample)
                return True
        return False

    def get_sample(self, sample_id: str) -> OnWaterRadiometrySample:
        """ Get a sample from the list of samples.

        Parameters:
        -----------
        sample_id : str
            The ID of the sample to be retrieved.

        Returns:
        --------
        OnWaterRadiometrySample:
            The sample with the specified ID.
        """
        for sample in self.samples:
            if sample.sample_id == sample_id:
                return sample
        return None

    def get_samples(self) -> list:
        """ Return the list of samples.

        Returns:
        --------
        list:
            The list of samples.
        """
        return self.samples

    def get_sample_ids(self) -> list:
        """ Return the list of sample IDs.

        Returns:
        --------
        list:
            The list of sample IDs.
        """
        return [sample.sample_id for sample in self.samples]

    def get_sample_count(self) -> int:
        """ Return the number of samples in the list.

        Returns:
        --------
        int:
            The number of samples in the list.
        """
        return len(self.samples)

    def get_samples_by_location(
            self,
            minLat: float,
            maxLat: float,
            minLon: float,
            maxLon: float):
        """ Return a list of samples collected at a specific location.

        Parameters:
        -----------
        minLat : float
            The minimum latitude of the location.
        maxLat : float
            The maximum latitude of the location.
        minLon : float
            The minimum longitude of the location.
        maxLon : float
            The maximum longitude of the location.

        Returns:
        --------
        OnWaterRadiometrySet:
            A list of samples collected at the specified location.
        """
        newSamples = OnWaterRadiometrySet()
        for sample in self.samples:
            if minLat <= sample.location[0] <= maxLat and minLon <= sample.location[1] <= maxLon:
                newSamples.add(sample)
        return newSamples

    def get_samples_by_time(self, from_dt: datetime, to_dt: datetime):
        """ Return a list of samples collected bewteen two time points.
        Parameters:
        -----------
        from_dt : datetime
            The start datetime.
        to_dt : datetime
            The end datetime.

        OnWaterRadiometrySet
            A set of samples collected between the specified datetimes.
        """
        newSamples = OnWaterRadiometrySet()
        for sample in self.samples:
            if from_dt <= sample.timestamp <= to_dt:
                newSamples.add(sample)
        return newSamples

    def set_verbose(self, verbose: bool) -> bool:
        """ Set the verbose attribute of the OnWaterRadiometrySample object.

        Parameters:
        -----------
        verbose : bool
            If True, print error messages.

        Returns:
        --------
        bool:
            The verbose attribute of the OnWaterRadiometrySample object.
        """
        for sample in self.samples:
            sample.set_verbose(verbose)
        return verbose

    def make_qwip_figure(self, qwip_thr: float = 0.2) -> tuple:
        """ Make a figure showing the QWIP scores for all samples in the set.

        Parameters:
        -----------
        qwip_thr : float (optional)
            The QWIP threshold. Default is 0.2.

        Returns:
        --------
        tuple:
            A tuple containing the figure, dataframe, and fit1.

        """
        import plotly.graph_objects as go
        import Rrs_quality_assurance.qwip_calculations as qwip
        df = []
        for sample in self.get_samples():
            Rrs = sample.get_reflectance_data()
            df.append(Rrs["Rrs"])
        wavenumbers = Rrs["wl"]
        df = pd.DataFrame(df, columns=wavenumbers)
        t = qwip.run_qwip_calculations(df, "hyper", wavenumbers)
        df_out, fit1, _, _, _, _, _, index_492, index_665 = t
        avw_poly = np.arange(400, 631)
        wave = np.arange(400, 701)

        # Generate figure to show NDI index relative to AVW
        qwip_threshold = qwip_thr
        fit4a = fit1 + qwip_threshold
        fit4b = fit1 - qwip_threshold

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=avw_poly,
                y=fit1,
                mode='lines',
                line=dict(
                    color="black",
                    width=1),
                name='QWIP Ideal'))
        fig.add_trace(
            go.Scatter(
                x=avw_poly,
                y=fit4a,
                mode='lines',
                line=dict(
                    color="red",
                    width=1,
                    dash='dash'),
                name=f'QWIP ± {qwip_threshold}'))
        fig.add_trace(
            go.Scatter(
                x=avw_poly,
                y=fit4b,
                mode='lines',
                line=dict(
                    color="red",
                    width=1,
                    dash='dash'),
                showlegend=False))

        marker_layout = dict(
            color=np.abs(
                df_out['qwip_score']),
            colorscale='viridis',
            size=7,
            opacity=0.7)
        fig.add_trace(
            go.Scatter(
                x=df_out['avw'],
                y=df_out['ndi'],
                mode='markers',
                marker=marker_layout,
                showlegend=False))

        fig.update_layout(
            xaxis_title='AVW (nm)', yaxis_title=f'NDI ({
                wave[index_492]},{
                wave[index_665]})')
        fig.update_xaxes(range=[440, 630])
        fig.update_yaxes(range=[-2.5, 2])

        # Make figure square
        fig.update_layout(width=800, height=800)
        # remove margins
        fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
        # fig.show()
        return fig, df_out, fit1

    def make_spectral_plot(self, variable: str = "rrs") -> go.Figure:
        """ Make a spectral plot of the samples in the set.

        Parameters:
        -----------
        variable : str
            The variable to plot. Default is "rrs". Valid variables are: "rrs", "lw", "es", "aph", "bb", "bbp", "a", "adg".

        Returns:
        --------
        go.Figure:
            A plotly figure.
        """
        valid_variables = ["rrs", "lw", "es", "aph", "bb", "bbp", "a", "adg"]
        if variable.lower() not in valid_variables:
            raise ValueError(
                f"Invalid variable. Valid variables are: {valid_variables}")

        fig = go.Figure()
        x = self.get_samples()[0].get_reflectance_data()["wl"]
        y = []
        variable = variable.lower()
        for sample in self.get_samples():
            wl = sample.get_reflectance_data()["wl"]
            if variable == "rrs":
                y.append(sample.get_reflectance_data()["Rrs"])
            elif variable == "lw":
                y.append(sample.get_reflectance_data()["Lw"])
            elif variable == "es":
                y.append(sample.get_reflectance_data()["Es"])
            elif variable == "aph":
                y.append(sample.get_iop_data()["aph"](wl))
            elif variable == "bb":
                y.append(sample.get_iop_data()["bb"](wl))
            elif variable == "bbp":
                y.append(sample.get_iop_data()["bbp"](wl))
            elif variable == "a":
                y.append(sample.get_iop_data()["a"](wl))
            elif variable == "adg":
                y.append(sample.get_iop_data()["adg"](wl))

        x = np.array(x)
        y = np.array(y)

        # add all the rows in the y array as traces
        for i in range(y.shape[0]):
            fig.add_trace(go.Scatter(x=x, y=y[i], mode='lines'))

        fig.update_layout(
            title=f"{variable} vs Wavelength",
            xaxis_title="Wavelength (nm)",
            yaxis_title=variable,
            showlegend=False
        )

        return fig

    def get_median_derived_data(self) -> dict:
        """ Get the median derived data from the samples.

        Returns:
        --------
        dict:
            A dictionary containing the median derived data.
        """
        median_derived_data = {}
        for key in self.samples[0].get_derived_data().keys():
            median_derived_data[key] = np.median(
                [s.get_derived_data()[key] for s in self.samples])
        return median_derived_data

    def write_to_excel(self, outPath):
        """ Write the samples to an Excel file.

        Parameters:
        -----------
        outPath : str
            The path to the Excel file.
        """

        import pandas as pd
        from datetime import datetime

        def create_sheet(writer, dataSet, key, col_names, data_func):
            sheet = pd.DataFrame(columns=col_names)
            for r in dataSet.get_samples():
                dt_obj = r.get_timestamp()
                isotime = dt_obj.strftime("%Y-%m-%dT%H:%M:%S")
                lat, lon = r.get_location()
                lat = round(float(lat.iloc[0]), 5)
                lon = round(float(lon.iloc[0]), 5)
                row = [isotime, lat, lon]
                row.extend(data_func(r, key))
                sheet.loc[len(sheet)] = row
            sheet.to_excel(writer, sheet_name=key)

        with pd.ExcelWriter(outPath, engine='openpyxl', mode='w') as writer:
            try:
                s = self.get_samples()[0]
            except IndexError:
                print("No samples in dataset")
                raise IndexError

            wl = s.get_reflectance_data()["wl"]

            # Reflectance data
            refl = s.get_reflectance_data()
            for key in refl.keys():
                if key == "wl":
                    continue
                col_names = ["istotime", "lat", "lon"] + list(wl)
                create_sheet(
                    writer,
                    self,
                    key,
                    col_names,
                    lambda r,
                    k: r.get_reflectance_data()[k])

            # IOP data
            iops = s.get_iop_data()
            for key in iops.keys():
                col_names = ["istotime", "lat", "lon"] + list(wl)
                create_sheet(
                    writer,
                    self,
                    key,
                    col_names,
                    lambda r,
                    k: r.get_iop_data()[k](wl))

            # Derived data
            derived = s.get_derived_data()
            col_names = ["istotime", "lat", "lon"] + list(derived.keys())
            create_sheet(writer, self, "derived", col_names, lambda r, k: [
                         round(float(r.get_derived_data()[k]), 3) for k in derived.keys()])


def aph2chl(aph440: float) -> float:
    """ Convert absorption coefficient at 440 nm to chlorophyll-a concentration (mg/m3).

    Based on: https://www.sciencedirect.com/science/article/pii/S0034425715300900

    Parameters:
    -----------
    aph440 : float
        Absorption due to phytoplankto at 440 nm.

    Returns:
    --------
    float:
        Chlorophyll-a concentration.
    """
    import numpy as np
    return np.exp(((np.log(aph440) - np.log(0.05)) / 0.65))


def OC6PACE(Rrs: np.array, wl: np.array) -> float:
    """ Compute the chlorophyll-a concentration using the OC6 algorithm.

    Based on: https://doi.org/10.1016/j.rse.2019.04.021

    Parameters:
    -----------
    Rrs : np.array
        Remote sensing reflectance.
    wl : np.array
        Wavelengths.

    Returns:
    --------
    float:
        Chlorophyll-a concentration.
    """
    # find index closest to values in wl
    idx_412 = np.argmin(np.abs(wl - 412))
    idx_443 = np.argmin(np.abs(wl - 443))
    idx_490 = np.argmin(np.abs(wl - 490))
    idx_510 = np.argmin(np.abs(wl - 510))
    idx_555 = np.argmin(np.abs(wl - 555))
    idx_678 = np.argmin(np.abs(wl - 678))

    MBR = np.max([Rrs[idx_412], Rrs[idx_443], Rrs[idx_490], Rrs[idx_510]])
    M = np.mean([Rrs[idx_555], Rrs[idx_678]])
    A = np.array([0.8502, 0.94297, -3.18493, 2.33682, -1.23923, 0.18697])

    chlExp = 0
    for i in range(A.size):
        chlExp += A[i] * (np.log10(MBR / M)) ** i
    chl = 10 ** chlExp
    return chl


def get_sza(
        date_time: datetime,
        latitude: float,
        longitude: float,
        timezone: str) -> float:
    """Calculate the solar zenith angle.

    Parameters
    ----------
    date_time : datetime
        Date and time.
    latitude : float
        Latitude of the location.
    longitude : float
        Longitude of the location.
    timezone : str
        Timezone of the location.

    Returns
    -------
    sza : float
        Solar zenith angle.
    """
    # Create a pandas Timestamp object
    date_time = pd.Timestamp(date_time, tz=timezone)

    # Calculate solar position using the pvlib library
    solpos = solarposition.get_solarposition(date_time, latitude, longitude)

    # Get the solar zenith angle
    sza = solpos['apparent_zenith'].values[0]

    return sza


def process_file_pamela_ramses(
        file_path: str,
        twoStageIrrCal: np.array = None,
        twoStageRadCal: np.array = None) -> dict:
    """
    Process a file containing spectral data from the pamela ramses system.

    Parameters:
    -----------
    file_path : str
        Path to the file containing the spectral data.

    Returns:
    --------
    dict:
        A dictionary containing the spectral data. Keys are "Es", "Lw", "Rrs", and "wl".
    """
    df = pd.read_csv(file_path, sep="\t", header=14)
    data_ramses = df.iloc[:, 1]
    data_ramses = {}

    if twoStageIrrCal is not None:
        data_ramses["Es"] = np.array(df.iloc[:, 1]) * twoStageIrrCal
    else:
        data_ramses["Es"] = np.array(df.iloc[:, 1])

    if twoStageRadCal is not None:
        data_ramses["Lw"] = np.array(df.iloc[:, 2]) * twoStageRadCal
    else:
        data_ramses["Lw"] = np.array(df.iloc[:, 2])

    # TODO: Check if this should be multiplied by pi
    data_ramses["Rrs"] = data_ramses["Lw"] / data_ramses["Es"]
    data_ramses["wl"] = np.array(df.iloc[:, 0])
    return data_ramses


def process_autonaut(
        file_path_down: str,
        file_path_up: str,
        CalRadDown: np.array = None,
        CalIrrUp: np.array = None,
        wavelengths: np.array = None) -> dict:
    """
    Process a file containing spectral data from the autonaut system.

    Parameters:
    -----------

    file_path_down : str
        Path to the file containing the spectral data for the sensor pointing down.
    file_path_up : str
        Path to the file containing the spectral data for the sensor pointing up.
    CalRadDown : np.array
        Calibration vector for the downwelling radiance.
    CalIrrUp : np.array
        Calibration vector for the upwelling irradiance.
    wavelengths : np.array
        Wavelengths.

    Returns:
    --------
    dict:
        A dictionary containing the spectral data. Keys are "Es", "Lw", "Rrs", and "wl".

    """

    from PyTrios_master.pytrios.TClasses import TPacket
    from PyTrios_master.python import _get_s2parse
    from PyTrios_master.python import handlePacket

    data_autonaut = {}

    for i, path in enumerate([file_path_up, file_path_down]):
        data = None
        with open(path, mode='rb') as file:
            fileContent = file.read()
            s, s2 = _get_s2parse(fileContent)
            while s is not None:
                packet = TPacket(s2)
                regch = handlePacket(packet)
                try:
                    if isinstance(regch.TSAM.lastRawSAM, type(
                            [])) and len(
                            regch.TSAM.lastRawSAM) == 256:
                        data = regch.TSAM.lastRawSAM
                except BaseException:
                    pass

                try:
                    s, s2 = _get_s2parse(s)
                except BaseException:
                    break
        if i == 0:
            if CalIrrUp is not None:
                data_autonaut["Es"] = np.array(data) * CalIrrUp
            else:
                data_autonaut["Es"] = np.array(data)
        else:
            if CalRadDown is not None:
                data_autonaut["Lw"] = np.array(data) * CalRadDown
            else:
                data_autonaut["Lw"] = np.array(data)

    data_autonaut["Rrs"] = data_autonaut["Lw"] / data_autonaut["Es"]
    data_autonaut["wl"] = wavelengths

    return data_autonaut


def process_incoming_csv(filepath: str, verbose=False) -> OnWaterRadiometrySet:
    """ Process incoming CSV file.

    Parameters:
    -----------
    filepath : str
        Path to the CSV file.
        Assuming the following columns: Date, Time, Lat, Lon Rrs 1, Rrs 2, ..., Rrs N

    Returns:
    --------
    OnWaterRadiometrySet:
        A set of samples as an OnWaterRadiometrySet object.
    """
    import datetime as dt
    # Initialize the dataset
    Set = OnWaterRadiometrySet()

    if verbose:
        print(f"Processing file: {filepath}")
        Set.set_verbose(True)

    # Read the file into a DataFrame
    if filepath.endswith(".xlsx"):
        df = pd.read_excel(filepath)
    elif filepath.endswith(".csv"):
        df = pd.read_csv(filepath)

    # Extract wavelength values from column names
    wl = np.array([float(c) for c in df.columns if str(c).isdigit()])

    # Identify latitude and longitude columns
    lat_column = [c for c in df.columns if "lat" in str(c).lower()]
    lon_column = [c for c in df.columns if "lon" in str(c).lower()]

    # Process each row in the DataFrame
    for i, row in df.iterrows():
        if filepath.endswith(".csv"):
            date_col = df.columns[0]
            time_col = df.columns[1]
            time_str = f"{int(row[date_col])} {int(row[time_col])}"
            now = dt.datetime.strptime(time_str, "%Y%m%d %H%M")
        elif filepath.endswith(".xlsx"):
            year = int(row["Year"])
            month = int(row["Month"])
            day = int(row["Day"])
            time_parts = list(map(int, str(row["Time"]).split(":")))
            now = dt.datetime(year, month, day, *time_parts)

        # Prepare reflectance data
        reflectance_data = {
            "Rrs": np.array([row[c] for c in df.columns if str(c).isdigit()]),
            "wl": wl
        }

        # Create a sample and set its location
        sample = OnWaterRadiometrySample(i, now, (-1, -1), reflectance_data)
        sample.set_location(row[lat_column], row[lon_column])
        sample.compute_iop_data()
        sample.analyze_sample()
        Set.add(sample)

    return Set


def write_to_excel(inPath, outPath, verbose=False):
    """ Write the samples to an Excel file.

    Parameters:
    -----------
    inPath : str
        The path to the input CSV file.
    outPath : str
        The path to the output Excel file.
    """

    if verbose:
        print("Increased verbosity is not implemented yet.")

    Set = process_incoming_csv(inPath, verbose)
    Set.write_to_excel(outPath)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process incoming CSV file.")

    parser.add_argument(
        "-i",
        dest="inPath",
        type=str,
        help="Path to the input CSV file.")
    parser.add_argument(
        "-o",
        dest="outPath",
        type=str,
        help="Path to the output Excel file.")
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Increase output verbosity.")

    args = parser.parse_args()

    # if no args are passed, print help
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    # check if the input file exists
    if not os.path.exists(args.inPath):
        print("Input file does not exist.")
        sys.exit(1)

    # check if the output is xlsx
    if not args.outPath.endswith(".xlsx"):
        # add xlsx extension
        args.outPath += ".xlsx"

    # check if output file folder path exists, if not create it
    outFolder = os.path.dirname(args.outPath)
    if outFolder == "":
        outFolder = "."
    elif not os.path.exists(outFolder):
        os.makedirs(outFolder)

    write_to_excel(args.inPath, args.outPath, args.verbose)
