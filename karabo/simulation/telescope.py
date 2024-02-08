from __future__ import annotations

import enum
import glob
import logging
import os
import re
import shutil
from itertools import product
from typing import Dict, List, Literal, Optional, Tuple, Type, Union, cast, get_args

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from oskar.telescope import Telescope as OskarTelescope
from rascil.processing_components.simulation.simulation_helpers import (
    plot_configuration,
)
from ska_sdp_datamodels.configuration.config_create import create_named_configuration
from ska_sdp_datamodels.configuration.config_model import Configuration

import karabo.error
from karabo.error import KaraboError
from karabo.simulation.coordinate_helper import east_north_to_long_lat
from karabo.simulation.east_north_coordinate import EastNorthCoordinate
from karabo.simulation.station import Station
from karabo.simulation.telescope_versions import (
    ACAVersions,
    ALMAVersions,
    ATCAVersions,
    CARMAVersions,
    NGVLAVersions,
    PDBIVersions,
    SMAVersions,
    VLAVersions,
)
from karabo.simulator_backend import SimulatorBackend
from karabo.util._types import DirPathType, NPFloatLike
from karabo.util.data_util import get_module_absolute_path
from karabo.util.file_handler import FileHandler
from karabo.util.math_util import long_lat_to_cartesian

OSKARTelescopesWithVersionType = Literal[
    "ACA",
    "ALMA",
    "ATCA",
    "CARMA",
    "NGVLA",
    "PDBI",
    "SMA",
    "VLA",
]
OSKARTelescopesWithoutVersionType = Literal[
    "EXAMPLE",
    "MeerKAT",
    "ASKAP",
    "LOFAR",
    "MKATPlus",
    "PDBI",
    "SKA1LOW",
    "SKA1MID",
    "VLBA",
    "WSRT",
]

OSKAR_TELESCOPE_TO_FILENAMES: Dict[
    Union[OSKARTelescopesWithVersionType, OSKARTelescopesWithoutVersionType],
    str,
] = {
    "EXAMPLE": "telescope.tm",
    "MeerKAT": "meerkat.tm",
    "ACA": "aca.{0}.tm",
    "ALMA": "alma.{0}.tm",
    "ASKAP": "askap.tm",
    "ATCA": "atca.{0}.tm",
    "CARMA": "carma.{0}.tm",
    "LOFAR": "lofar.tm",
    "MKATPlus": "mkatplus.tm",
    "NGVLA": "ngvla-{0}.tm",
    "PDBI": "pdbi-{0}.tm",
    "SKA1LOW": "ska1low.tm",
    "SKA1MID": "ska1mid.tm",
    "SMA": "sma.{0}.tm",
    "VLA": "vla.{0}.tm",
    "VLBA": "vlba.tm",
    "WSRT": "WSRT.tm",
}
OSKAR_TELESCOPE_TO_VERSIONS: Dict[OSKARTelescopesWithVersionType, Type[enum.Enum]] = {
    "ACA": ACAVersions,
    "ALMA": ALMAVersions,
    "ATCA": ATCAVersions,
    "CARMA": CARMAVersions,
    "NGVLA": NGVLAVersions,
    "PDBI": PDBIVersions,
    "SMA": SMAVersions,
    "VLA": VLAVersions,
}


class Telescope:
    """Telescope

    WGS84 longitude and latitude and altitude in metres centre of the telescope.png
    centre. A telescope is described as follows:

    Each row represents one station, with the elements being the horizontal x (east),
    horizontal y (north), and horizontal z (up) coordinates,
    followed by the errors in horizontal y (east), horizontal y (north),
    and horizontal z (up).
    Example: [[x, y, z, error_x, error_y, error_z], [...]]

    centre_longitude : float
        WGS84 longitude at the center of the telescope.
    centre_latitude : float
        WGS84 latitude at the center of the telescope.
    centre_altitude : float
        Altitude (in meters) at the center of the telescope.
    """

    def __init__(
        self, longitude: float, latitude: float, altitude: float = 0.0
    ) -> None:
        """__init__ method

        Parameters
        ----------
        longitude : float
            WGS84 longitude at the center of the telescope.
        latitude : float
            WGS84 latitude at the center of the telescope.
        altitude : float, optional
            Altitude (in meters) at the center of the telescope, default is 0.
        """
        self.path: Optional[DirPathType] = None
        self.centre_longitude = longitude
        self.centre_latitude = latitude
        self.centre_altitude = altitude

        self.stations: List[Station] = []

        self.backend: SimulatorBackend = SimulatorBackend.OSKAR

        self.RASCIL_configuration: Optional[Configuration] = None

    @classmethod
    def constructor(
        cls,
        name: Union[OSKARTelescopesWithVersionType, OSKARTelescopesWithoutVersionType],
        version: Optional[enum.Enum] = None,
        backend: SimulatorBackend = SimulatorBackend.OSKAR,
    ) -> Telescope:
        """Main constructor to obtain a pre-configured telescope instance.
        :param name: Name of the desired telescope configuration.
            This name, together with the backend, is used as the key
            to look up the correct telescope specification file.
        :param version: Version details required for some
            telescope configurations. Defaults to None.
        :param backend: Underlying package to be used for the telescope configuration,
            since each package stores the arrays in a different format.
            Defaults to OSKAR.
        :raises: ValueError if the combination of input parameters is invalid.
            Specifically, if the requested telescope requires a version,
            but an invalid version (or no version) is provided,
            or if the requested telescope name is not
            supported by the requested backend.
        :returns: Telescope instance.
        """
        if backend is SimulatorBackend.OSKAR:
            # Verify if requested telescope has an existing configuration
            datapath = OSKAR_TELESCOPE_TO_FILENAMES.get(name, None)
            if datapath is None:
                raise ValueError(
                    f"""{name} not supported for backend {SimulatorBackend.OSKAR.value}.
                    The valid options for name are:
                    {list(OSKAR_TELESCOPE_TO_FILENAMES.keys())}."""
                )

            # Explicitly cast name depending on whether it requires a telescope version
            # This should no longer be necessary when mypy starts supporting
            # type narrowing with get_args.
            # https://github.com/python/mypy/issues/12535
            if name in get_args(OSKARTelescopesWithVersionType):
                name = cast(OSKARTelescopesWithVersionType, name)
                accepted_versions = OSKAR_TELESCOPE_TO_VERSIONS[name]
                if (version is None) or (version not in accepted_versions):
                    raise ValueError(
                        f"""{version} is not valid for telescope {name}.
                        List of valid versions: {accepted_versions}."""
                    )
                datapath = datapath.format(version.value)
            else:
                if version is not None:
                    raise ValueError(
                        f"""version is not a required field for telescope {name},
                    but {version} was provided.
                    Please do not provide a value for the version field."""
                    )

            path = os.path.join(get_module_absolute_path(), "data", datapath)
            return cls.read_OSKAR_tm_file(path)
        elif backend is SimulatorBackend.RASCIL:
            if version is not None:
                logging.warning(
                    f"""The version parameter is not supported
    by the backend {backend}.
    The version value {version} provided will be ignored."""
                )
            try:
                configuration = create_named_configuration(name)
            except ValueError:
                raise ValueError(
                    f"""Requested telescope {name} is not supported by this backend.
                    For more details, see
    https://gitlab.com/ska-telescope/sdp/ska-sdp-datamodels/-/blob/d6dcce6288a7bf6d9ce63ab16e799977723e7ae5/src/ska_sdp_datamodels/configuration/config_create.py"""  # noqa
                )

            config_earth_location = configuration.location
            telescope = Telescope(
                longitude=config_earth_location.lon.to("deg").value,
                latitude=config_earth_location.lat.to("deg").value,
                altitude=config_earth_location.height.to("m").value,
            )
            telescope.backend = SimulatorBackend.RASCIL
            telescope.RASCIL_configuration = configuration

            return telescope
        else:
            raise ValueError(
                f"{backend} not supported, see valid options within SimulatorBackend."
            )

    def get_backend_specific_information(self) -> Union[DirPathType, Configuration]:
        if self.backend is SimulatorBackend.OSKAR:
            return self.path
        if self.backend is SimulatorBackend.RASCIL:
            return self.RASCIL_configuration

        raise ValueError(
            f"""Unexpected: current backend is set to {self.backend},
        but expected one of {SimulatorBackend}.
        Verify the construction of this Telescope instance."""
        )

    def add_station(
        self,
        horizontal_x: float,
        horizontal_y: float,
        horizontal_z: float = 0.0,
        horizontal_x_coordinate_error: float = 0.0,
        horizontal_y_coordinate_error: float = 0.0,
        horizontal_z_coordinate_error: float = 0.0,
    ) -> None:
        """
        Specify the stations as relative to the centre position
        :param horizontal_x: east coordinate relative to centre
        :param horizontal_y: north coordinate relative to centre
        :param horizontal_z: up coordinate
        :param horizontal_x_coordinate_error: east coordinate error
        :param horizontal_y_coordinate_error: north coordinate error
        :param horizontal_z_coordinate_error: up coordinate error
        """
        self.stations.append(
            Station(
                EastNorthCoordinate(
                    horizontal_x,
                    horizontal_y,
                    horizontal_z,
                    horizontal_x_coordinate_error,
                    horizontal_y_coordinate_error,
                    horizontal_z_coordinate_error,
                ),
                self.centre_longitude,
                self.centre_latitude,
                self.centre_altitude,
            )
        )

    def add_antenna_to_station(
        self,
        station_index: int,
        horizontal_x: float,
        horizontal_y: float,
        horizontal_z: float = 0,
        horizontal_x_coordinate_error: float = 0,
        horizontal_y_coordinate_error: float = 0,
        horizontal_z_coordinate_error: float = 0,
    ) -> None:
        """
        Add a new antenna to an existing station

        :param station_index: Index of station to add antenna to
        :param horizontal_x: east coordinate relative to the station center in metres
        :param horizontal_y: north coordinate relative to the station center in metres
        :param horizontal_z: altitude of antenna
        :param horizontal_x_coordinate_error: east coordinate error
        relative to the station center in metres
        :param horizontal_y_coordinate_error: north coordinate error
        relative to the station center in metres
        :param horizontal_z_coordinate_error: altitude of antenna error
        :return:
        """
        if station_index < len(self.stations):
            station = self.stations[station_index]
            station.add_station_antenna(
                EastNorthCoordinate(
                    horizontal_x,
                    horizontal_y,
                    horizontal_z,
                    horizontal_x_coordinate_error,
                    horizontal_y_coordinate_error,
                    horizontal_z_coordinate_error,
                )
            )

    def plot_telescope(self, file: Optional[str] = None) -> None:
        """
        Plot the telescope according to which backend is being used,
        and save the resulting image into a file, if any is provided.
        """
        if self.backend is SimulatorBackend.OSKAR:
            self.plot_telescope_OSKAR(file)
        elif self.backend is SimulatorBackend.RASCIL:
            plot_configuration(self.get_backend_specific_information())
        else:
            logging.warning(
                f"""Backend {self.backend} is not valid.
            Proceeding without any further actions."""
            )
            return

    def plot_telescope_OSKAR(self, file: Optional[str] = None) -> None:
        """
        Plot the telescope and all its stations and antennas with longitude altitude
        """
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        antenna_x = []
        antenna_y = []
        station_x = []
        station_y = []
        for station in self.stations:
            station_x.append(station.longitude)
            station_y.append(station.latitude)

            for antenna in station.antennas:
                long, lat = east_north_to_long_lat(
                    antenna.x, antenna.y, station.longitude, station.latitude
                )
                antenna_x.append(long)
                antenna_y.append(lat)

        ax.scatter(antenna_x, antenna_y, label="Antennas")
        ax.scatter(station_x, station_y, label="Stations")

        x = np.array([self.centre_longitude])
        y = np.array([self.centre_latitude])

        ax.scatter(x, y, label="Centre")
        ax.ticklabel_format(useOffset=False)
        ax.set_xlabel("Longitude [deg]")
        ax.set_ylabel("Latitude [deg]")
        ax.set_title("Antenna Locations")
        ax.legend(loc="upper left", shadow=False, fontsize="medium")

        if file is not None:
            plt.savefig(file)
            plt.close(fig)
        else:
            plt.show(block=False)
            plt.pause(1)

    def get_OSKAR_telescope(self) -> OskarTelescope:
        """
        Retrieve the OSKAR Telescope object from the karabo.Telescope object.

        Note: Once this function is called, it returns the same `OskarTelescope`
            for each function call bound to this object-instance. Thus, changing
            Telescope-parameters on this instance after calling this function
            won't have an effect on the returned `OskarTelescope` anymore.

        :return: OSKAR Telescope object
        """
        tmp_dir = FileHandler().get_tmp_dir(
            prefix="telescope-",
            purpose="telescope disk-cache",
            unique=self,
            mkdir=False,
        )
        tmp_dir = os.path.join(tmp_dir, "oskar-telescope")
        os.makedirs(tmp_dir, exist_ok=True)
        if not FileHandler.is_dir_empty(dirname=tmp_dir):
            FileHandler.empty_dir(dir_path=tmp_dir)
        self.write_to_file(tmp_dir)
        tel = OskarTelescope()
        tel.load(tmp_dir)
        self.path = tmp_dir
        return tel

    def write_to_file(self, dir: DirPathType) -> None:
        """
        Create .tm telescope configuration at the specified path
        :param dir: directory in which the configuration will be saved in.
        """
        self.__write_position_txt(os.path.join(dir, "position.txt"))
        self.__write_layout_txt(
            os.path.join(dir, "layout.txt"),
            [station.position for station in self.stations],
        )
        for i, station in enumerate(self.stations):
            station_path = f"{dir}{os.path.sep}station{'{:03d}'.format(i)}"
            os.mkdir(station_path)
            self.__write_layout_txt(
                os.path.join(station_path, "layout.txt"),
                station.antennas,
            )

    def __write_position_txt(self, position_file_path: str) -> None:
        position_file = open(position_file_path, "a")
        position_file.write(
            f"{self.centre_longitude} {self.centre_latitude} {self.centre_altitude} \n"
        )
        position_file.close()

    def __write_layout_txt(
        self, layout_path: str, elements: List[EastNorthCoordinate]
    ) -> None:
        layout_file = open(layout_path, "a")
        for element in elements:
            layout_file.write(
                f"{element.x}, {element.y}, {element.z}, {element.x_error}, "
                + f"{element.y_error}, {element.z_error} \n"
            )
        layout_file.close()

    def get_cartesian_position(self) -> NDArray[np.float_]:
        return long_lat_to_cartesian(self.centre_latitude, self.centre_longitude)

    @classmethod
    def read_from_file(cls, path: str) -> Optional[Telescope]:
        raise DeprecationWarning("Use Telescope.read_OSKAR_tm_file(path) instead.")

    @classmethod
    def get_MEERKAT_Telescope(cls) -> Telescope:
        raise DeprecationWarning('Use Telescope.constructor("MeerKAT") instead.')

    @classmethod
    def get_ACA_Telescope(cls, version: ACAVersions) -> Telescope:
        raise DeprecationWarning('Use Telescope.constructor("ACA", version) instead.')

    @classmethod
    def get_ALMA_Telescope(cls, version: ALMAVersions) -> Telescope:
        raise DeprecationWarning('Use Telescope.constructor("ALMA", version) instead.')

    @classmethod
    def get_ASKAP_Telescope(cls) -> Telescope:
        raise DeprecationWarning('Use Telescope.constructor("ASKAP") instead.')

    @classmethod
    def get_ATCA_Telescope(cls, version: ATCAVersions) -> Telescope:
        raise DeprecationWarning('Use Telescope.constructor("ATCA", version) instead.')

    @classmethod
    def get_CARMA_Telescope(cls, version: CARMAVersions) -> Telescope:
        raise DeprecationWarning('Use Telescope.constructor("CARMA", version) instead.')

    @classmethod
    def get_LOFAR_Telescope(cls) -> Telescope:
        raise DeprecationWarning('Use Telescope.constructor("LOFAR") instead.')

    @classmethod
    def get_MKATPLUS_Telescope(cls) -> Telescope:
        raise DeprecationWarning('Use Telescope.constructor("MKATPlus") instead.')

    @classmethod
    def get_NG_VLA_Telescope(cls, version: NGVLAVersions) -> Telescope:
        raise DeprecationWarning('Use Telescope.constructor("NGVLA", version) instead.')

    @classmethod
    def get_PDBI_Telescope(cls, version: PDBIVersions) -> Telescope:
        raise DeprecationWarning('Use Telescope.constructor("PDBI", version) instead.')

    @classmethod
    def get_SKA1_LOW_Telescope(cls) -> Telescope:
        raise DeprecationWarning('Use Telescope.constructor("SKA1LOW") instead.')

    @classmethod
    def get_SKA1_MID_Telescope(cls) -> Telescope:
        raise DeprecationWarning('Use Telescope.constructor("SKA1MID") instead.')

    @classmethod
    def get_SMA_Telescope(cls, version: SMAVersions) -> Telescope:
        raise DeprecationWarning('Use Telescope.constructor("SMA", version) instead.')

    @classmethod
    def get_VLA_Telescope(cls, version: VLAVersions) -> Telescope:
        raise DeprecationWarning('Use Telescope.constructor("VLA", version) instead.')

    @classmethod
    def get_VLBA_Telescope(cls) -> Telescope:
        raise DeprecationWarning('Use Telescope.constructor("VLBA") instead.')

    @classmethod
    def get_WSRT_Telescope(cls) -> Telescope:
        raise DeprecationWarning('Use Telescope.constructor("WSRT") instead.')

    @classmethod
    def get_OSKAR_Example_Telescope(cls) -> Telescope:
        raise DeprecationWarning('Use Telescope.constructor("EXAMPLE") instead.')

    @classmethod
    def read_OSKAR_tm_file(cls, path: DirPathType) -> Telescope:
        path_ = str(path)
        abs_station_dir_paths = []
        center_position_file = None
        station_layout_file = None
        for file_or_dir in os.listdir(path_):
            if file_or_dir.startswith("position"):
                center_position_file = os.path.abspath(os.path.join(path_, file_or_dir))
            if file_or_dir.startswith("layout"):
                station_layout_file = os.path.abspath(os.path.join(path_, file_or_dir))
            if file_or_dir.startswith("station"):
                abs_station_dir_paths.append(
                    os.path.abspath(os.path.join(path_, file_or_dir))
                )

        if center_position_file is None:
            raise karabo.error.KaraboError("Missing crucial position.txt file_or_dir")

        if station_layout_file is None:
            raise karabo.error.KaraboError(
                "Missing layout.txt file in station directory. "
                "Only Layout.txt is support. "
                "The layout_ecef.txt and layout_wgs84.txt as "
                "defined in the OSKAR Telescope .tm specification are not "
                "supported currently."
            )

        telescope = None

        position_file = open(center_position_file)
        lines = position_file.readlines()
        for line in lines:
            match = re.match(
                r"^\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s+([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)?\s*$",  # noqa: E501
                line.strip(),
            )  # one line with two or three numbers
            if match:
                numbers = [float(num) for num in match.group().split()]
                long = numbers[0]
                lat = numbers[1]
                alt = 0.0
                if len(numbers) == 3:
                    alt = float(numbers[2])
                telescope = Telescope(long, lat, alt)
                break

        if telescope is None:
            raise karabo.error.KaraboError(
                "Could not create Telescope from position.txt file_or_dir. "
                + "It must contain one line with two or three numbers."
            )

        position_file.close()

        station_positions = cls.__read_layout_txt(station_layout_file)
        for station_position in station_positions:
            telescope.add_station(
                station_position[0],
                station_position[1],
                station_position[2],
                station_position[3],
                station_position[4],
                station_position[5],
            )

        if len(abs_station_dir_paths) != len(telescope.stations):
            raise karabo.error.KaraboError(
                f"There are {len(telescope.stations)} stations "
                f"but {len(abs_station_dir_paths)} "
                f"station directories."
            )

        for station_dir, station in zip(abs_station_dir_paths, telescope.stations):
            antenna_positions = cls.__read_layout_txt(
                os.path.join(station_dir, "layout.txt")
            )
            for antenna_pos in antenna_positions:
                station.add_station_antenna(
                    EastNorthCoordinate(
                        antenna_pos[0],
                        antenna_pos[1],
                        antenna_pos[2],
                        antenna_pos[3],
                        antenna_pos[4],
                        antenna_pos[5],
                    )
                )

        telescope.path = path
        telescope.backend = SimulatorBackend.OSKAR
        return telescope

    @classmethod
    def __read_layout_txt(cls, path: str) -> List[List[float]]:
        positions: List[List[float]] = []
        layout_file = open(path)
        lines = layout_file.readlines()
        for line in lines:
            station_position = re.split("[\\s,]+", line)
            values = np.zeros(6)
            i = 0
            for pos in station_position:
                values[i] = cls.__float_try_parse(pos)
                i += 1
            positions.append(
                [values[0], values[1], values[2], values[3], values[4], values[5]]
            )
        layout_file.close()
        return positions

    @classmethod
    def __float_try_parse(cls, value: str) -> float:
        try:
            return float(value)
        except ValueError:
            return 0.0

    @classmethod
    def _get_station_infos(cls, tel_path: DirPathType) -> pd.DataFrame:
        """Creates a pd.DataFrame with telescope-station infos.

        - "station-nr": Station-number inside the .tm file.
        - "station-path": Path of the according station.
        - "x": x-position
        - "y": y-position

        Args:
            tel_path: .tm dir-path to get infos from.

        Returns:
            pd.DataFrame with the according infos.
        """
        station_paths = glob.glob(f"{tel_path}{os.path.sep}station[0-9]*")
        if len(station_paths) <= 0:
            raise FileNotFoundError(f"No stations found in {tel_path}")
        station_numbers = list()
        for station_path in station_paths:
            station_number = os.path.split(station_path)[-1].split("station")[1]
            station_numbers.append(int(station_number))
        df_tel = (
            pd.DataFrame(
                {
                    "station-nr": station_numbers,
                    "station-path": station_paths,
                }
            )
            .sort_values(by="station-nr")
            .reset_index(drop=True)
        )
        if not np.all(df_tel["station-nr"].to_numpy() == np.arange(0, df_tel.shape[0])):
            raise KaraboError(
                f"Stations found in {tel_path} are not ascending from station<0 - n>. "
            )
        stations = np.loadtxt(os.path.join(tel_path, "layout.txt"))
        if (n_stations_layout := stations.shape[0]) != (n_stations := df_tel.shape[0]):
            raise KaraboError(
                f"Number of stations missmatch of {n_stations_layout=} & {n_stations=}"
            )
        df_tel["x"] = stations[:, 0]
        df_tel["y"] = stations[:, 1]
        return df_tel

    @classmethod
    def create_baseline_cut_telelescope(
        cls,
        lcut: NPFloatLike,
        hcut: NPFloatLike,
        tel: Telescope,
        tm_path: Optional[DirPathType] = None,
    ) -> Tuple[DirPathType, Dict[str, str]]:
        """Cut telescope `tel` for baseline-lengths.

        Args:
            lcut: Lower cut
            hcut: Higher cut
            tel: Telescope to cut off
            tm_path: .tm file-path to save the cut-telescope.
                `tm_path` will get overwritten if it already exists.

        Returns:
            .tm file-path & station-name conversion (e.g. station055 -> station009)
        """
        if tel.path is None:
            raise KaraboError(
                "`tel.path` None indicates that there is not telescope.tm file "
                + "available for `tel`, which is not allowed here."
            )
        if tm_path is not None and not str(tm_path).endswith(".tm"):
            raise KaraboError(f"{tm_path=} must end with '.tm'.")
        df_tel = Telescope._get_station_infos(tel_path=tel.path)
        n_stations = df_tel.shape[0]
        station_x = df_tel["x"].to_numpy()
        station_y = df_tel["y"].to_numpy()
        baselines: List[Tuple[int, int]] = sorted(
            [  # each unique combination-idx a station with another station
                tuple(station_idx)  # type: ignore[misc]
                for station_idx in set(
                    map(
                        frozenset, product(np.arange(n_stations), np.arange(n_stations))
                    )
                )
                if len(station_idx) > 1
            ]
        )
        n_baselines = len(baselines)
        baseline_dist = np.zeros(n_baselines)
        for i, (x, y) in enumerate(baselines):
            baseline_dist[i] = np.linalg.norm(station_x[x] - station_y[y])
        cut_idx = np.where((baseline_dist > lcut) & (baseline_dist < hcut))
        cut_station_list = np.unique(np.array(baselines)[cut_idx])
        df_tel = df_tel[df_tel["station-nr"].isin(cut_station_list)].reset_index(
            drop=True
        )

        if cut_station_list.shape[0] == 0:
            raise KaraboError("All telescope-stations were cut off.")

        if tm_path is None:
            disk_cache = FileHandler().get_tmp_dir(
                prefix="telescope-baseline-cut-",
                mkdir=False,
            )
            tm_path = os.path.join(disk_cache, "telescope-baseline-cut.tm")
        else:
            if os.path.exists(tm_path):
                shutil.rmtree(tm_path)
        os.makedirs(tm_path, exist_ok=False)

        conversions: Dict[str, str] = dict()
        for i in range(df_tel.shape[0]):
            source_path = df_tel.iloc[i]["station-path"]
            number_str = str(i).zfill(3)
            target_station = f"station{number_str}"
            target_path = os.path.join(tm_path, target_station)
            source_station = os.path.split(source_path)[-1]
            conversions[source_station] = target_station
            shutil.copytree(src=source_path, dst=target_path)

        shutil.copyfile(
            src=os.path.join(tel.path, "position.txt"),
            dst=os.path.join(tm_path, "position.txt"),
        )
        cut_stations = df_tel[["x", "y"]].to_numpy()
        np.savetxt(os.path.join(tm_path, "layout.txt"), cut_stations)
        return tm_path, conversions
