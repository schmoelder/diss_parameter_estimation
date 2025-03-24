import csv
from typing import Optional

import numpy as np

from CADETProcess.dataStructure import get_nested_value
from CADETProcess.processModel import ComponentSystem
from CADETProcess.processModel import BindingBaseClass, NoBinding
from CADETProcess.processModel import (
    Inlet, Outlet, Cstr, TubularReactor, ChromatographicColumnBase
)
from CADETProcess.processModel import FlowSheet
from CADETProcess.processModel import Process
from CADETProcess.simulator import Cadet
from CADETProcess.solution import slice_solution
from CADETProcess.reference import ReferenceIO


# %% Flow Sheets

class KnauerSystem(FlowSheet):
    def __init__(
        self,
        component_system: ComponentSystem,
        sample_loop_volume: float,
        sample_loop_id: float,
        ColumnModel: Optional[ChromatographicColumnBase] = None,
        BindingModel: Optional[BindingBaseClass] = None,
        bypass_units: Optional[list[str]] = None,
        *args, **kwargs
    ) -> None:

        super().__init__(component_system, *args, **kwargs)

        # Pre-injection Unit Operations
        buffer_a = Inlet(component_system, name="buffer_a")
        self.add_unit(buffer_a)

        buffer_b = Inlet(component_system, name="buffer_b")
        self.add_unit(buffer_b)

        mixer = Cstr(component_system, "mixer")
        self.add_unit(mixer)

        tubing_pre_injection = TubularReactor(
            component_system, name="tubing_pre_injection"
        )
        self.add_unit(tubing_pre_injection)

        sample_loop = TubularReactor(
            component_system, name="sample_loop"
        )
        sample_loop.diameter = sample_loop_id
        sample_loop.length = sample_loop_volume / sample_loop.cross_section_area
        sample_loop.axial_dispersion = 0
        self.add_unit(sample_loop)

        # Post-injection Unit Operations
        if bypass_units is None:
            bypass_units = set()
        else:
            bypass_units = set(bypass_units)

        first_unit = None

        if "tubing_pre_column" not in bypass_units:
            tubing_pre_column = TubularReactor(
                component_system, name="tubing_pre_column"
            )
            self.add_unit(tubing_pre_column)
            if first_unit is None:
                first_unit = tubing_pre_column
        else:
            bypass_units.remove("tubing_pre_column")

        if "column" not in bypass_units:
            if ColumnModel is None:
                raise ValueError(
                    "ColumnModel needs to be specified if column is not bypassed."
                )
            column = ColumnModel(component_system, "column")
            if BindingModel is not None:
                column.binding_model = BindingModel(component_system)
            self.add_unit(column)
            if first_unit is None:
                first_unit = column
        else:
            bypass_units.remove("column")

        if "tubing_post_column" not in bypass_units:
            tubing_post_column = TubularReactor(
                component_system, name="tubing_post_column"
            )
            self.add_unit(tubing_post_column)
            if first_unit is None:
                first_unit = tubing_post_column
        else:
            bypass_units.remove("tubing_post_column")

        if "tubing_detectors" not in bypass_units:
            tubing_detectors = TubularReactor(
                component_system, name="tubing_detectors"
            )
            self.add_unit(tubing_detectors)
            if first_unit is None:
                first_unit = tubing_detectors
        else:
            bypass_units.remove("tubing_detectors")

        if len(bypass_units) != 0:
            raise ValueError(f"Unexpected bypass unit(s): {bypass_units}")

        outlet = Outlet(component_system, name="outlet")
        self.add_unit(outlet)
        if first_unit is None:
            first_unit = outlet

        # Pre-injection connections
        self.add_connection(buffer_a, mixer)
        self.add_connection(buffer_b, mixer)

        self.add_connection(mixer, tubing_pre_injection)

        self.add_connection(tubing_pre_injection, sample_loop)
        self.add_connection(tubing_pre_injection, first_unit)
        self.set_output_state(tubing_pre_injection, {"sample_loop": 1})
        self.add_connection(sample_loop, first_unit)

        # Post-injection connections
        connection_order = [
            "tubing_pre_column",
            "column",
            "tubing_post_column",
            "tubing_detectors",
            "outlet"
        ]
        origin = None
        for destination in connection_order:
            if origin is None:
                if destination == first_unit.name:
                    origin = first_unit.name
                continue
            if destination in self:  # Only use non-bypassed units
                self.add_connection(self[origin], self[destination])
                origin = destination  # Update the last valid unit

    def get_system_dead_volume(self, exclude=None, ignore_missing=False):
        if exclude is None:
            exclude = []

        def _get_unit_volume(name, ignore_missing=True):
            try:
                unit = self.units_dict[name]
            except KeyError:
                if not ignore_missing:
                    raise KeyError(f"Cannot find unit: {name}.")

            return unit.volume

        system_units = [
                "mixer",
                "tubing_pre_injection",
                "tubing_post_column",
                "tubing_detectors",
            ]

        return sum([
            _get_unit_volume(unit, ignore_missing)
            for unit in system_units
            if unit not in exclude
        ])

    @property
    def system_dead_volume(self) -> float:
        """float: The total system dead volume."""
        return self.get_system_dead_volume(ignore_missing=True)


# %% Processes

class KnauerSystemProcess(Process):
    """Base class for Knauer System processes."""

    def __init__(
        self,
        name: str,
        component_system: ComponentSystem,
        **knauer_system_options: dict,
    ) -> None:
        flow_sheet = KnauerSystem(component_system, **knauer_system_options)
        super().__init__(flow_sheet, name)

    def generate_synthetic_data(
        self,
        solution_path: str = "outlet.outlet",
        components: Optional[list[str]] = None,
    ) -> ReferenceIO:
        """
        Generate synthetic data to test calibration.

        Parameters
        ----------
        solution_path : str, optional
            Path to the solution in the simulation results.
            The default is "outlet.outlet".
        components : Optional[list[str]], optional
            Components used for parameter estimation. The default is None.

        Returns
        -------
        solution : ReferenceIO
            The simulation results.

        """
        simulator = Cadet()
        simulation_results = simulator.simulate(self)
        solution = get_nested_value(simulation_results.solution, solution_path)
        solution = slice_solution(
            solution,
            components=components,
        )

        return solution


class PulseInjection(KnauerSystemProcess):
    """
    Process with pulse injection on Knauer System.

    Assumes that system is equilibrated with Buffer A and a constant flow rate.
    """

    def __init__(
        self,
        name: str,
        component_system: ComponentSystem,
        knauer_system_options: dict,
        c_buffer_a: list[float],
        c_sample: list[float],
        cycle_time: float,
        flow_rate: float,
        *args, **kwargs
    ) -> None:

        super().__init__(name, component_system, **knauer_system_options)

        self.cycle_time = cycle_time

        self.flow_sheet.buffer_a.flow_rate = flow_rate
        self.flow_sheet.buffer_b.flow_rate = 0

        for unit in self.flow_sheet.units:
            if "c" in unit.parameters:
                unit.c = c_buffer_a
            if "cp" in unit.parameters:
                unit.cp = c_buffer_a

        self.flow_sheet.sample_loop.c = c_sample


class Step(KnauerSystemProcess):
    """
    Process with pulse injection on Knauer System.

    Assumes that system is equilibrated with Buffer A and switches to Buffer B at t=0.
    Moreover, a constant flow rate is assumed.
    """

    def __init__(
        self,
        name: str,
        component_system: ComponentSystem,
        knauer_system_options: dict,
        c_buffer_a: list[float],
        c_buffer_b: list[float],
        cycle_time: float,
        flow_rate: float,
    ) -> None:

        super().__init__(name, component_system, **knauer_system_options)

        self.cycle_time = cycle_time

        self.flow_sheet.buffer_a.flow_rate = 0
        self.flow_sheet.buffer_b.flow_rate = flow_rate

        for unit in self.flow_sheet.units:
            if "c" not in unit.parameters:
                continue
            unit.c = c_buffer_a

        self.flow_sheet.buffer_b.c = c_buffer_b


class LWE(KnauerSystemProcess):
    """
    Load / Wash / Elute Process on Knauer System.

    Assumes that system is equilibrated with Buffer A. After the wash phase, a linear
    gradient is performed, switching to Buffer B.
    """

    def __init__(
        self,
        name: str,
        component_system: ComponentSystem,
        knauer_system_options: dict,
        c_buffer_a: list[float],
        c_buffer_b: list[float],
        c_sample: list[float],
        is_kinetic: bool,
        delta_t_wash: float,
        delta_t_elute: float,
        delta_t_final_wash: float,
        flow_rate_wash: float,
        flow_rate_elute: Optional[float] = None,
        flow_rate_final_wash: Optional[float] = None,
    ) -> None:

        self.delta_t_wash = delta_t_wash
        self.delta_t_elute = delta_t_elute
        self.delta_t_final_wash = delta_t_final_wash

        super().__init__(name, component_system, **knauer_system_options)

        self.flow_sheet.column.binding_model.is_kinetic = is_kinetic

        if flow_rate_elute is None:
            flow_rate_elute = flow_rate_wash
        if flow_rate_final_wash is None:
            flow_rate_final_wash = flow_rate_elute

        t_wash_start = 0
        t_wash_end = np.round(t_wash_start + delta_t_wash, 2)

        t_elute_start = t_wash_end
        t_elute_end = np.round(t_elute_start + delta_t_elute, 2)

        t_final_wash_start = t_elute_end
        t_final_wash_end = np.round(t_final_wash_start + delta_t_final_wash, 2)

        cycle_time = t_final_wash_end
        self.cycle_time = cycle_time

        for unit in self.flow_sheet.units:
            if "c" in unit.parameters:
                unit.c = c_buffer_a
            if "cp" in unit.parameters:
                unit.cp = c_buffer_a

        self.flow_sheet.buffer_b.c = c_buffer_b
        self.flow_sheet.sample_loop.c = c_sample

        # Load / Wash
        self.add_event(
            'wash_start_low',
            'flow_sheet.buffer_a.flow_rate',
            flow_rate_wash,
            0,
        )
        self.add_event(
            'wash_start_high',
            'flow_sheet.buffer_b.flow_rate',
            0,
            0,
        )

        # Elute
        self.add_event(
            'elute_start_low',
            'flow_sheet.buffer_a.flow_rate',
            [flow_rate_elute, -flow_rate_elute / delta_t_elute],
            t_elute_start,
        )
        self.add_event(
            'elute_start_high',
            'flow_sheet.buffer_b.flow_rate',
            [0, flow_rate_elute / delta_t_elute],
            t_elute_start,
        )

        # Final Wash
        self.add_event(
            'wash_final_start_low',
            'flow_sheet.buffer_a.flow_rate',
            0,
            t_final_wash_start,
        )
        self.add_event(
            'wash_final_start_high',
            'flow_sheet.buffer_b.flow_rate',
            flow_rate_final_wash,
            t_final_wash_start,
        )


# %% Experimental Data


def load_knauer_data(file_path):
    """
    Load experimental data from a Knauer System.

    The first 6 lines are metadata (key/value pairs). The header row is on line 7,
    and the data rows start from line 8. The returned data is indexed by column,
    mapping each column header to a list of values.

    Parameters
    ----------
    file_path : str
        Path to the CSV file.

    Returns
    -------
    metadata : dict
        Dictionary containing metadata extracted from the first 6 lines.
    data : dict
        Dictionary mapping each column header to a list of values from that column.

    Examples
    --------
    >>> meta, data = load_knauer_data("Aceton_no_column_short_tube_0002.rfp")
    >>> print(meta["FileName"])
    Aceton_no_column_short_tube_0002.rfp
    >>> print(data["UV Channel 1 [mAU]"])
    ['-4.067', '-4.067', '-3.361', '-2.402', ...]
    """
    metadata = {}
    data = {}
    delimiter = ";"

    with open(file_path, newline="", encoding='latin-1') as csvfile:
        # Process the first 6 lines for metadata.
        for _ in range(6):
            line = next(csvfile).strip()
            if not line:
                continue
            parts = line.split(delimiter)
            key = parts[0].strip()
            value = parts[1].strip() if len(parts) > 1 else ""
            if key:
                metadata[key] = value

        # Read the header row (line 7) and parse column names.
        header_line = next(csvfile).strip()
        header = [col.strip() for col in header_line.split(delimiter)]

        # Initialize the data dictionary with an empty list for each column.
        for col in header:
            data[col] = []

        # Process each subsequent data row.
        reader = csv.reader(csvfile, delimiter=delimiter)
        for row in reader:
            # Skip empty rows.
            if not row or all(not cell.strip() for cell in row):
                continue
            for col, cell in zip(header, row):
                stripped = cell.strip()
                if stripped:  # Only append if the cell is not empty.
                    data[col].append(stripped)

    return metadata, data


class KnauerExperimentalData:
    """
    Represent experimental data from a Knauer System.

    Load the experimental data from a CSV file and store important metadata. For each
    data column, create a ReferenceIO object representing a time series.

    Attributes
    ----------
    file_name : str
        File name from the metadata.
    flow_rate : float
        The flow rate of the experiment.
    sample : str
        Sample name from the metadata.
    date : str
        Date from the metadata.
    data_points : str
        Data points information from the metadata.
    slice_width : str
        Slice width information from the metadata.
    time_offset : float
        Time offset to account for delayed injection. Default is 0.0.
    duration : float, optional
        Duration of the experiment in s.
    uv_1 : ReferenceIO
        Time series object for 'UV Channel 1 [mAU]'.
    uv_2 : ReferenceIO
        Time series object for 'UV Channel 2 [mAU]'.
    uv_3 : ReferenceIO
        Time series object for 'UV Channel 3 [mAU]'.
    uv_4 : ReferenceIO
        Time series object for 'UVChannel 4 [mAU]'.
    conductivity : ReferenceIO
        Time series object for 'Conductivity [mS/cm]'.
    pressure : ReferenceIO
        Time series object for 'Pressure Channel [Bar]'.
    solvent_a : ReferenceIO
        Time series object for 'Solvent A'.
    solvent_b : ReferenceIO
        Time series object for 'Solvent B'.
    solvent_c : ReferenceIO
        Time series object for 'Solvent C'.
    solvent_d : ReferenceIO
        Time series object for 'Solvent D'.
    """

    def __init__(
        self,
        file_path: str,
        flow_rate: float,
        time_offset: Optional[float] = 0.0,
        duration: Optional[float] = -1,
    ):
        """
        Initialize KnauerExperimentalData by loading data from a CSV file.

        Parameters
        ----------
        file_path : str
            Path to the CSV file containing the experimental data.
        flow_rate : float
            The flow rate of the experiment.
        time_offset : float, optional
            Time offset to account for delayed injection. Default is 0.0.
        duration : float, optional
            Duration of the experiment in s. Default is -1, indicating that all data
            points are to be used. Note, this refers to the duration after applying the
            time_offset.
        """
        self.metadata, self.data = load_knauer_data(file_path)
        self.file_name = self.metadata.get("FileName")
        self.sample = self.metadata.get("Sample")
        self.date = self.metadata.get("Date")
        self.data_points = self.metadata.get("Data Points")
        self.slice_width = self.metadata.get("SliceWidth")

        self.flow_rate = flow_rate
        self.time_offset = time_offset
        if duration == -1:
            duration = np.inf
        self.duration = duration

        # Create ReferenceIO objects using the internal helper method
        self.solvent_a = self._create_reference_io(
            "Solvent A",
            alias="solvent_a",
        )
        self.solvent_b = self._create_reference_io(
            "Solvent B",
            alias="solvent_b",
        )
        self.solvent_c = self._create_reference_io(
            "Solvent C",
            alias="solvent_c",
        )
        self.solvent_d = self._create_reference_io(
            "Solvent D",
            alias="solvent_d",
        )

        self.uv_1 = self._create_reference_io(
            "UV Channel 1 [mAU]",
            alias="uv_1"
        )
        self.uv_2 = self._create_reference_io(
            "UV Channel 2 [mAU]",
            alias="uv_2"
        )
        self.uv_3 = self._create_reference_io(
            "UV Channel 3 [mAU]",
            alias="uv_3"
        )
        self.uv_4 = self._create_reference_io(
            "UVChannel 4 [mAU]",
            alias="uv_4"
        )

        self.conductivity = self._create_reference_io(
            "Conductivity [mS/cm]",
            alias="conductivity"
        )
        self.pressure = self._create_reference_io(
            "Pressure Channel [Bar]",
            alias="pressure"
        )

    def _create_reference_io(
        self,
        column_name: str,
        alias: Optional[str] = None,
    ) -> Optional[ReferenceIO]:
        """
        Internal helper method to create ReferenceIO objects and handle calibration.

        Parameters
        ----------
        column_name : str
            The name of the data column in the loaded dataset.
        alias : Optional[str]
            Alternative name for the reference. Defaults to None.

        Returns
        -------
        Optional[ReferenceIO]
            The created ReferenceIO object or None if data is not available.
        """
        data_array = np.array(
            self.data.get(column_name, []), dtype=float
        ).reshape(-1, 1)

        if len(data_array) == 0:
            return None

        # Clean up time vector and remove negative values
        time_min = np.array(self.data.get("Time [Min]", []), dtype=float)
        time = time_min * 60 - self.time_offset
        time = time[:len(data_array)]

        # Remove negative time values and synchronize concentration array
        valid_indices = np.where((time >= 0) & (time <= self.duration))
        time = time[valid_indices]
        data_array = data_array[valid_indices]

        name = alias if alias else column_name

        reference_io = ReferenceIO(name, time, data_array, self.flow_rate)

        return reference_io
