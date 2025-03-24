"""
Common parameters.

This module contains commonly used parameters and configurations that are known before
running calibraiton experiments.

"""
import numpy as np

from CADETProcess.processModel import ComponentSystem

# %% Common parameters

sample_loop_volume = 50e-9
sample_loop_id = 0.75e-3

use_synthetic_data = True

# Acetone
acetone_molecular_weight = 58.08e-3  # kg/mol
acetone_stock_mass = 0.7652e-3   # kg
acetone_stock_volume = 0.1e-3     # m³
acetone_stock_n = acetone_stock_mass / acetone_molecular_weight     # mol
c_acetone = acetone_stock_n / acetone_stock_volume
n_sample_acetone = sample_loop_volume * c_acetone

component_system_acetone = ComponentSystem(["Acetone"])

# Salt
c_salt_low = 20  # mM
c_salt_high = 1020  # mM
n_sample_salt = sample_loop_volume * c_salt_high

component_system_salt = ComponentSystem(["Salt"])

# Blue Dextran
dextran_molecular_weight = 2000000.0e-3     # kg/mol
dextran_stock_mass = 7.3e-6  # kg
dextran_stock_n = dextran_stock_mass / dextran_molecular_weight   # mol
dextran_stock_volume = 7.3e-6    # m³
c_dextran = dextran_stock_n / dextran_stock_volume
n_sample_dextran = sample_loop_volume * c_dextran

component_system_dextran = ComponentSystem(["Blue Dextran"])

# NaOH

component_system_naoh = ComponentSystem(["NaOH"])

# Lysozyme
c_lysozyme = 0.2   # mM
n_sample_lysozyme = sample_loop_volume * c_lysozyme

component_system_lysozyme = ComponentSystem(["Salt", "Lysozyme"])

# Process parameters
time_offset = 60  # s / Offset before injection

cycle_time_system_periphery = 800   # s
cycle_time_bed = 1000   # s

flow_rate_ml_min = 0.5  # ml/min
flow_rate = flow_rate_ml_min / 60 / 1e6  # m^3 / s

column_volume = 4.7e-6  # m^3
column_length = 0.1
column_diameter = np.sqrt(4/np.pi * column_volume / column_length)

volume_wash = 2*column_volume  # m^3
delta_t_wash = volume_wash / flow_rate

volume_final_wash = 10e-6   # m^3
delta_t_final_wash = volume_final_wash / flow_rate


# %% System options

knauer_system_options = {
    "sample_loop_volume": sample_loop_volume,
    "sample_loop_id": sample_loop_id,
}
knauer_system_options_bypass_column_post_detector = {
    "sample_loop_volume": sample_loop_volume,
    "sample_loop_id": sample_loop_id,
    "bypass_units": ["column", "tubing_post_column", "tubing_detectors"],
}
knauer_system_options_bypass_column_detector = {
    "sample_loop_volume": sample_loop_volume,
    "sample_loop_id": sample_loop_id,
    "bypass_units": ["column", "tubing_detectors"],
}
knauer_system_options_bypass_pre_column = {
    "sample_loop_volume": sample_loop_volume,
    "sample_loop_id": sample_loop_id,
    "bypass_units": ["tubing_pre_column", "column"],
}
knauer_system_options_bypass_column = {
    "sample_loop_volume": sample_loop_volume,
    "sample_loop_id": sample_loop_id,
    "bypass_units": ["column"],
}


# %% Optimization Options

metrics = ["NRMSE"]

optimizer_options_debugging = {
    "optimizer": "U_NSGA3",
    "n_cores": 8,
    "pop_size": 8,
    "n_max_gen": 16,
    "progress_frequency": 1,
}


optimizer_options = {
    "optimizer": "U_NSGA3",
}
